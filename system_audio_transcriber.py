#!/usr/bin/env python3
"""
Uso:
    1. Avvia un'istanza di SimulStreaming:
       python simulstreaming_whisper_server.py --lan it --task transcribe --vac --min-chunk-size 1 --model_path ./small.pt --audio_max_len 10 --frame_threshold 15 --host localhost --port 43001

    2. Elenca gli altoparlanti disponibili (per scegliere quello giusto se ne hai più di uno):
       python system_audio_transcriber.py --list-devices
s
    3. Avvia la cattura e la pagina di trascrizione:
       python system_audio_transcriber.py --whisper-port 43001
"""

import argparse
import asyncio
import functools
import http.server
import json
import queue
import socket
import sys
import threading
import time
import warnings
import webbrowser
from pathlib import Path

import numpy as np
import soundcard as sc
import websockets

# "data discontinuity in recording" è un warning noto e per lo più benigno
# della libreria soundcard su Windows/WASAPI: viene emesso quando il driver
# audio segnala un buco nel flusso (tipicamente durante brevi silenzi o
# transizioni), non un errore fatale. Lo silenziamo per non intasare il log;
# la mitigazione vera è il blocksize più ampio impostato sotto.
warnings.filterwarnings("ignore", message="data discontinuity in recording")

SAMPLE_RATE = 16000        # richiesto da SimulStreaming/Whisper
CHUNK_FRAMES = 1600        # 0.1s a 16kHz per lettura, bassa latenza
RECORDER_BLOCKSIZE = CHUNK_FRAMES * 4  # buffer interno più ampio, riduce i discontinuity
HTML_FILE = "transcript_viewer.html"

CONNECTED_CLIENTS: "set[websockets.WebSocketServerProtocol]" = set()


# --------------------------------------------------------------------------
# Cattura audio di sistema (loopback)
# --------------------------------------------------------------------------

def list_speakers():
    print("Altoparlanti / dispositivi di output disponibili:")
    for spk in sc.all_speakers():
        marker = " (default)" if spk.name == sc.default_speaker().name else ""
        print(f"  - {spk.name}{marker}")


def resolve_loopback_microphone(speaker_substring: str | None):
    """Trova il dispositivo 'loopback' corrispondente all'altoparlante scelto.

    Su Windows/WASAPI, ogni altoparlante ha un dispositivo di 'ingresso
    ombra' che intercetta esattamente ciò che sta riproducendo: è quello
    che vogliamo, non il microfono fisico.
    """
    if speaker_substring:
        candidates = [s for s in sc.all_speakers() if speaker_substring.lower() in s.name.lower()]
        if not candidates:
            raise RuntimeError(
                f"Nessun altoparlante trovato con nome contenente '{speaker_substring}'. "
                f"Usa --list-devices per vedere quelli disponibili."
            )
        speaker = candidates[0]
    else:
        speaker = sc.default_speaker()

    loopback_mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
    return speaker, loopback_mic


class LoopbackStreamer:
    """Cattura l'audio di sistema e lo mette in coda come bytes PCM16 mono 16kHz."""

    def __init__(self, loopback_mic, stop_event: threading.Event):
        self.loopback_mic = loopback_mic
        self.stop_event = stop_event
        self.audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self._thread = None

    def _capture_loop(self):
        # NOTA: su Windows/WASAPI, registrare un solo canale (channels=1)
        # e' noto produrre audio corrotto in alcune versioni della libreria
        # soundcard. Per questo registriamo in stereo (2 canali) e facciamo
        # noi il downmix a mono in software, che e' l'approccio robusto
        # raccomandato dagli stessi manutentori della libreria.
        try:
            with self.loopback_mic.recorder(
                samplerate=SAMPLE_RATE, channels=2, blocksize=RECORDER_BLOCKSIZE
            ) as rec:
                while not self.stop_event.is_set():
                    data = rec.record(numframes=CHUNK_FRAMES)  # shape (N, 2), float32 [-1, 1]
                    mono = data.mean(axis=1)
                    mono = np.clip(mono, -1.0, 1.0)
                    pcm16 = (mono * 32767.0).astype(np.int16)
                    self.audio_queue.put(pcm16.tobytes())
        except Exception as e:
            print(f"[audio] Errore nella cattura dell'audio di sistema: {e}", file=sys.stderr)
            self.stop_event.set()

    def start(self):
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)


# --------------------------------------------------------------------------
# Collegamento al server SimulStreaming (stesso protocollo del client microfono)
# --------------------------------------------------------------------------

def sender_thread(sock: socket.socket, streamer: LoopbackStreamer, stop_event: threading.Event):
    """Invia i chunk audio catturati al server SimulStreaming.

    Socket bloccante senza timeout: lo stop cooperativo avviene chiudendo
    il socket dal thread principale (sock.shutdown), non tramite timeout
    condivisi, per evitare di interrompere spuriamente l'invio (vedi la
    stessa nota nel client del microfono).
    """
    while not stop_event.is_set():
        try:
            chunk = streamer.audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            sock.sendall(chunk)
        except OSError as e:
            if not stop_event.is_set():
                print(f"\n[whisper] connessione persa durante l'invio: {e}", file=sys.stderr)
            stop_event.set()
            break


def receiver_thread(sock: socket.socket, stop_event: threading.Event, out_queue: "queue.Queue[str]"):
    """Legge le righe JSONL dal server e le inoltra (come testo grezzo) alla coda
    che verrà trasmessa via WebSocket al browser."""
    buffer = b""
    while not stop_event.is_set():
        try:
            data = sock.recv(4096)
        except OSError as e:
            if not stop_event.is_set():
                print(f"\n[whisper] connessione chiusa: {e}", file=sys.stderr)
            stop_event.set()
            break

        if not data:
            if not stop_event.is_set():
                print("\n[whisper] il server ha chiuso la connessione.", file=sys.stderr)
            stop_event.set()
            break

        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)  # valida che sia JSON prima di inoltrarlo
            except json.JSONDecodeError:
                continue
            out_queue.put(line.decode("utf-8", errors="replace"))


# --------------------------------------------------------------------------
# Server HTTP statico (serve transcript_viewer.html) + server WebSocket
# --------------------------------------------------------------------------

def start_http_server(directory: Path, port: int, stop_event: threading.Event):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("localhost", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def _watch_stop():
        stop_event.wait()
        httpd.shutdown()

    threading.Thread(target=_watch_stop, daemon=True).start()
    return httpd


async def ws_handler(websocket):
    CONNECTED_CLIENTS.add(websocket)
    print(f"[web] client connesso ({len(CONNECTED_CLIENTS)} totali)")
    try:
        async for _ in websocket:
            pass  # non ci aspettiamo messaggi in ingresso dal browser
    finally:
        CONNECTED_CLIENTS.discard(websocket)
        print(f"[web] client disconnesso ({len(CONNECTED_CLIENTS)} totali)")


async def pump_to_websockets(sync_queue: "queue.Queue[str]", stop_event: threading.Event):
    loop = asyncio.get_running_loop()
    while not stop_event.is_set():
        try:
            message = await loop.run_in_executor(None, sync_queue.get, True, 0.5)
        except queue.Empty:
            continue
        if CONNECTED_CLIENTS:
            await asyncio.gather(
                *(ws.send(message) for ws in list(CONNECTED_CLIENTS)),
                return_exceptions=True,
            )


async def run_async_part(ws_host, ws_port, sync_queue, stop_event):
    async with websockets.serve(ws_handler, ws_host, ws_port):
        print(f"[web] server WebSocket in ascolto su ws://{ws_host}:{ws_port}")
        await pump_to_websockets(sync_queue, stop_event)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trascrizione in tempo reale dell'audio di sistema")
    parser.add_argument("--whisper-host", default="localhost", help="Host del server SimulStreaming")
    parser.add_argument("--whisper-port", type=int, default=43002,
                         help="Porta del server SimulStreaming dedicato all'audio di sistema")
    parser.add_argument("--ws-port", type=int, default=8765, help="Porta del server WebSocket")
    parser.add_argument("--http-port", type=int, default=8000, help="Porta della pagina web locale")
    parser.add_argument("--speaker", default=None,
                         help="Sottostringa del nome dell'altoparlante da cui catturare (default: quello di sistema)")
    parser.add_argument("--list-devices", action="store_true", help="Elenca gli altoparlanti ed esce")
    parser.add_argument("--no-browser", action="store_true", help="Non aprire automaticamente il browser")
    args = parser.parse_args()

    if args.list_devices:
        list_speakers()
        return

    script_dir = Path(__file__).resolve().parent
    html_path = script_dir / HTML_FILE
    if not html_path.exists():
        print(f"[avviso] {HTML_FILE} non trovato in {script_dir}. "
              f"Assicurati che sia nella stessa cartella di questo script.", file=sys.stderr)

    try:
        speaker, loopback_mic = resolve_loopback_microphone(args.speaker)
    except RuntimeError as e:
        print(f"[errore] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[audio] Cattura loopback da: {speaker.name}")

    print(f"[whisper] Connessione al server SimulStreaming {args.whisper_host}:{args.whisper_port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.whisper_host, args.whisper_port))
    except ConnectionRefusedError:
        print(f"[errore] Impossibile connettersi a {args.whisper_host}:{args.whisper_port}. "
              f"Hai avviato una seconda istanza di simulstreaming_whisper_server.py su questa porta?",
              file=sys.stderr)
        sys.exit(1)
    print("[whisper] Connesso.")

    stop_event = threading.Event()
    transcript_queue: "queue.Queue[str]" = queue.Queue()

    streamer = LoopbackStreamer(loopback_mic, stop_event)
    streamer.start()

    t_send = threading.Thread(target=sender_thread, args=(sock, streamer, stop_event), daemon=True)
    t_recv = threading.Thread(target=receiver_thread, args=(sock, stop_event, transcript_queue), daemon=True)
    t_send.start()
    t_recv.start()

    start_http_server(script_dir, args.http_port, stop_event)
    page_url = f"http://localhost:{args.http_port}/{HTML_FILE}?ws_port={args.ws_port}"
    print(f"[web] Pagina di trascrizione: {page_url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(page_url)).start()

    print("[avvio] In ascolto sull'audio di sistema... avvia un video o un audio qualsiasi.")
    print("[avvio] Premi Ctrl+C per interrompere.\n")

    try:
        asyncio.run(run_async_part("localhost", args.ws_port, transcript_queue, stop_event))
    except KeyboardInterrupt:
        print("\n[main] Interruzione richiesta dall'utente.")
    finally:
        stop_event.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        t_send.join(timeout=2)
        t_recv.join(timeout=2)
        streamer.join(timeout=2)
        sock.close()
        print("[main] Terminato.")


if __name__ == "__main__":
    main()