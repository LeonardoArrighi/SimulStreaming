#!/usr/bin/env python3
"""
mic_client.py
--------------
Client per SimulStreaming: cattura l'audio dal microfono del computer e lo
invia in streaming al server `simulstreaming_whisper_server.py`, mostrando
la trascrizione in tempo reale (bassa latenza).

Prerequisiti:
    pip install sounddevice numpy

Uso:
    1. Avvia prima il server, ad esempio:
       python simulstreaming_whisper_server.py --lan it --task transcribe --vac --min-chunk-size 1 --host localhost --port 43001

    2. Poi lancia questo client in un altro terminale:
       python mic_client.py --host localhost --port 43001

    3. Parla nel microfono: vedrai il testo confermato apparire in tempo reale.
       Premi Ctrl+C per interrompere.
"""

import argparse
import json
import queue
import socket
import sys
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000          # SimulStreaming/Whisper si aspetta 16kHz
CHANNELS = 1                 # mono
DTYPE = "int16"              # PCM S16_LE, come richiesto dal server
BLOCK_DURATION_SEC = 0.1     # dimensione dei blocchi catturati dal microfono


def list_input_devices():
    """Stampa i dispositivi audio di input disponibili (utile per --device)."""
    print("Dispositivi di input disponibili:")
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            print(f"  [{i}] {dev['name']} "
                  f"(default samplerate: {dev['default_samplerate']:.0f} Hz)")


class MicStreamer:
    """Cattura audio dal microfono e lo mette in una coda come bytes PCM16."""

    def __init__(self, device=None):
        self.device = device
        self.audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[mic] warning: {status}", file=sys.stderr)
        # indata è già int16 mono se configurato correttamente sotto
        self.audio_queue.put(bytes(indata))

    def start(self):
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION_SEC),
            device=self.device,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


def sender_thread(sock: socket.socket, mic: MicStreamer, stop_event: threading.Event):
    """Preleva i chunk audio dalla coda e li invia al server via TCP.

    NOTA: questo socket resta bloccante senza timeout. Non usare
    sock.settimeout() da nessun'altra parte su questo stesso oggetto:
    il timeout è un'impostazione dell'intero oggetto socket, quindi si
    applicherebbe anche a questa sendall() e potrebbe farla fallire
    spuriamente, interrompendo lo streaming. Lo stop cooperativo è
    ottenuto chiudendo il socket dal thread principale (vedi main()),
    che sblocca sia questo sendall() sia la recv() del receiver_thread.
    """
    while not stop_event.is_set():
        try:
            chunk = mic.audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            sock.sendall(chunk)
        except OSError as e:
            if not stop_event.is_set():
                print(f"\n[client] connessione al server persa durante l'invio: {e}", file=sys.stderr)
            stop_event.set()
            break


def receiver_thread(sock: socket.socket, stop_event: threading.Event):
    """Legge le righe JSONL restituite dal server e mostra la trascrizione.

    Socket bloccante, nessun settimeout qui (vedi nota in sender_thread).
    Lo stop viene rilevato quando recv() ritorna b'' (EOF, dopo che il
    thread principale ha fatto socket.shutdown) oppure solleva OSError.
    """
    buffer = b""
    confirmed_text = ""
    last_partial_len = 0

    while not stop_event.is_set():
        try:
            data = sock.recv(4096)
        except OSError as e:
            if not stop_event.is_set():
                print(f"\n[client] connessione al server chiusa: {e}", file=sys.stderr)
            stop_event.set()
            break

        if not data:
            if not stop_event.is_set():
                print("\n[client] il server ha chiuso la connessione.", file=sys.stderr)
            stop_event.set()
            break

        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "")
            is_final = obj.get("is_final", False)

            if text:
                confirmed_text += text
                # Riscrive la riga del terminale con il testo aggiornato
                display = confirmed_text.strip()
                pad = max(0, last_partial_len - len(display))
                print(f"\r{display}{' ' * pad}", end="", flush=True)
                last_partial_len = len(display)

            if is_final:
                print()  # vai a capo a fine frase/segmento vocale


def main():
    parser = argparse.ArgumentParser(description="Client microfono per SimulStreaming")
    parser.add_argument("--host", default="localhost", help="Host del server SimulStreaming")
    parser.add_argument("--port", type=int, default=43001, help="Porta del server SimulStreaming")
    parser.add_argument("--device", type=int, default=None,
                         help="Indice del dispositivo di input audio (vedi --list-devices)")
    parser.add_argument("--list-devices", action="store_true",
                         help="Elenca i dispositivi audio disponibili ed esce")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    print(f"[client] Connessione al server SimulStreaming {args.host}:{args.port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.host, args.port))
    except ConnectionRefusedError:
        print(f"[client] Impossibile connettersi a {args.host}:{args.port}. "
              f"Il server è avviato?", file=sys.stderr)
        sys.exit(1)

    print("[client] Connesso. Avvio cattura microfono...")
    mic = MicStreamer(device=args.device)
    stop_event = threading.Event()

    try:
        mic.start()
    except Exception as e:
        print(f"[client] Errore nell'apertura del microfono: {e}", file=sys.stderr)
        print("[client] Usa --list-devices per vedere i dispositivi disponibili "
              "e --device N per selezionarne uno.", file=sys.stderr)
        sock.close()
        sys.exit(1)

    t_send = threading.Thread(target=sender_thread, args=(sock, mic, stop_event), daemon=True)
    t_recv = threading.Thread(target=receiver_thread, args=(sock, stop_event), daemon=True)
    t_send.start()
    t_recv.start()

    print("[client] In ascolto... parla pure. Premi Ctrl+C per interrompere.\n")

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[client] Interruzione richiesta dall'utente.")
    finally:
        stop_event.set()
        mic.stop()
        # shutdown() sblocca immediatamente sia sendall() nel sender_thread
        # sia recv() nel receiver_thread, che sono bloccanti e senza
        # timeout. close() arriva dopo, quando i thread si sono già
        # fermati, per evitare race condition sul file descriptor.
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        t_send.join(timeout=2)
        t_recv.join(timeout=2)
        sock.close()
        print("[client] Terminato.")


if __name__ == "__main__":
    main()