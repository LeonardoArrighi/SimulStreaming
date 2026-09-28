from simulstreaming.whisper.whisper_streaming.base import OnlineProcessorInterface, ASRBase
import argparse

import sys
import os
import logging
import torch
import numpy as np

from simulstreaming.whisper.simul_whisper.config import AlignAttConfig
from simulstreaming.whisper.simul_whisper.simul_whisper import PaddedAlignAttWhisper

logger = logging.getLogger(__name__)

def simulwhisper_args(parser):
    group = parser.add_argument_group('Whisper / Faster-Whisper arguments')
    group.add_argument('--backend', type=str, default='simulstreaming',
                        choices=['simulstreaming', 'faster-whisper', 'ctranslate2'],
                        help='ASR backend: "simulstreaming" (policy AlignAtt) oppure "faster-whisper" / "ctranslate2" (full CTranslate2 con quantizzazione INT8).')
    group.add_argument('--encoder-backend', type=str, default='faster-whisper',
                        choices=['faster-whisper', 'whisper', 'ctranslate2'],
                        help='Backend encoder per SimulStreaming: "faster-whisper" (accelerazione CTranslate2 INT8/FP16) '
                             'oppure "whisper" (PyTorch vanilla). Default: faster-whisper.')
    group.add_argument('--compute-type', '--compute_type', type=str, default='int8_float16',
                        choices=['int8_float16', 'int8', 'int8_bfloat16', 'float16', 'float32', 'auto'],
                        help='Tipo di quantizzazione per Faster-Whisper / CTranslate2 (default: int8_float16). '
                             'Dimezza la memoria VRAM e velocizza i calcoli di 2x-4x.')
    group.add_argument('--model_path', type=str, default='./medium.pt', 
                        help='The file path to the Whisper .pt model or model size for faster-whisper (e.g. medium, large-v3).')
    group.add_argument("--beams","-b", type=int, default=1, help="Number of beams for beam search decoding. If 1, GreedyDecoder is used.")
    group.add_argument("--decoder",type=str, default=None, help="Override automatic selection of beam or greedy decoder. "
                        "If beams > 1 and greedy: invalid.")

    group = parser.add_argument_group('Audio buffer')
    group.add_argument('--audio_max_len', type=float, default=30.0, 
                        help='Max length of the audio buffer, in seconds.')
    group.add_argument('--audio_min_len', type=float, default=0.0, 
                        help='Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.')


    group = parser.add_argument_group('AlignAtt argument')
    group.add_argument('--frame_threshold', type=int, default=25, 
                        help='Threshold for the attention-guided decoding. The AlignAtt policy will decode only ' \
                            'until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model. ')

    group = parser.add_argument_group('Truncation of the last decoded word (from Simul-Whisper)')
    group.add_argument('--cif_ckpt_path', type=str, default=None, 
                        help='The file path to the Simul-Whisper\'s CIF model checkpoint that detects whether there is' \
                        'end of word at the end of the chunk. If not, the last decoded space-separated word is truncated ' \
                        'because it is often wrong -- transcribing a word in the middle.' \
                        'The CIF model adapted for the Whisper model version should be used. ' \
                        'Find the models in https://github.com/backspacetg/simul_whisper/tree/main/cif_models . ' \
                        'Note that there is no model for large-v3.')
    group.add_argument("--never_fire", action=argparse.BooleanOptionalAction, default=False, 
                       help="Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. " \
                       ". If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. " \
                        "Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed.")

    group = parser.add_argument_group("Prompt and context")
    group.add_argument("--init_prompt",type=str, default=None, help="Init prompt for the model. It should be in the target language.")
    group.add_argument("--static_init_prompt",type=str, default=None, help="Do not scroll over this text. It can contain terminology that should be relevant over all document.")
    group.add_argument("--max_context_tokens",type=int, default=None, help="Max context tokens for the model. Default is 0.")


def simul_asr_factory(args):
    logger.setLevel(args.log_level)

    # 1. Se richiesto backend full faster-whisper (CTranslate2)
    backend = getattr(args, "backend", "simulstreaming")
    if backend in ("faster-whisper", "ctranslate2"):
        asr = FasterWhisperASR(
            language=args.lan,
            model_path=args.model_path,
            compute_type=getattr(args, "compute_type", "int8_float16"),
            task=args.task,
            init_prompt=getattr(args, "init_prompt", None) or getattr(args, "static_init_prompt", None),
            beam_size=args.beams,
        )
        return asr, FasterWhisperOnline(asr, min_chunk_size=args.min_chunk_size)

    # 2. Altrimenti SimulStreaming (AlignAtt policy con supporto encoder Faster-Whisper INT8)
    decoder = args.decoder
    if args.beams > 1:
        if decoder == "greedy":
            raise ValueError("Invalid 'greedy' decoder type for beams > 1. Use 'beam'.")
        elif decoder is None or decoder == "beam":
            decoder = "beam"
        else:
            raise ValueError("Invalid decoder type. Use 'beam' or 'greedy'.")
    else:
        if decoder is None:
            decoder = "greedy"
        elif decoder not in ("beam","greedy"):
            raise ValueError("Invalid decoder type. Use 'beam' or 'greedy'.")
        # else: it is greedy or beam, that's ok 
    
    a = { v:getattr(args, v) for v in ["model_path", "cif_ckpt_path", "frame_threshold", "audio_min_len", "audio_max_len", "beams", "task",
                                       "never_fire", 'init_prompt', 'static_init_prompt', 'max_context_tokens', "logdir",
                                       "encoder_backend", "compute_type"
                                       ]}
    a["language"] = args.lan
    a["segment_length"] = args.min_chunk_size
    a["decoder_type"] = decoder

    if args.min_chunk_size >= args.audio_max_len:
        raise ValueError("min_chunk_size must be smaller than audio_max_len")
    if args.audio_min_len > args.audio_max_len:
        raise ValueError("audio_min_len must be smaller than audio_max_len")
    logger.info(f"Arguments: {a}")
    asr = SimulWhisperASR(**a)
    return asr, SimulWhisperOnline(asr)

class SimulWhisperASR(ASRBase):
    
    sep = " "

    def __init__(self, language, model_path, cif_ckpt_path, frame_threshold, audio_max_len, audio_min_len, segment_length, beams, task, 
                 decoder_type, never_fire, init_prompt, static_init_prompt, max_context_tokens, logdir,
                 encoder_backend="faster-whisper", compute_type="int8_float16"):
        cfg = AlignAttConfig(
            model_path=model_path, 
            segment_length=segment_length,
            frame_threshold=frame_threshold,
            language=language,
            audio_max_len=audio_max_len, 
            audio_min_len=audio_min_len,
            cif_ckpt_path=cif_ckpt_path,
            decoder_type=decoder_type, #"greedy" if beams==1 else "beam",
            beam_size=beams,
            task=task,
            never_fire=never_fire,
            init_prompt=init_prompt,
            max_context_tokens=max_context_tokens,
            static_init_prompt=static_init_prompt,
            logdir=logdir,
            encoder_backend=encoder_backend,
            compute_type=compute_type,
        )
        logger.info(f"Language: {language} | Encoder: {encoder_backend} ({compute_type})")
        self.model = PaddedAlignAttWhisper(cfg)

    def transcribe(self, audio, init_prompt=""):
        logger.info("SimulWhisperASR's transcribe() should not be used. It's here only temporarily." \
        "Instead, use SimulWhisperOnline.process_iter().")
        raise NotImplementedError("Use SimulWhisperOnline.process_iter() instead of transcribe().")

    def warmup(self, audio, init_prompt=""):
        self.model.insert_audio(audio)
        self.model.infer(True)
        self.model.refresh_segment(complete=True)
    
    def use_vad(self):
        print("VAD not implemented",file=sys.stderr)

    def set_translate_task(self):
        # this is not used. Translate task is set another way.
        pass


class SimulWhisperOnline(OnlineProcessorInterface):

    def __init__(self, asr):
        self.model = asr.model
        self.file = None
        self.init()

    def init(self, offset=None):
        self.audio_chunks = []
        if offset is not None:
            self.offset = offset
        else:
            self.offset = 0
        self.is_last = False
        self.beg = self.offset
        self.end = self.offset

        self.audio_bufer_offset = self.offset
        self.last_ts = -1
        self.model.refresh_segment(complete=True)

        self.unicode_buffer = []  # hide incomplete unicode character for the next iteration

    def insert_audio_chunk(self, audio):
        self.audio_chunks.append(torch.from_numpy(audio))

    def timestamped_text(self, tokens, generation):
        if not generation:
            return []

        pr = generation["progress"]
        if "result" not in generation or self.unicode_buffer != []:
            split_words, split_tokens = self.model.tokenizer.split_to_word_tokens(tokens)
        else:
            split_words, split_tokens = generation["result"]["split_words"], generation["result"]["split_tokens"]

        frames = [p["most_attended_frames"][0] for p in pr]
        if self.unicode_buffer != []:
            a = [frames[0]] * len(self.unicode_buffer)
            frames = a + frames
            
        tokens = tokens.copy()
        ret = []
        for sw,st in zip(split_words,split_tokens):
            b = None
            for stt in st:
                t,f = tokens.pop(0), frames.pop(0)
                if t != stt:
                    raise ValueError(f"Token mismatch: {t} != {stt} at frame {f}.")
                if b is None:
                    b = f
            e = f
            out = {
                'start': b * 0.02 + self.audio_bufer_offset,
                'end': e * 0.02 + self.audio_bufer_offset,
                'text': sw,
                'tokens': st
                }
            ret.append(out)
            logger.debug(f"TS-WORD-INFO: {out}")
        return ret

    def hide_incomplete_unicode(self, tokens):
        """Sometimes, the last token is an imcomplete unicode character, e.g. a part of "ň" or "ř".
        Without this, the outputs can end with '�' = Unicode Replacement Character, and the next output also
        starts with '�'.
        This function hides the last incomplete unicode character and adds it in the next iteration.
        """
        if self.unicode_buffer != []:
            logger.debug(f"Hiding incomplete unicode character: {self.unicode_buffer}")
            tokens = self.unicode_buffer + tokens
            self.unicode_buffer = []  # clear the buffer after processing
        chars, _ = self.model.tokenizer.split_tokens_on_unicode(tokens)
        if len(chars) > 0 and chars[-1].endswith('�'):
            self.unicode_buffer = tokens[-1:]  # keep the last incomplete unicode character
            logger.debug(f"Hiding incomplete unicode character: {tokens[-1:]}")
            return tokens[:-1]  # remove the last token, which is incomplete unicode character
        return tokens

    def process_iter(self):
        if len(self.audio_chunks) == 0:
            audio = None
        else:
            audio = torch.cat(self.audio_chunks, dim=0)
            if audio.shape[0] == 0:
                audio = None
            else:
                self.end += audio.shape[0] / self.SAMPLING_RATE
        self.audio_chunks = []
        self.audio_bufer_offset += self.model.insert_audio(audio)
        tokens, generation_progress = self.model.infer(is_last=self.is_last)

        tokens = self.hide_incomplete_unicode(tokens)

        text = self.model.tokenizer.decode(tokens)
        if len(text) == 0:
            return {}
        
        # word-level timestamps
        ts_words = self.timestamped_text(tokens, generation_progress)
        
        self.beg = min(word['start'] for word in ts_words)  # it should be this
        self.beg = max(self.beg, self.last_ts + 0.001)  # but let's create the timestamps non-decreasing -- at least last beg + 1
        if self.is_last:
            e = self.end
        else:
            e = max(word['end'] for word in ts_words)
        e = max(e, self.beg + 0.001)

        self.last_ts = e

        # return (self.beg,e,text)
        return {
            'start': self.beg,
            'end': e,
            'text': text,
            'tokens': tokens,
            'words': ts_words
        }

    def finish(self):
        logger.info("Finish")
        self.is_last = True
        o = self.process_iter()
        self.is_last = False
        self.model.refresh_segment(complete=True)
        return o


class FasterWhisperASR(ASRBase):
    """Backend nativo Faster-Whisper (CTranslate2) con supporto INT8."""
    sep = ""

    def __init__(self, language, model_path, compute_type="int8_float16", task="transcribe",
                 init_prompt=None, beam_size=1):
        from faster_whisper import WhisperModel

        self.language = language if language != "auto" else None
        self.task = task
        self.init_prompt = init_prompt
        self.beam_size = beam_size
        self.compute_type = compute_type

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = os.path.basename(model_path).replace(".pt", "") if model_path.endswith(".pt") else model_path
        download_root = os.path.dirname(os.path.abspath(model_path)) if model_path.endswith(".pt") and os.path.isdir(os.path.dirname(os.path.abspath(model_path))) else None

        logger.info(f"[faster-whisper] Caricamento modello completo CTranslate2: '{model_name}' (device={device}, compute_type={compute_type})")
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        logger.info(f"[faster-whisper] Backend CTranslate2 (INT8) caricato con successo!")

    def warmup(self, audio, init_prompt=""):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        self.model.transcribe(audio, language=self.language, beam_size=1)

    def set_translate_task(self):
        self.task = "translate"


class FasterWhisperOnline(OnlineProcessorInterface):
    """Processore online per Faster-Whisper (CTranslate2) con quantizzazione INT8."""

    def __init__(self, asr: FasterWhisperASR, min_chunk_size=0.5):
        self.asr = asr
        self.model = asr.model
        self.min_chunk_size = min_chunk_size
        self.init()

    def init(self, offset=None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.offset = offset if offset is not None else 0.0
        self.last_ts = self.offset
        self.is_last = False

    def insert_audio_chunk(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def process_iter(self):
        buf_len_sec = len(self.audio_buffer) / self.SAMPLING_RATE
        if buf_len_sec < self.min_chunk_size and not self.is_last:
            return {}

        segments, _ = self.model.transcribe(
            self.audio_buffer,
            language=self.asr.language,
            task=self.asr.task,
            initial_prompt=self.asr.init_prompt,
            beam_size=self.asr.beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        words = []
        full_text = []
        for s in segments:
            full_text.append(s.text)
            if s.words:
                for w in s.words:
                    words.append({
                        "start": float(self.offset + w.start),
                        "end": float(self.offset + w.end),
                        "text": w.word
                    })

        text = "".join(full_text).strip()
        if not text:
            return {}

        start_ts = float(words[0]["start"]) if words else float(self.offset)
        end_ts = float(words[-1]["end"]) if words else float(self.offset + buf_len_sec)

        # Se il buffer supera 20 secondi, conserva gli ultimi 5 secondi per non saturare la memoria
        if buf_len_sec > 20.0 and words:
            trim_sec = max(0.0, buf_len_sec - 5.0)
            trim_samples = int(trim_sec * self.SAMPLING_RATE)
            self.audio_buffer = self.audio_buffer[trim_samples:]
            self.offset += trim_sec

        return {
            "start": start_ts,
            "end": end_ts,
            "text": text,
            "words": words,
            "is_final": self.is_last
        }

    def finish(self):
        self.is_last = True
        res = self.process_iter()
        self.is_last = False
        self.init()
        return res


if __name__ == "__main__":

    from simulstreaming.whisper.whisper_streaming.whisper_online_main import main_simulation_from_file
    main_simulation_from_file(simul_asr_factory, add_args=simulwhisper_args)