import os
import warnings
from pathlib import Path
import numpy as np
import torch

def is_onnx_available() -> bool:
    try:
        import onnxruntime
        return True
    except ImportError:
        return False

class OnnxSession:
    """Sessione condivisa ONNX per il modello Silero VAD (stateless)."""
    def __init__(self, path: str, force_onnx_cpu: bool = True):
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        if force_onnx_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)
        self.path = path
        self.sample_rates = [8000, 16000]

class OnnxWrapper:
    """Wrapper ONNX Runtime per Silero VAD con tracciamento di stato interno."""
    def __init__(self, session: OnnxSession):
        self._shared_session = session
        self.sample_rates = session.sample_rates
        self.reset_states()

    @property
    def session(self):
        return self._shared_session.session

    def reset_states(self, batch_size: int = 1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros((batch_size, 0), dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        if hasattr(x, "numpy"):
            x = x.numpy()
        elif hasattr(x, "cpu"):
            x = x.cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        num_samples = 512 if sr == 16000 else 256
        if x.shape[-1] != num_samples:
            raise ValueError(f"Dimensione chunk errata: {x.shape[-1]} (richiesti 512 campioni a 16kHz)")

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size or self._last_batch_size != batch_size or (self._last_sr and self._last_sr != sr):
            self.reset_states(batch_size)

        if self._context.shape[1] == 0:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x_with_context = np.concatenate([self._context, x], axis=1)
        ort_inputs = {
            'input': x_with_context,
            'state': self._state,
            'sr': np.array(sr, dtype=np.int64)
        }
        out, new_state = self.session.run(None, ort_inputs)
        self._state = new_state
        self._context = x_with_context[:, -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        return out[0, 0]


def load_silero_vad(model_path: str = None, force_onnx_cpu: bool = True):
    """Carica Silero VAD usando ONNX Runtime (preferito) o torch.hub come fallback."""
    # 1. Cerca il file ONNX
    candidate_paths = []
    if model_path:
        candidate_paths.append(Path(model_path))
    env_path = os.environ.get("SILERO_VAD_ONNX_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))

    here = Path(__file__).resolve().parent
    candidate_paths.append(here / "silero_vad_models" / "silero_vad.onnx")
    # Percorso in WhisperLiveKit se presente nel workspace RSI
    try:
        rsi_root = here.parents[3]
        candidate_paths.append(rsi_root / "WhisperLiveKit" / "whisperlivekit" / "silero_vad_models" / "silero_vad.onnx")
        candidate_paths.append(rsi_root / "SimulTranscription" / "silero_vad_models" / "silero_vad.onnx")
    except Exception:
        pass

    chosen_onnx = None
    for p in candidate_paths:
        if p.is_file():
            chosen_onnx = p
            break

    if is_onnx_available() and chosen_onnx is not None:
        try:
            sess = OnnxSession(str(chosen_onnx), force_onnx_cpu=force_onnx_cpu)
            wrapper = OnnxWrapper(sess)
            import logging
            logging.getLogger(__name__).info(f"[VAC] Silero VAD ONNX caricato da: {chosen_onnx}")
            return wrapper
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[VAC] Errore caricamento ONNX ({e}), fallback su torch.hub")

    # Fallback standard su torch.hub
    import logging
    logging.getLogger(__name__).info("[VAC] Caricamento Silero VAD via torch.hub...")
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    return model


# This is copied from silero-vad's vad_utils.py:
# https://github.com/snakers4/silero-vad/blob/94811cbe1207ec24bc0f5370b895364b8934936f/src/silero_vad/utils_vad.py#L398C1-L489C20
# (except changed defaults)

# Their licence is MIT, same as ours: https://github.com/snakers4/silero-vad/blob/94811cbe1207ec24bc0f5370b895364b8934936f/LICENSE

class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,  # makes sense on one recording that I checked
                 speech_pad_ms: int = 100             # same 
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)

        time_resolution: int (default - 1)
            time resolution of speech coordinates when requested as seconds
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, time_resolution)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)}

        return None

#######################
# because Silero now requires exactly 512-sized audio chunks 

import numpy as np
class FixedVADIterator(VADIterator):
    '''It fixes VADIterator by allowing to process any audio length, not only exactly 512 frames at once.
    If audio to be processed at once is long and multiple voiced segments detected, 
    then __call__ returns the start of the first segment, and end (or middle, which means no end) of the last segment. 
    '''

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([],dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        self.buffer = np.append(self.buffer, x) 
        ret = None
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if 'end' in r:
                    ret['end'] = r['end']  # the latter end
                if 'start' in r and 'end' in ret:  # there is an earlier start.
                    # Remove end, merging this segment with the previous one.
                    del ret['end']
        return ret if ret != {} else None

if __name__ == "__main__":
    # test/demonstrate the need for FixedVADIterator:

    import torch
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    vac = FixedVADIterator(model)
#   vac = VADIterator(model)  # the second case crashes with this

    # this works: for both
    audio_buffer = np.array([0]*(512),dtype=np.float32)
    vac(audio_buffer)

    # this crashes on the non FixedVADIterator with 
    # ops.prim.RaiseException("Input audio chunk is too short", "builtins.ValueError")
    audio_buffer = np.array([0]*(512-1),dtype=np.float32)
    vac(audio_buffer)
