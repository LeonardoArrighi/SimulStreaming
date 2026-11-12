#!/usr/bin/env python3

import torch

# Disattiva forzatamente TF32 su matmul e cuDNN per evitare instabilità numerica sui Tensor Core di GB10 / CUDA 13
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

from simulstreaming_whisper import simulwhisper_args, simul_asr_factory
from simulstreaming.whisper.whisper_streaming.whisper_server import main_server

if __name__ == "__main__":
    main_server(simul_asr_factory, add_args=simulwhisper_args)