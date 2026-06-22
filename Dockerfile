# Worker GPU do RunPod — Chatterbox Multilingual TTS (Resemble AI, MIT).
# 23 idiomas incl PT/ES/EN, voice cloning, language_id explicito.
#
# Base: PyTorch 2.7.1 + CUDA 12.8 (mesmo do omnivoice-worker pra suportar
# Blackwell + Hopper). Chatterbox roda em torch >= 2.4.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV HF_HOME=/app/hf \
    T3_MODEL=v3 \
    MP3_BITRATE=64k \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System libs (soundfile/ffmpeg) — base official PyTorch e mais slim que runpod
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Chatterbox + RunPod handler. --upgrade-strategy=only-if-needed pra nao mexer
# em torch/torchaudio ja matched pela imagem oficial.
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
      runpod chatterbox-tts soundfile numpy huggingface_hub

# Pre-baixa pesos (cold start nao precisa baixar). Roda em CPU mode pra evitar
# need de GPU no build host. Se falhar (modelo precisa CUDA), download cai no
# runtime — perde alguns segundos no 1o request mas funciona.
RUN python -c "\
import os; os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='600';\
try:\
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS;\
    ChatterboxMultilingualTTS.from_pretrained(device='cpu', t3_model='v3');\
    print('PRE-BAKE OK');\
except Exception as e:\
    print(f'PRE-BAKE SKIP: {type(e).__name__}: {str(e)[:200]}')\
" || true

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
