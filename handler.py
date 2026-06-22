"""RunPod serverless worker — Chatterbox Multilingual TTS (Resemble AI).

Licenca MIT (livre comercial). 23 idiomas incl PT/ES/EN. Voice cloning a partir
de audio_prompt_path + language_id explicito (diferente do OmniVoice que
auto-detecta do texto).

Output: mp3 64k (mesmo formato do worker OmniVoice, pra compatibilidade no
gateway).
"""
import base64
import hashlib
import io
import os
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import runpod
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

T3_MODEL = os.environ.get("T3_MODEL", "v3")
DEVICE = os.environ.get("DEVICE", "cuda")
MP3_BITRATE = os.environ.get("MP3_BITRATE", "64k")

import sys, traceback
print(f"[boot] python={sys.version.split()[0]} torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[boot] cuda_device={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)
print(f"[boot] carregando ChatterboxMultilingualTTS (t3={T3_MODEL}, device={DEVICE}) ...", flush=True)
_t0 = time.time()
try:
    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE, t3_model=T3_MODEL)
    SR = int(model.sr)
    print(f"[boot] modelo carregado em {time.time() - _t0:.1f}s | sr={SR}Hz", flush=True)
except Exception as e:
    print(f"[boot] ERRO no model load: {type(e).__name__}: {e}", flush=True, file=sys.stderr)
    traceback.print_exc()
    sys.stderr.flush()
    sys.stdout.flush()
    raise

_ref_cache = {}  # sha256 -> path da ref no disco


def _get_ref_path(ref_audio_b64, ref_text):
    key = hashlib.sha256((ref_audio_b64[:1024] + "|" + ref_text).encode()).hexdigest()
    cached = _ref_cache.get(key)
    if cached and os.path.exists(cached):
        return cached
    path = os.path.join(tempfile.gettempdir(), f"chatterbox_ref_{key[:16]}.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(ref_audio_b64))
    _ref_cache[key] = path
    return path


def handler(job):
    inp = job["input"]
    texts = inp["texts"]
    ref_audio_b64 = inp["ref_audio_b64"]
    ref_text = inp.get("ref_text", "")
    # Chatterbox precisa do language_id explicito (default 'en')
    language = inp.get("language") or inp.get("lang") or "en"

    ref_path = _get_ref_path(ref_audio_b64, ref_text)

    pieces = []
    durations = []
    t0 = time.time()
    for txt in texts:
        wav = model.generate(
            txt,
            language_id=language,
            audio_prompt_path=ref_path,
        )
        # Chatterbox retorna tensor — converte pra numpy
        arr = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        pieces.append(arr)
        durations.append(round(len(arr) / SR, 3))
    gen_seconds = round(time.time() - t0, 3)

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    audio_seconds = round(len(audio) / SR, 3)

    # WAV -> MP3 via ffmpeg pipe (mesmo padrao do OmniVoice worker)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, SR, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-i", "pipe:0",
         "-c:a", "libmp3lame", "-b:a", MP3_BITRATE,
         "-f", "mp3", "pipe:1"],
        input=wav_buf.read(), capture_output=True, check=True,
    )
    mp3_bytes = proc.stdout

    return {
        "audio_b64": base64.b64encode(mp3_bytes).decode(),
        "audio_format": "mp3",
        "sample_rate": SR,
        "n_chunks": len(texts),
        "chunk_durations": durations,
        "gen_seconds": gen_seconds,
        "audio_seconds": audio_seconds,
        "rtf": round(gen_seconds / audio_seconds, 4) if audio_seconds else None,
        "model": f"chatterbox-multilingual-{T3_MODEL}",
    }


runpod.serverless.start({"handler": handler})
