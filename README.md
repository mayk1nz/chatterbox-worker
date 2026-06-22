# Chatterbox Multilingual Worker (RunPod Serverless)

Worker paralelo ao OmniVoice pra A/B de qualidade + custo. **MIT license** (livre
comercial). 23 idiomas incl PT/ES/EN.

## API do request

POST pro endpoint Runpod:
```json
{
  "input": {
    "texts": ["frase 1", "frase 2"],
    "ref_audio_b64": "<base64 mp3/wav 6-10s>",
    "ref_text": "(opcional) transcricao da ref pra ref_text",
    "language": "pt"     // language_id explicito (Chatterbox precisa)
  }
}
```

## Diferenças vs OmniVoice worker

| | OmniVoice | Chatterbox |
|---|---|---|
| `language` param | auto-detect do texto | **explícito** (`language_id`) |
| 23 idiomas | 600+ idiomas | 23 |
| Licença | Apache 2.0 | MIT |
| Cross-lingual quality | medio | melhor (per benchmark HF Emergent TTS) |
| Emocao | melhor (lider HF) | medio |

## Idiomas suportados (23)
ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, **pt**, ru, sv, sw, tr, zh

## Deploy

Mesma logica do omnivoice-worker:
1. Push pro GitHub → RunPod auto-build
2. Criar endpoint serverless apontando pra este repo, GPU 24GB, max=8
3. Adicionar endpoint ID no gateway via env `RUNPOD_ENDPOINT_ID_CHATTERBOX`
