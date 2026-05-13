# Audio to Text

`audio2text` runs offline speech recognition and exports a text artifact. The first core model is `sensevoice-small`, which adds the voice-understanding entrypoint for digital-human workflows.

## CLI

```bash
omnirt generate \
  --task audio2text \
  --model sensevoice-small \
  --audio speech.wav \
  --language auto \
  --backend auto \
  --output-dir outputs/asr
```

## Python API

```python
from omnirt import generate, requests

req = requests.audio2text(
    model="sensevoice-small",
    audio="speech.wav",
    language="auto",
)
result = generate(req)
print(result.outputs[0].path)
```

## Config

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_path` | `str` | `iic/SenseVoiceSmall` | FunASR model id or local path |
| `language` | `str` | `auto` | Language hint such as `auto` / `zh` / `en` |
| `use_itn` | `bool` | `true` | Enables inverse text normalization |
| `batch_size_s` | `int` | `60` | Offline batch window |
| `device` | `str` | `auto` | `auto` maps to CUDA / NPU / CPU from the selected backend |

Install the ASR extra before running the real model:

```bash
pip install -e '.[asr]'
```
