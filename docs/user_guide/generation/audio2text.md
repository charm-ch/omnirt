# 音频到文本

`audio2text` 用于离线语音识别，把一段音频转写为文本产物。第一版主线模型是 `sensevoice-small`，定位是补齐数字人链路里的语音理解入口。

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

## 配置

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_path` | `str` | `iic/SenseVoiceSmall` | FunASR 模型 id 或本地路径 |
| `language` | `str` | `auto` | 语言提示，例如 `auto` / `zh` / `en` |
| `use_itn` | `bool` | `true` | 是否启用 inverse text normalization |
| `batch_size_s` | `int` | `60` | 离线 batch 窗口 |
| `device` | `str` | `auto` | `auto` 会按后端解析到 CUDA / NPU / CPU |

需要真实运行时请安装 ASR 依赖：

```bash
pip install -e '.[asr]'
```
