# OmniRT Realtime Avatar WebSocket

OmniRT Native Realtime Avatar WebSocket is the long-term protocol for model-agnostic digital-human streaming. It keeps the efficient `AUDI` / `VIDX` binary framing from the FlashTalk-compatible path, but uses an OmniRT session control plane with `session_id`, `trace_id`, structured errors, and metrics.

## Endpoint

```text
WS /v1/avatar/realtime
GET /v1/audio2video/models
WS /v1/audio2video/flashtalk
WS /v1/audio2video/wav2lip
```

`/v1/audio2video/flashtalk` and `/v1/audio2video/wav2lip` are the public
FlashTalk-compatible streaming paths for OpenTalking. `/v1/avatar/flashtalk`
and `/v1/avatar/wav2lip` remain compatibility aliases. `/v1/avatar/realtime`
is the model-agnostic control-plane protocol.

## Session create

```json
{
  "type": "session.create",
  "model": "soulx-flashtalk-14b",
  "backend": "auto",
  "inputs": {
    "image_b64": "<base64 png/jpeg>",
    "prompt": "A person is talking naturally."
  },
  "config": {
    "preset": "realtime",
    "seed": 9999,
    "wav2lip_postprocess_mode": false,
    "mouth_metadata": {
      "source_image_hash": "<sha256>",
      "animation": {
        "mouth_center": [0.5, 0.56],
        "mouth_rx": 0.06,
        "mouth_ry": 0.02,
        "outer_lip": [[0.45, 0.55], [0.5, 0.53], [0.55, 0.55]]
      }
    }
  }
}
```

Response:

```json
{
  "type": "session.created",
  "session_id": "avt_...",
  "trace_id": "trace_...",
  "audio": {
    "format": "pcm_s16le",
    "sample_rate": 16000,
    "channels": 1,
    "chunk_samples": 17920
  },
  "video": {
    "encoding": "jpeg-seq",
    "wire_magic": "VIDX",
    "fps": 25,
    "width": 416,
    "height": 704
  }
}
```

## Wav2Lip postprocess mode

Wav2Lip sessions accept `wav2lip_postprocess_mode` and optional
`mouth_metadata` in session config. When disabled, OmniRT keeps native Wav2Lip
output behavior. When enabled, the Wav2Lip runtime can use the supplied mouth
polygon to blend the generated mouth region back into the reference frame with
lower-lip coverage, feathering, and color matching.

The service default is off. It can be enabled process-wide with:

```bash
OMNIRT_WAV2LIP_POSTPROCESS_MODE=1 omnirt serve ...
```

The enhanced path exposes separate knobs for lower-lip coverage and jaw motion
transfer:

```bash
OMNIRT_WAV2LIP_LOWER_LIP_DYNAMIC_EXPAND=0.25
OMNIRT_WAV2LIP_ENABLE_JAW_MOTION_BLEND=1
OMNIRT_WAV2LIP_JAW_BLEND_ALPHA=0.22
OMNIRT_WAV2LIP_JAW_MASK_EXPAND_X=0.25
OMNIRT_WAV2LIP_JAW_MASK_EXPAND_Y=0.55
```

Jaw motion blending is disabled by default so enhanced mouth blending and jaw
motion can be A/B tested independently.

OpenTalking-compatible clients may also send the same fields in the `init`
message to `/v1/audio2video/wav2lip`.

## Audio and video chunks

Send audio:

```text
b"AUDI" + pcm_s16le
```

The server sends a metrics event, then a video binary payload:

```json
{"type": "metrics", "chunk_index": 1, "infer_ms": 0, "encode_ms": 0}
```

```text
b"VIDX" + uint32(frame_count) + repeated(uint32(jpeg_len) + jpeg_bytes)
```

## Control messages

```json
{"type": "session.cancel"}
{"type": "session.close"}
{"type": "ping"}
```

## Runtime 模式

v1 endpoint 保持 wire contract 稳定，不同部署可以选择不同 runtime：

| 模式 | 选择方式 | 说明 |
|---|---|---|
| `fake` | 默认，或 `OMNIRT_REALTIME_AVATAR_RUNTIME=fake` | 为协议测试和 CPU-stub demo 输出确定性 JPEG chunk |
| `proxy` | `OMNIRT_REALTIME_AVATAR_RUNTIME=proxy` + `OMNIRT_AVATAR_FLASHTALK_WS_URL` | 把 FlashTalk-compatible 路由转发到已有 WebSocket 服务 |
| `resident` | `OMNIRT_REALTIME_AVATAR_RUNTIME=resident` | 通过 OmniRT resident `soulx-flashtalk-14b` 执行路径渲染 chunk |

`GET /v1/audio2video/models` 会返回 `fallback_runtime`、`proxy` 或 `resident_runtime`，客户端可以据此区分协议测试模式和真实模型后端。
