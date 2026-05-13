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
    "seed": 9999
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

## Runtime modes

The v1 endpoint keeps the wire contract stable while the backing runtime can be selected per deployment:

| Mode | Selection | Notes |
|---|---|---|
| `fake` | default, or `OMNIRT_REALTIME_AVATAR_RUNTIME=fake` | deterministic JPEG chunks for protocol tests and CPU-stub demos |
| `proxy` | `OMNIRT_REALTIME_AVATAR_RUNTIME=proxy` plus `OMNIRT_AVATAR_FLASHTALK_WS_URL` | forwards the FlashTalk-compatible route to an existing WebSocket service |
| `resident` | `OMNIRT_REALTIME_AVATAR_RUNTIME=resident` | renders chunks through OmniRT's resident `soulx-flashtalk-14b` execution path |

`GET /v1/audio2video/models` reports the active reason as `fallback_runtime`, `proxy`, or `resident_runtime` so clients can distinguish protocol-only test mode from a model-backed deployment.
