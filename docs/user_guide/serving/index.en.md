# Serving

OmniRT offers batch generation entry points and realtime avatar entry points. Batch entry points share the same `GenerateRequest` contract; realtime entry points serve the audio chunk -> video frames path for OpenTalking and new clients.

| Entry point | Best for | Page |
|---|---|---|
| **Python API** | embedding in existing Python apps, notebook experiments | [Python API](python_api.md) |
| **CLI** | scripted batches, one-off `validate` / `generate` | [CLI](cli.md) |
| **HTTP server** | microservice, multi-tenant, OpenAI-compatible API, Prometheus / OTLP hooks | [HTTP Server](http_server.md) |
| **FlashTalk WS** | compatibility for existing OpenTalking clients, using `AUDI` / `VIDX` binary frames | [FlashTalk WebSocket](flashtalk_ws.md) |
| **Realtime Avatar WS** | recommended OmniRT-native realtime avatar protocol for new integrations | [Realtime Avatar WebSocket](realtime_avatar_ws.md) |
| **Worker server** | gRPC execution node used by `serve --remote-worker` | [Distributed Serving](../deployment/distributed_serving.md) |

!!! tip "Recommended order"
    For offline generation, start in Python or CLI to validate the contract, then deploy the HTTP server for concurrency, batching, and policy tuning. For an existing realtime avatar frontend, use the FlashTalk-compatible WebSocket path to connect the service first.

Realtime avatar serving can run with three runtime modes:

- `fake`: deterministic local JPEG chunks for protocol tests and CPU-stub demos.
- `proxy`: pass the FlashTalk-compatible route through `OMNIRT_AVATAR_FLASHTALK_WS_URL`.
- `resident`: serve chunks through OmniRT's resident `soulx-flashtalk-14b` path via `OMNIRT_REALTIME_AVATAR_RUNTIME=resident`.
