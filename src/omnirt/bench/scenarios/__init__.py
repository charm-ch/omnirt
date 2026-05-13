"""Built-in benchmark scenarios."""

from __future__ import annotations

from omnirt.core.types import GenerateRequest

from ..runner import BenchScenario


_SCENARIOS = {
    "core_audio2video_flashtalk_smoke": BenchScenario(
        name="core_audio2video_flashtalk_smoke",
        request_template=GenerateRequest(
            task="audio2video",
            model="soulx-flashtalk-14b",
            backend="ascend",
            inputs={"image": "input.png", "audio": "input.wav"},
            config={"audio_encode_mode": "once", "max_chunks": 1},
        ),
        concurrency=1,
        total_requests=10,
        warmup=0,
    ),
    "core_realtime_avatar_flashtalk_chunk": BenchScenario(
        name="core_realtime_avatar_flashtalk_chunk",
        request_template=GenerateRequest(
            task="audio2video",
            model="soulx-flashtalk-14b",
            backend="ascend",
            inputs={"image": "input.png", "audio": "chunk.wav"},
            config={"audio_encode_mode": "once", "max_chunks": 1, "resident_autostart": True},
        ),
        concurrency=1,
        total_requests=10,
        warmup=1,
    ),
    "core_audio2video_flashhead_resident_warm": BenchScenario(
        name="core_audio2video_flashhead_resident_warm",
        request_template=GenerateRequest(
            task="audio2video",
            model="soulx-flashhead-1.3b",
            backend="ascend",
            inputs={"image": "input.png", "audio": "input.wav"},
            config={"audio_encode_mode": "once", "sample_steps": 2},
        ),
        concurrency=1,
        total_requests=3,
        warmup=1,
    ),
    "core_audio2video_liveact_resident_warm": BenchScenario(
        name="core_audio2video_liveact_resident_warm",
        request_template=GenerateRequest(
            task="audio2video",
            model="soulx-liveact-14b",
            backend="ascend",
            inputs={"image": "input.png", "audio": "input.wav"},
            config={"sample_steps": 1, "rank0_t5_only": True, "t5_cpu": False},
        ),
        concurrency=1,
        total_requests=3,
        warmup=1,
    ),
    "core_text2audio_cosyvoice_first_packet": BenchScenario(
        name="core_text2audio_cosyvoice_first_packet",
        request_template=GenerateRequest(
            task="text2audio",
            model="cosyvoice3-triton-trtllm",
            backend="cuda",
            inputs={"prompt": "你好，欢迎使用 OmniRT。", "audio": "reference.wav", "reference_text": "参考音色文本"},
            config={"sample_rate": 24000},
        ),
        concurrency=1,
        total_requests=10,
        warmup=1,
    ),
    "core_audio2text_sensevoice_batch": BenchScenario(
        name="core_audio2text_sensevoice_batch",
        request_template=GenerateRequest(
            task="audio2text",
            model="sensevoice-small",
            backend="auto",
            inputs={"audio": "speech.wav"},
            config={"language": "auto", "batch_size_s": 60},
        ),
        concurrency=1,
        total_requests=10,
        warmup=1,
    ),
    "adjacent_text2image_sdxl_concurrent4": BenchScenario(
        name="adjacent_text2image_sdxl_concurrent4",
        request_template=GenerateRequest(
            task="text2image",
            model="sdxl-base-1.0",
            backend="auto",
            inputs={"prompt": "a cinematic portrait of a traveler under neon rain"},
            config={"width": 1024, "height": 1024, "num_inference_steps": 30, "guidance_scale": 5.0},
        ),
        concurrency=4,
        total_requests=100,
        warmup=2,
        batch_window_ms=50,
        max_batch_size=4,
    ),
    "text2image_sdxl_concurrent4": BenchScenario(
        name="text2image_sdxl_concurrent4",
        request_template=GenerateRequest(
            task="text2image",
            model="sdxl-base-1.0",
            backend="auto",
            inputs={"prompt": "a cinematic portrait of a traveler under neon rain"},
            config={"width": 1024, "height": 1024, "num_inference_steps": 30, "guidance_scale": 5.0},
        ),
        concurrency=4,
        total_requests=100,
        warmup=2,
        batch_window_ms=50,
        max_batch_size=4,
    ),
}


def get_bench_scenario(name: str) -> BenchScenario:
    try:
        return _SCENARIOS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_SCENARIOS))
        raise ValueError(f"Unknown bench scenario {name!r}. Available: {known}") from exc


def list_bench_scenarios() -> list[str]:
    return sorted(_SCENARIOS)
