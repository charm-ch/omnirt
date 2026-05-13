from pathlib import Path
import os

from tests.integration.conftest import require_local_model_dir, require_module
import pytest

from omnirt.api import generate


def _require_local_path(env_var: str, default: Path | None = None) -> str:
    raw = os.getenv(env_var)
    candidate = Path(raw).expanduser() if raw else default
    if candidate is None or not candidate.exists():
        pytest.skip(f"{env_var} does not exist locally: {candidate}")
    return str(candidate)


def test_flashhead_resident_ascend_smoke(tmp_path) -> None:
    require_module("torch_npu", "torch_npu is unavailable")

    repo_path = require_local_model_dir("OMNIRT_FLASHHEAD_REPO_PATH")
    repo_root = Path(repo_path)
    image_path = _require_local_path("OMNIRT_FLASHHEAD_IMAGE_PATH", repo_root / "examples" / "image.png")
    audio_path = _require_local_path("OMNIRT_FLASHHEAD_AUDIO_PATH", repo_root / "examples" / "audio.wav")
    python_executable = _require_local_path("OMNIRT_FLASHHEAD_PYTHON_EXECUTABLE")

    result = generate(
        {
            "task": "audio2video",
            "model": "soulx-flashhead-1.3b",
            "backend": "ascend",
            "inputs": {"image": image_path, "audio": audio_path},
            "config": {
                "repo_path": repo_path,
                "ckpt_dir": os.getenv("OMNIRT_FLASHHEAD_CKPT_DIR", "models/SoulX-FlashHead"),
                "wav2vec_dir": os.getenv("OMNIRT_FLASHHEAD_WAV2VEC_DIR", "models/chinese-wav2vec2-base"),
                "output_dir": str(tmp_path),
                "python_executable": python_executable,
                "ascend_env_script": os.getenv("OMNIRT_FLASHHEAD_ASCEND_ENV_SCRIPT"),
                "launcher": "torchrun",
                "nproc_per_node": int(os.getenv("OMNIRT_FLASHHEAD_NPROC_PER_NODE", "2")),
                "audio_encode_mode": "once",
                "sample_steps": 2,
            },
        }
    )

    assert result.outputs
    assert Path(result.outputs[0].path).exists()
    assert result.metadata.execution_mode == "persistent_worker"


def test_liveact_resident_ascend_smoke(tmp_path) -> None:
    require_module("torch_npu", "torch_npu is unavailable")

    repo_path = require_local_model_dir("OMNIRT_LIVEACT_REPO_PATH")
    repo_root = Path(repo_path)
    image_path = _require_local_path("OMNIRT_LIVEACT_IMAGE_PATH", repo_root / "examples" / "image" / "1.png")
    audio_path = _require_local_path("OMNIRT_LIVEACT_AUDIO_PATH", repo_root / "examples" / "audio" / "1.wav")
    python_executable = _require_local_path("OMNIRT_LIVEACT_PYTHON_EXECUTABLE")

    result = generate(
        {
            "task": "audio2video",
            "model": "soulx-liveact-14b",
            "backend": "ascend",
            "inputs": {"image": image_path, "audio": audio_path},
            "config": {
                "repo_path": repo_path,
                "ckpt_dir": os.getenv("OMNIRT_LIVEACT_CKPT_DIR", "models/SoulX-LiveAct-14B"),
                "wav2vec_dir": os.getenv("OMNIRT_LIVEACT_WAV2VEC_DIR", "models/chinese-wav2vec2-base"),
                "output_dir": str(tmp_path),
                "python_executable": python_executable,
                "ascend_env_script": os.getenv("OMNIRT_LIVEACT_ASCEND_ENV_SCRIPT"),
                "launcher": "torchrun",
                "nproc_per_node": int(os.getenv("OMNIRT_LIVEACT_NPROC_PER_NODE", "4")),
                "visible_devices": os.getenv("OMNIRT_LIVEACT_VISIBLE_DEVICES"),
                "text_cache_visible_devices": os.getenv("OMNIRT_LIVEACT_TEXT_CACHE_VISIBLE_DEVICES"),
                "sample_steps": 1,
                "rank0_t5_only": True,
                "t5_cpu": False,
            },
        }
    )

    assert result.outputs
    assert Path(result.outputs[0].path).exists()
    assert result.metadata.execution_mode == "persistent_worker"
    assert result.metadata.config_resolved["t5_cpu"] is False
