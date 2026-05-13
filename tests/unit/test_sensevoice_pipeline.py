from __future__ import annotations

from pathlib import Path

from omnirt.backends.cpu_stub import CpuStubBackend
from omnirt.core.registry import get_model
from omnirt.core.types import GenerateRequest
from omnirt.core.validation import validate_request
from omnirt.models import ensure_registered
from omnirt.models.sensevoice.pipeline import SenseVoicePipeline


def test_sensevoice_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("sensevoice-small", task="audio2text")

    assert spec.task == "audio2text"
    assert spec.capabilities.artifact_kind == "text"
    assert spec.capabilities.chain_role == "voice-understanding"


def test_sensevoice_pipeline_exports_text(tmp_path: Path, monkeypatch) -> None:
    ensure_registered()
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fake wav")
    spec = get_model("sensevoice-small", task="audio2text")

    class FakeAutoModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, **kwargs):
            return [{"text": "hello omni"}]

    monkeypatch.setattr(SenseVoicePipeline, "_automodel_cls", staticmethod(lambda: FakeAutoModel))

    pipeline = SenseVoicePipeline(runtime=CpuStubBackend(), model_spec=spec)
    result = pipeline.run(
        GenerateRequest(
            task="audio2text",
            model="sensevoice-small",
            backend="cpu-stub",
            inputs={"audio": str(audio)},
            config={"output_dir": str(tmp_path), "language": "auto"},
        )
    )

    assert result.outputs[0].kind == "text"
    assert Path(result.outputs[0].path).read_text(encoding="utf-8") == "hello omni"
    assert result.metadata.backend == "cpu-stub"
    assert result.metadata.config_resolved["language"] == "auto"


def test_sensevoice_validation_allows_cpu_stub_execution(tmp_path: Path) -> None:
    ensure_registered()
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fake wav")

    validation = validate_request(
        GenerateRequest(
            task="audio2text",
            model="sensevoice-small",
            backend="cpu-stub",
            inputs={"audio": str(audio)},
        )
    )

    assert validation.ok is True
    assert not any("full generation still needs CUDA or Ascend" in issue.message for issue in validation.warnings)
