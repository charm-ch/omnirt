from __future__ import annotations

from omnirt.core.registry import get_model
from omnirt.models import ensure_registered
from omnirt.models.flashhead.pipeline import FlashHeadPipeline
from omnirt.models.liveact.pipeline import LiveActPipeline
from omnirt.workers import GrpcResidentWorkerProxy, PipelineResidentWorker


class FakeRuntime:
    name = "ascend"


def test_flashhead_uses_persistent_worker_surface() -> None:
    ensure_registered()
    spec = get_model("soulx-flashhead-1.3b", task="audio2video")

    worker = FlashHeadPipeline.create_persistent_worker(
        runtime=FakeRuntime(),
        model_spec=spec,
        config={},
        adapters=None,
    )

    assert spec.execution_mode == "persistent_worker"
    assert isinstance(worker, PipelineResidentWorker)


def test_liveact_uses_persistent_worker_surface() -> None:
    ensure_registered()
    spec = get_model("soulx-liveact-14b", task="audio2video")

    worker = LiveActPipeline.create_persistent_worker(
        runtime=FakeRuntime(),
        model_spec=spec,
        config={},
        adapters=None,
    )

    assert spec.execution_mode == "persistent_worker"
    assert isinstance(worker, PipelineResidentWorker)


def test_core_resident_workers_can_proxy_remote_targets() -> None:
    ensure_registered()
    flashhead = get_model("soulx-flashhead-1.3b", task="audio2video")
    liveact = get_model("soulx-liveact-14b", task="audio2video")

    assert isinstance(
        FlashHeadPipeline.create_persistent_worker(
            runtime=FakeRuntime(),
            model_spec=flashhead,
            config={"resident_target": "127.0.0.1:50071"},
            adapters=None,
        ),
        GrpcResidentWorkerProxy,
    )
    assert isinstance(
        LiveActPipeline.create_persistent_worker(
            runtime=FakeRuntime(),
            model_spec=liveact,
            config={"resident_target": "127.0.0.1:50072"},
            adapters=None,
        ),
        GrpcResidentWorkerProxy,
    )
