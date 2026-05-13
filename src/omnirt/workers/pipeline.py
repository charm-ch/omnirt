"""Generic in-process resident worker for pipeline-backed models."""

from __future__ import annotations

from typing import Any

from omnirt.core.types import GenerateRequest, GenerateResult


class PipelineResidentWorker:
    """Keep a pipeline instance alive behind the resident-worker interface."""

    def __init__(self, *, pipeline_cls, runtime, model_spec, config, adapters) -> None:
        self.pipeline_cls = pipeline_cls
        self.runtime = runtime
        self.model_spec = model_spec
        self.config = dict(config or {})
        self.adapters = list(adapters or [])
        self.pipeline: Any | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.pipeline = self.pipeline_cls(
            runtime=self.runtime,
            model_spec=self.model_spec,
            adapters=self.adapters,
        )
        self._started = True

    def ready(self) -> bool:
        return self._started and self.pipeline is not None

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.start()
        if self.pipeline is None:
            raise RuntimeError("Pipeline resident worker did not initialize.")
        return self.pipeline.run(request)

    def shutdown(self) -> None:
        pipeline = self.pipeline
        if pipeline is not None:
            release = getattr(pipeline, "release", None)
            if callable(release):
                release()
        self.pipeline = None
        self._started = False
