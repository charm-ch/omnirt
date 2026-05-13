"""Resident worker abstractions."""

from omnirt.workers.resident import ResidentModelWorker, ResidentWorkerHandle
from omnirt.workers.pipeline import PipelineResidentWorker
from omnirt.workers.remote import GrpcResidentWorkerProxy, ResidentWorkerService
from omnirt.workers.managed import ManagedGrpcResidentWorkerProxy

__all__ = [
    "GrpcResidentWorkerProxy",
    "ManagedGrpcResidentWorkerProxy",
    "PipelineResidentWorker",
    "ResidentModelWorker",
    "ResidentWorkerHandle",
    "ResidentWorkerService",
]
