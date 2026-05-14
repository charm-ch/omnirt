"""Runtime manifest loading."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from omnirt.runtime.paths import expand_path_template, project_root


@dataclass(frozen=True)
class RuntimeManifest:
    name: str
    device: str
    profile: str
    manifest_path: Path
    repo_url: str
    checkpoint_url: str | None
    wav2vec_repo_id: str | None
    runtime_dir: Path
    repo_dir: Path
    requirements_file: Path
    env_script: Path | None
    server_path: Path
    ckpt_dir: str
    wav2vec_dir: str
    python_version: str
    nproc_per_node: int
    pip_index_url: str
    pip_extra_index_url: str | None
    hf_endpoint: str
    repo_marker_dir: str | None

    @property
    def venv_dir(self) -> Path:
        return self.runtime_dir / "venv"

    @property
    def python_path(self) -> Path:
        return self.venv_dir / "bin" / "python"

    @property
    def torchrun_path(self) -> Path:
        return self.venv_dir / "bin" / "torchrun"

    @property
    def activate_path(self) -> Path:
        return self.venv_dir / "bin" / "activate"

    @property
    def resolved_ckpt_dir(self) -> Path:
        path = Path(self.ckpt_dir).expanduser()
        return path if path.is_absolute() else self.repo_dir / path

    @property
    def resolved_wav2vec_dir(self) -> Path:
        path = Path(self.wav2vec_dir).expanduser()
        return path if path.is_absolute() else self.repo_dir / path

    def with_overrides(
        self,
        *,
        repo_dir: str | Path | None = None,
        ckpt_dir: str | Path | None = None,
        wav2vec_dir: str | Path | None = None,
    ) -> "RuntimeManifest":
        values: dict[str, object] = {}
        if repo_dir is not None:
            values["repo_dir"] = Path(repo_dir).expanduser().resolve()
        if ckpt_dir is not None:
            values["ckpt_dir"] = str(Path(ckpt_dir).expanduser())
        if wav2vec_dir is not None:
            values["wav2vec_dir"] = str(Path(wav2vec_dir).expanduser())
        return replace(self, **values)


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"runtime manifest field {key!r} must be a mapping")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime manifest field {key!r} must be a non-empty string")
    return value.strip()


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"runtime manifest field {key!r} must be a string")
    value = value.strip()
    return value or None


def _default_repo_marker_dir(name: str) -> str | None:
    if name == "flashtalk":
        return "flash_talk"
    return None


def manifest_path(name: str, device: str) -> Path:
    candidates = [
        project_root() / "model_backends" / name / f"runtime.{device}.yaml",
        project_root() / "model_backends" / name / f"runtime.{device}_910b.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No runtime manifest found for {name!r} on {device!r}")


def load_manifest(name: str, device: str = "ascend") -> RuntimeManifest:
    path = manifest_path(name, device)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    manifest_name = _required_string(data, "name")
    manifest_device = _required_string(data, "device")
    profile = str(data.get("profile") or manifest_device)
    sources = _required_mapping(data, "sources")
    runtime = _required_mapping(data, "runtime")
    paths = _required_mapping(data, "paths")
    launch = _required_mapping(data, "launch")
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("runtime manifest field 'defaults' must be a mapping")
    install = _required_mapping(data, "install")

    requirements_file = expand_path_template(
        _required_string(install, "requirements_file"),
        name=manifest_name,
        device=manifest_device,
    )
    server_path = expand_path_template(
        _required_string(launch, "server_path"),
        name=manifest_name,
        device=manifest_device,
    )

    return RuntimeManifest(
        name=manifest_name,
        device=manifest_device,
        profile=profile,
        manifest_path=path,
        repo_url=_required_string(sources, "repo_url"),
        checkpoint_url=_optional_string(sources, "checkpoint_url"),
        wav2vec_repo_id=_optional_string(sources, "wav2vec_repo_id"),
        runtime_dir=expand_path_template(
            _required_string(paths, "runtime_dir"),
            name=manifest_name,
            device=manifest_device,
        ),
        repo_dir=expand_path_template(
            _required_string(paths, "repo_dir"),
            name=manifest_name,
            device=manifest_device,
        ),
        requirements_file=requirements_file,
        env_script=(
            expand_path_template(_optional_string(runtime, "env_script"), name=manifest_name, device=manifest_device)
            if _optional_string(runtime, "env_script")
            else None
        ),
        server_path=server_path,
        ckpt_dir=str(defaults.get("ckpt_dir") or ""),
        wav2vec_dir=str(defaults.get("wav2vec_dir") or ""),
        python_version=str(runtime.get("python", "3.10")),
        nproc_per_node=int(runtime.get("nproc_per_node", 8)),
        pip_index_url=str(install.get("pip_index_url", "https://pypi.tuna.tsinghua.edu.cn/simple")),
        pip_extra_index_url=_optional_string(install, "pip_extra_index_url"),
        hf_endpoint=str(install.get("hf_endpoint", "https://hf-mirror.com")),
        repo_marker_dir=_optional_string(install, "repo_marker_dir") or _default_repo_marker_dir(manifest_name),
    )
