"""Installer for OmniRT-managed model runtimes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from omnirt.runtime.manifest import RuntimeManifest
from omnirt.runtime.paths import project_root
from omnirt.runtime.state import RuntimeState, write_state


def _patch_soulx_wan_t5_for_cpu_torch(repo_dir: Path) -> bool:
    """Patch SoulX-FlashTalk to avoid torch.cuda.* at import time when torch is CPU-only.

    Upstream `flash_talk/wan/modules/t5.py` defaults `device=` to `torch.cuda.current_device()`,
    which raises when torch is built without CUDA (common on Ascend wheel installs).
    """
    path = repo_dir / "flash_talk" / "wan" / "modules" / "t5.py"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if "def _omnirt_default_torch_device_index()" in text:
        return False

    helper = (
        "\n\ndef _omnirt_default_torch_device_index() -> int:\n"
        "    if hasattr(torch, 'npu') and torch.npu.is_available():\n"
        "        return int(torch.npu.current_device())\n"
        "    if torch.cuda.is_available():\n"
        "        return int(torch.cuda.current_device())\n"
        "    return 0\n"
    )
    needle = "\n__all__ = ["
    if needle not in text:
        needle = "__all__ = ["
        if needle not in text:
            return False
        text = text.replace(needle, helper + needle, 1)
    else:
        text = text.replace(needle, "\n" + helper + "__all__ = [", 1)

    import re

    pattern = r"device\s*=\s*torch\.cuda\.current_device\(\)"
    replacement = "device=_omnirt_default_torch_device_index()"
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        return False
    text = new_text
    path.write_text(text, encoding="utf-8")
    return True


@dataclass(frozen=True)
class RuntimeInstallResult:
    state: RuntimeState
    state_path: Path | None
    commands: list[list[str]]


class RuntimeInstaller:
    def __init__(self, manifest: RuntimeManifest) -> None:
        self.manifest = manifest
        self.commands: list[list[str]] = []

    def install(
        self,
        *,
        dry_run: bool = False,
        update: bool = True,
        recreate_venv: bool = False,
    ) -> RuntimeInstallResult:
        self._require_local_file(self.manifest.requirements_file, "requirements")
        if self.manifest.env_script is not None:
            self._require_local_file(self.manifest.env_script, "Ascend/CANN environment script")
        state = RuntimeState.from_manifest(self.manifest)

        if dry_run:
            self.commands = self.plan_commands(update=update, recreate_venv=recreate_venv)
            return RuntimeInstallResult(state=state, state_path=None, commands=self.commands)

        self._require_command("python3")

        self._clone_or_update(
            self.manifest.repo_url,
            self.manifest.repo_dir,
            update=update,
            marker_dir=self.manifest.repo_marker_dir,
            label=self._repo_label,
        )
        if self.manifest.name == "flashtalk":
            self._apply_soulx_ascend_patch(self.manifest.repo_dir)
            if _patch_soulx_wan_t5_for_cpu_torch(self.manifest.repo_dir):
                self.commands.append(["patch", "SoulX-FlashTalk", "flash_talk/wan/modules/t5.py", "import-time device default"])
        self._prepare_venv(recreate=recreate_venv)
        self._run(self._bootstrap_pip_command())
        self._install_requirements()
        if self.manifest.checkpoint_url or self.manifest.wav2vec_repo_id:
            self._prepare_checkpoints(update=update)

        state_path = write_state(state)
        return RuntimeInstallResult(state=state, state_path=state_path, commands=self.commands)

    def plan_commands(self, *, update: bool = True, recreate_venv: bool = False) -> list[list[str]]:
        commands = [
            self._plan_clone_or_update(
                self.manifest.repo_url,
                self.manifest.repo_dir,
                update=update,
                marker_dir=self.manifest.repo_marker_dir,
                label=self._repo_label,
            ),
        ]
        if self.manifest.name == "flashtalk":
            commands.extend(
                [
                    [
                        "git",
                        "-C",
                        str(self.manifest.repo_dir),
                        "apply",
                        str(project_root() / "model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch"),
                        "(skip if reverse --check passes: already applied; skip if no .git)",
                    ],
                    ["patch", "SoulX-FlashTalk", "flash_talk/wan/modules/t5.py", "import-time device default (if needed)"],
                ]
            )
        if recreate_venv and self.manifest.venv_dir.exists():
            commands.append(["recreate-venv", str(self.manifest.venv_dir)])
        elif self.manifest.python_path.is_file():
            commands.append(["skip", "venv", str(self.manifest.venv_dir), "already exists"])
        else:
            commands.append(["python3", "-m", "venv", str(self.manifest.venv_dir)])
        commands.extend(
            [
                self._bootstrap_pip_command(),
                self._planned_install_requirements_command(),
            ]
        )
        if self.manifest.checkpoint_url:
            commands.append(
                self._plan_clone_or_update(
                    self.manifest.checkpoint_url,
                    self.manifest.resolved_ckpt_dir,
                    update=update,
                    marker_dir=None,
                    label=self._checkpoint_label,
                )
            )
        if not self.manifest.wav2vec_repo_id:
            return commands
        wav2vec = self.manifest.resolved_wav2vec_dir
        if self._has_content(wav2vec):
            commands.append(["skip", "wav2vec", str(wav2vec), "already exists"])
        else:
            commands.append(
                [
                    str(self.manifest.python_path.parent / "hf"),
                    "download",
                    self.manifest.wav2vec_repo_id,
                    "--local-dir",
                    str(wav2vec),
                ]
            )
        return commands

    @property
    def _repo_label(self) -> str:
        if self.manifest.name == "flashtalk":
            return "SoulX-FlashTalk checkout"
        if self.manifest.name == "musetalk":
            return "MuseTalk checkout"
        return f"{self.manifest.name} checkout"

    @property
    def _checkpoint_label(self) -> str:
        if self.manifest.name == "flashtalk":
            return "FlashTalk checkpoint"
        return f"{self.manifest.name} checkpoint"

    def _install_requirements_command(self) -> list[str]:
        command = [
            str(self.manifest.python_path),
            "-m",
            "pip",
            "install",
            "-i",
            self.manifest.pip_index_url,
        ]
        if self.manifest.pip_extra_index_url:
            command.extend(["--extra-index-url", self.manifest.pip_extra_index_url])
        command.extend(["-r", str(self.manifest.requirements_file)])
        return command

    def _bootstrap_pip_command(self) -> list[str]:
        command = [
            str(self.manifest.python_path),
            "-m",
            "pip",
            "install",
            "-U",
            "pip",
            "setuptools",
            "wheel",
        ]
        if self.manifest.pip_index_url:
            command.extend(["-i", self.manifest.pip_index_url])
        return command

    def _install_requirements(self) -> None:
        uv = shutil.which("uv")
        if uv:
            self._run(self._uv_install_requirements_command(uv))
            return
        self._run(self._install_requirements_command())

    def _planned_install_requirements_command(self) -> list[str]:
        uv = shutil.which("uv")
        if uv:
            return self._uv_install_requirements_command(uv)
        return self._install_requirements_command()

    def _uv_install_requirements_command(self, uv: str) -> list[str]:
        command = [
            uv,
            "pip",
            "install",
            "--python",
            str(self.manifest.python_path),
            "--index-strategy",
            "unsafe-best-match",
            "-i",
            self.manifest.pip_index_url,
        ]
        if self.manifest.pip_extra_index_url:
            command.extend(["--extra-index-url", self.manifest.pip_extra_index_url])
        command.extend(["-r", str(self.manifest.requirements_file)])
        return command

    def _prepare_venv(self, *, recreate: bool) -> None:
        if recreate and self.manifest.venv_dir.exists():
            shutil.rmtree(self.manifest.venv_dir)
        if self.manifest.python_path.is_file():
            self.commands.append(["skip", "venv", str(self.manifest.venv_dir), "already exists"])
            return
        self._run(["python3", "-m", "venv", str(self.manifest.venv_dir)])

    def _prepare_checkpoints(self, *, update: bool) -> None:
        if self.manifest.checkpoint_url:
            self.manifest.resolved_ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
            self._clone_or_update(
                self.manifest.checkpoint_url,
                self.manifest.resolved_ckpt_dir,
                update=update,
                marker_dir=None,
                label=self._checkpoint_label,
            )
        if not self.manifest.wav2vec_repo_id:
            return
        wav2vec = self.manifest.resolved_wav2vec_dir
        if self._has_content(wav2vec):
            self.commands.append(["skip", "wav2vec", str(wav2vec), "already exists"])
            return
        env = dict(os.environ)
        env["HF_ENDPOINT"] = self.manifest.hf_endpoint
        env["HF_HUB_DISABLE_XET"] = env.get("HF_HUB_DISABLE_XET", "1")
        self._run(
            [
                str(self.manifest.python_path.parent / "hf"),
                "download",
                self.manifest.wav2vec_repo_id,
                "--local-dir",
                str(wav2vec),
            ],
            env=env,
        )

    def _clone_or_update(
        self,
        url: str,
        directory: Path,
        *,
        update: bool,
        marker_dir: str | None,
        label: str,
    ) -> None:
        if (directory / ".git").exists():
            if update:
                self._require_command("git")
                self._run(["git", "-C", str(directory), "fetch", "--all", "--tags"])
                self._run(["git", "-C", str(directory), "pull", "--ff-only"], allow_failure=True)
            else:
                self.commands.append(["skip", label, str(directory), "--no-update"])
            return
        if directory.exists():
            if marker_dir and (directory / marker_dir).is_dir():
                self.commands.append(["skip", label, str(directory), f"contains {marker_dir}/"])
                return
            # Allow pre-populated model dirs (e.g. symlinks to existing weights without .git).
            if marker_dir is None and self._has_content(directory):
                self.commands.append(["skip", label, str(directory), "already exists"])
                return
            raise RuntimeError(f"{label} path exists but is not usable: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        self._require_command("git")
        self._run(["git", "clone", url, str(directory)])

    def _apply_soulx_ascend_patch(self, repo_dir: Path) -> None:
        """Apply OmniRT-maintained Ascend compatibility patch under flash_talk/."""
        patch = (
            project_root()
            / "model_backends"
            / "flashtalk"
            / "patches"
            / "soulx-flashtalk-ascend-omnirt.patch"
        )
        if not patch.is_file():
            self.commands.append(["skip", "SoulX Ascend patch", str(patch), "missing"])
            return
        if not (repo_dir / ".git").is_dir():
            self.commands.append(["skip", "SoulX Ascend patch", str(repo_dir), "no .git"])
            return
        self._require_command("git")
        reverse = subprocess.run(
            ["git", "-C", str(repo_dir), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        if reverse.returncode == 0:
            self.commands.append(["skip", "SoulX Ascend patch", str(repo_dir), "already applied"])
            return
        check = subprocess.run(
            ["git", "-C", str(repo_dir), "apply", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            msg = (check.stderr or check.stdout or "").strip()
            raise RuntimeError(
                "SoulX Ascend patch does not apply to "
                f"{repo_dir}. Regenerate "
                "model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch "
                "from the current upstream SoulX-FlashTalk commit (see patches/README.md), "
                f"or resolve local edits.\n{msg}"
            )
        self._run(["git", "-C", str(repo_dir), "apply", str(patch)])

    def _plan_clone_or_update(
        self,
        url: str,
        directory: Path,
        *,
        update: bool,
        marker_dir: str | None,
        label: str,
    ) -> list[str]:
        if (directory / ".git").exists():
            if update:
                return ["git", "-C", str(directory), "fetch", "--all", "--tags", "&&", "git", "-C", str(directory), "pull", "--ff-only"]
            return ["skip", label, str(directory), "--no-update"]
        if directory.exists():
            if marker_dir and (directory / marker_dir).is_dir():
                return ["skip", label, str(directory), f"contains {marker_dir}/"]
            if marker_dir is None and self._has_content(directory):
                return ["skip", label, str(directory), "already exists"]
            return ["error", label, str(directory), "path exists but is not usable"]
        return ["git", "clone", url, str(directory)]

    def _run(self, command: list[str], *, env: dict[str, str] | None = None, allow_failure: bool = False) -> None:
        self.commands.append(command)
        result = subprocess.run(command, env=env, check=False)
        if result.returncode != 0 and not allow_failure:
            quoted = " ".join(shlex.quote(part) for part in command)
            raise RuntimeError(f"command failed ({result.returncode}): {quoted}")

    @staticmethod
    def _require_command(name: str) -> None:
        if shutil.which(name) is None:
            raise RuntimeError(f"required command not found: {name}")

    @staticmethod
    def _require_local_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    @staticmethod
    def _has_content(path: Path) -> bool:
        return path.is_dir() and any(path.iterdir())
