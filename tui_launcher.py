#!/usr/bin/env python3
"""Terminal UI entrypoint for AutoDeepSeek.

The TUI guides users through environment detection, optional setup, LLM
selection, and launching the agent without automatically downloading
anything on startup.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Sequence
from typing import Dict, List, Optional, Sequence, Tuple

from colorama import Fore, Style, init as color_init
from packaging.requirements import Requirement

from system_profile import SystemProfile, describe_profile, detect_system_profile

color_init(autoreset=True)

SUPPORTED_MODELS: List[str] = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "NousResearch/Meta-Llama-3-8B-Instruct",
@dataclass
class EnvironmentCandidate:
    name: str
    python_path: Path
    origin: str
    exists: bool = True
    is_current: bool = False


@dataclass
class EnvironmentStatus(EnvironmentCandidate):
    present_packages: List[str] = field(default_factory=list)
    missing_packages: List[str] = field(default_factory=list)

    def score(self) -> Tuple[int, int]:
        """Higher score means more requirements are already satisfied."""

        # prefer environments with more packages and give current env a small boost
        return (len(self.present_packages or []), 1 if self.is_current else 0)


@dataclass
class InstallStep:
    name: str
    commands: Sequence[str]
    note: Optional[str] = None


@dataclass
class InstallationReport:
    ready: bool
    environments: List[EnvironmentStatus]
    target_environment: EnvironmentStatus
    requirement_specs: Dict[str, Requirement]


MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def has_package(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None
def _load_requirement_specs() -> Dict[str, Requirement]:
    req_path = Path(__file__).parent / "requirements.txt"
    specs: Dict[str, Requirement] = {}
    if not req_path.exists():
        return specs

    for raw in req_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        specs[req.name] = req
    return specs


def _python_from_env_dir(env_dir: Path) -> Optional[Path]:
    unix_candidate = env_dir / "bin" / "python"
    win_candidate = env_dir / "Scripts" / "python.exe"
    if unix_candidate.exists():
        return unix_candidate
    if win_candidate.exists():
        return win_candidate
    return None


def _environment_root_from_python(python_path: Path) -> Path:
    if python_path.name.lower().startswith("python"):
        return python_path.parent.parent
    return python_path


def _discover_environment_candidates() -> List[EnvironmentCandidate]:
    candidates: List[EnvironmentCandidate] = []
    seen: set[str] = set()

    def add_candidate(python_path: Path, name: str, origin: str, exists: bool = True, is_current: bool = False) -> None:
        key = str(python_path.resolve())
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            EnvironmentCandidate(
                name=name,
                python_path=python_path,
                origin=origin,
                exists=exists,
                is_current=is_current,
            )
        )

    add_candidate(Path(sys.executable), "Current environment", "sys", is_current=True)

    for folder in ["venv", ".venv", "env", ".env"]:
        env_dir = Path.cwd() / folder
        python_candidate = _python_from_env_dir(env_dir)
        if python_candidate:
            add_candidate(python_candidate, f"{folder} (project)", "local-folder")

    for env_root in filter(None, [os.environ.get("WORKON_HOME"), str(Path.home() / ".virtualenvs")]):
        root_path = Path(env_root).expanduser()
        if not root_path.exists() or not root_path.is_dir():
            continue
        for env_dir in root_path.iterdir():
            python_candidate = _python_from_env_dir(env_dir)
            if python_candidate:
                add_candidate(python_candidate, env_dir.name, "virtualenvwrapper")

    conda_exe = shutil.which("conda")
    if conda_exe:
        try:
            conda_info = subprocess.run(
                [conda_exe, "env", "list", "--json"], capture_output=True, text=True, check=True
            )
            envs = conda_info.stdout
            import json

            env_data = json.loads(envs)
            for env_path in env_data.get("envs", []):
                python_candidate = _python_from_env_dir(Path(env_path))
                if python_candidate:
                    add_candidate(python_candidate, Path(env_path).name, "conda")
        except Exception:
            pass

    return candidates


def _evaluate_environment(candidate: EnvironmentCandidate, requirement_names: List[str]) -> EnvironmentStatus:
    present: List[str] = []
    missing: List[str] = list(requirement_names)

    if candidate.exists:
        probe_script = "import importlib.util, sys; pkgs=sys.argv[1:]; present=[p for p in pkgs if importlib.util.find_spec(p)]; missing=[p for p in pkgs if p not in present]; print('\\n'.join([','.join(present), ','.join(missing)]))"
        try:
            result = subprocess.run(
                [str(candidate.python_path), "-c", probe_script, *requirement_names],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if lines:
                present = lines[0].split(",") if lines[0] else []
                missing = lines[1].split(",") if len(lines) > 1 and lines[1] else []
        except Exception:
            pass

    return EnvironmentStatus(
        name=candidate.name,
        python_path=candidate.python_path,
        origin=candidate.origin,
        exists=candidate.exists,
        is_current=candidate.is_current,
        present_packages=present,
        missing_packages=missing,
    )


def _select_target_environment(statuses: List[EnvironmentStatus], requirement_names: List[str]) -> EnvironmentStatus:
    existing_best = max(statuses, key=lambda env: env.score(), default=None)

    if existing_best and existing_best.present_packages:
        return existing_best

    proposed_env_dir = Path.cwd() / "venv"
    python_candidate = _python_from_env_dir(proposed_env_dir)
    if not python_candidate:
        python_candidate = proposed_env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    return EnvironmentStatus(
        name="New virtualenv (./venv)",
        python_path=python_candidate,
        origin="proposed",
        exists=proposed_env_dir.exists(),
        is_current=False,
        present_packages=[],
        missing_packages=list(requirement_names),
    )


def _map_missing_specs(specs: Dict[str, Requirement], missing_names: List[str]) -> List[str]:
    return [str(specs[name]) for name in missing_names if name in specs]


def check_installation() -> InstallationReport:
    missing = [pkg for pkg in REQUIRED_PACKAGES if not has_package(pkg)]
    requirement_specs = _load_requirement_specs()
    requirement_names = list(requirement_specs.keys()) or REQUIRED_PACKAGES

    candidates = _discover_environment_candidates()
    statuses = [_evaluate_environment(candidate, requirement_names) for candidate in candidates]

    target = _select_target_environment(statuses, requirement_names)
    if all(env.name != target.name for env in statuses):
        statuses.append(target)

    missing = target.missing_packages or []
    has_firefox = bool(shutil.which("firefox") or shutil.which("Mozilla Firefox"))
    notes: List[str] = []
    if not has_firefox:
        notes.append("Firefox not detected (required for browsing features)")
    return InstallationReport(ready=not missing, missing_packages=missing, has_firefox=has_firefox, notes=notes)

    ready = target.exists and not missing
    if target.exists and missing:
        missing_specs = _map_missing_specs(requirement_specs, missing)
        readable_specs = missing_specs if missing_specs else missing
        notes.append(
            f"{target.name} selected. Will install missing packages: {', '.join(readable_specs)}"
        )
    elif not target.exists:
        notes.append(f"No suitable environment found. A new one will be created at {target.python_path.parent}.")
    else:
        notes.append(f"{target.name} satisfies all Python requirements.")

    return InstallationReport(
        ready=ready,
        missing_packages=missing,
        has_firefox=has_firefox,
        notes=notes,
        environments=statuses,
        target_environment=target,
        requirement_specs=requirement_specs,
    )


def _rocm_steps() -> List[str]:
    return [
        "sudo apt update",
        "sudo apt install -y wget gnupg software-properties-common",
        "wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -",
        "echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list",
        "sudo apt update",
        "sudo apt install -y rocm-dkms rocm-libs rocm-dev rocm-utils",
def build_install_plan(profile: SystemProfile) -> List[InstallStep]:
def build_install_plan(profile: SystemProfile, report: InstallationReport) -> List[InstallStep]:
    steps: List[InstallStep] = []

    if profile.os_name == "Linux":
        steps.append(
            InstallStep(
                name="System packages",
                commands=["sudo apt update", "sudo apt install -y python3-venv git firefox"],
                note="Adapt these commands if you are not using an apt-based distribution.",
            )
        )
    target_env = report.target_environment
    target_root = _environment_root_from_python(target_env.python_path)
    missing_specs = _map_missing_specs(report.requirement_specs, target_env.missing_packages) or target_env.missing_packages

    env_setup_commands: List[str] = []
    if not target_env.exists:
        env_setup_commands.append(f"{sys.executable} -m venv {target_root}")

    install_python = target_env.python_path if target_env.exists else target_root / (
        "Scripts/python.exe" if profile.os_name == "Windows" else "bin/python"
    )

    if missing_specs:
        env_setup_commands.append(f"{install_python} -m pip install -U pip")
        env_setup_commands.append(f"{install_python} -m pip install {' '.join(missing_specs)}")
    else:
        env_setup_commands.append("echo Environment already satisfies Python requirements.")

    steps.append(
        InstallStep(
            name="Python environment",
            commands=[
                f"{sys.executable} -m venv venv",
                "source venv/bin/activate" if profile.os_name != "Windows" else ".\\venv\\Scripts\\activate",
                "pip install -U pip",
                "pip install -r requirements.txt",
            ],
            note="Run in your project root. The commands are executed sequentially inside this TUI when approved.",
            commands=env_setup_commands,
            note=(
                f"Target environment: {target_env.name}. Commands run directly with its python executable"
                " so activation is not required."
            ),
        )
    )

    steps.append(
        InstallStep(
            name="Verify PyTorch hardware backend",
            commands=[
                f"{sys.executable} - <<'PY'\nimport torch;print('CUDA', torch.cuda.is_available());print('MPS', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())\nPY"
            ],
            note="Ensure the reported backend matches your hardware.",
def _render_environment_table(report: InstallationReport) -> str:
    total_requirements = len(report.requirement_specs) or len(REQUIRED_PACKAGES)
    lines = [f"{Fore.BLUE}{Style.BRIGHT}ENVIRONMENT SCAN › {len(report.environments)} candidates{Style.RESET_ALL}"]

    for env in sorted(report.environments, key=lambda e: e.score(), reverse=True):
        is_target = env.name == report.target_environment.name and env.origin == report.target_environment.origin
        badge = f"{Fore.MAGENTA}◎{Style.RESET_ALL}" if is_target else f"{Fore.WHITE}○{Style.RESET_ALL}"

        if not env.exists:
            status_label = f"{Fore.MAGENTA}pending creation{Style.RESET_ALL}"
        elif env.missing_packages:
            status_label = f"{Fore.YELLOW}{len(env.present_packages)}/{total_requirements} ready{Style.RESET_ALL}"
        else:
            status_label = f"{Fore.GREEN}all {total_requirements} ready{Style.RESET_ALL}"

        missing_preview = ", ".join(env.missing_packages[:3]) if env.missing_packages else ""
        extras = []
        if env.is_current:
            extras.append("current")
        if env.origin != "sys":
            extras.append(env.origin)
        if env.missing_packages:
            extras.append(f"missing: {missing_preview or 'packages'}")

        suffix = f" ({'; '.join(extras)})" if extras else ""
        lines.append(f" {badge} {env.name} → {status_label}{suffix}")

    return "\n".join(lines)


def show_menu(status: InstallationReport, selected_model: Optional[str]) -> None:
    status_icon = f"{Fore.GREEN}●{Style.RESET_ALL}" if status.ready else f"{Fore.YELLOW}●{Style.RESET_ALL}"
    status_pill = f"{Fore.GREEN}{Style.BRIGHT}READY{Style.RESET_ALL}" if status.ready else f"{Fore.YELLOW}{Style.BRIGHT}SETUP NEEDED{Style.RESET_ALL}"
    env_table = _render_environment_table(status)
    neon_rule = f"{Fore.MAGENTA}{'═' * 58}{Style.RESET_ALL}"
    print(
        dedent(
            f"""
            ==================================================
            AutoDeepSeek Setup & Launch TUI {status_icon}
            ==================================================
            Installation status : {'READY' if status.ready else 'NEEDS SETUP'}
            Missing packages    : {', '.join(status.missing_packages) if status.missing_packages else 'None'}
            Firefox available   : {'Yes' if status.has_firefox else 'No'}
            Selected LLM        : {selected_model or 'None'}
            --------------------------------------------------
            [1] Inspect detected system profile
            [2] Run guided setup steps
            [3] Refresh installation status
            [4] Choose and install an LLM
            [5] Launch AutoDeepSeek agent
            [0] Exit
            --------------------------------------------------

            {Fore.CYAN}{Style.BRIGHT}┏━━━━━━━━━━━ AUTO-DEEPSEEK CONTROL DECK ━━━━━━━━━━━┓{Style.RESET_ALL}
            {env_table}
            {neon_rule}
            Status        : {status_pill}
            Firefox       : {'✔ Detected' if status.has_firefox else '✖ Missing'}
            Target env    : {status.target_environment.name}
            Missing pkgs  : {', '.join(status.missing_packages) if status.missing_packages else 'None'}
            Selected LLM  : {selected_model or 'None'}
            {neon_rule}
            [1] System profile    [2] Guided setup
            [3] Refresh status    [4] Choose/Cache LLM
            [5] Launch agent      [0] Exit
            {Fore.CYAN}{Style.BRIGHT}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛{Style.RESET_ALL}
            """
        ).strip()
    )


def pick_model(current: Optional[str]) -> str:
    print("\nAvailable models:")
    for idx, model in enumerate(SUPPORTED_MODELS, start=1):
        print(f"  [{idx}] {model}")
    print("  [C] Custom model ID")
def guided_setup(profile: SystemProfile) -> None:
    steps = build_install_plan(profile)
def guided_setup(profile: SystemProfile, report: InstallationReport) -> None:
    steps = build_install_plan(profile, report)
    print("\nSetup steps tailored to your system:")
    for idx, step in enumerate(steps, start=1):
        print(f"\n[{idx}] {step.name}")
        if step.note:
            print(f"   Note: {step.note}")
        for cmd in step.commands:
            print(f"   - {cmd}")
        choice = input("Run this step now? [y/N] ").strip().lower()
        if choice == "y":
            run_commands(step.commands)
            guided_setup(profile)
            guided_setup(profile, status)
            status = check_installation()
        elif choice == "3":
            status = check_installation()
            print("Status refreshed.")
        elif choice == "4":
            selected_model = pick_model(selected_model)
            install_prompt = input("Download/cached selected model now? [y/N] ").strip().lower()
            if install_prompt == "y":
                install_model(selected_model)
        elif choice == "5":