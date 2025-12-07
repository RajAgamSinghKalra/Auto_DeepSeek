#!/usr/bin/env python3
"""Touch-first terminal UI for AutoDeepSeek setup and launch."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Lazy-install and import prompt_toolkit so we stay usable even if the package is missing.
def _ensure_prompt_toolkit() -> None:
    try:
        import prompt_toolkit  # type: ignore
        return
    except Exception:
        print("prompt-toolkit not found. Attempting to install 'prompt-toolkit>=3.0.48' ...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "prompt-toolkit>=3.0.48"],
                check=True,
            )
        except Exception as exc:
            print(f"Auto-install failed: {exc}\nInstall manually with: pip install \"prompt-toolkit>=3.0.48\"")
            sys.exit(1)

    try:
        import prompt_toolkit  # type: ignore
    except Exception:
        print("prompt-toolkit still unavailable. Install manually with: pip install \"prompt-toolkit>=3.0.48\"")
        sys.exit(1)


_ensure_prompt_toolkit()

from prompt_toolkit.application import Application, run_in_terminal  # type: ignore
from prompt_toolkit.formatted_text import FormattedText  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore
from prompt_toolkit.layout import HSplit, VSplit, Layout, Window  # type: ignore
from prompt_toolkit.layout.containers import DynamicContainer  # type: ignore
from prompt_toolkit.layout.controls import FormattedTextControl  # type: ignore
from prompt_toolkit.shortcuts import button_dialog, input_dialog, message_dialog, yes_no_dialog  # type: ignore
from prompt_toolkit.styles import Style  # type: ignore
from prompt_toolkit.widgets import Box, Button, Frame, TextArea  # type: ignore

from colorama import init as color_init
from packaging.requirements import Requirement

from system_profile import SystemProfile, describe_profile, detect_system_profile

color_init(autoreset=True)

SUPPORTED_MODELS: List[str] = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "NousResearch/Meta-Llama-3-8B-Instruct",
]

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "selenium",
    "requests",
    "psutil",
    "GPUtil",
]

CONFIG_PATH = Path.home() / ".autodeepseek_tui.json"
MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
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
        return (len(self.present_packages or []), 1 if self.is_current else 0)


@dataclass
class InstallStep:
    name: str
    commands: Sequence[str]
    note: Optional[str] = None


@dataclass
class InstallationReport:
    ready: bool
    missing_packages: List[str]
    has_firefox: bool
    notes: List[str]
    environments: List[EnvironmentStatus]
    target_environment: EnvironmentStatus
    requirement_specs: Dict[str, Requirement]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
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

    if candidate.exists and candidate.python_path.exists():
        probe_script = (
            "import importlib.util, json, sys; "
            "pkgs=sys.argv[1:]; present=[]; missing=[]; "
            "for p in pkgs: "
            " spec=importlib.util.find_spec(p); "
            " (present if spec else missing).append(p); "
            "print(json.dumps({'present':present,'missing':missing}))"
        )
        try:
            result = subprocess.run(
                [str(candidate.python_path), "-c", probe_script, *requirement_names],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout.strip() or "{}")
            present = payload.get("present", []) or []
            missing = payload.get("missing", []) or []
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
    if statuses:
        best = max(statuses, key=lambda env: env.score())
    else:
        best = None

    if best and (not best.missing_packages or best.is_current):
        return best

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


def detect_firefox() -> bool:
    candidates = [
        "firefox",
        "Mozilla Firefox",
        "firefox.exe",
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        "/Applications/Firefox.app/Contents/MacOS/firefox",
    ]
    for candidate in candidates:
        if shutil.which(candidate) or Path(candidate).exists():
            return True
    return False


def _load_config() -> Dict[str, str]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _save_config(selected_model: Optional[str]) -> None:
    data = {"model": selected_model} if selected_model else {}
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def check_installation(target_hint: Optional[Tuple[str, str]] = None) -> InstallationReport:
    requirement_specs = _load_requirement_specs()
    requirement_names = list(requirement_specs.keys()) or REQUIRED_PACKAGES

    candidates = _discover_environment_candidates()
    statuses = [_evaluate_environment(candidate, requirement_names) for candidate in candidates]

    target = _select_target_environment(statuses, requirement_names)
    if target_hint:
        for env in statuses:
            if env.name == target_hint[0] and env.origin == target_hint[1]:
                target = env
                break

    if all(env.name != target.name or env.origin != target.origin for env in statuses):
        statuses.append(target)

    missing = target.missing_packages or []
    has_firefox = detect_firefox()
    notes: List[str] = []

    ready = target.exists and not missing
    if not has_firefox:
        notes.append("Firefox not detected (required for browsing features).")

    if target.exists and missing:
        missing_specs = _map_missing_specs(requirement_specs, missing)
        readable_specs = missing_specs if missing_specs else missing
        notes.append(f"{target.name} selected. Missing packages: {', '.join(readable_specs)}")
    elif not target.exists:
        notes.append(f"No suitable environment found. A new one will be created at {target.python_path.parent}.")
    else:
        notes.append(f"{target.name} satisfies Python requirements.")

    return InstallationReport(
        ready=ready,
        missing_packages=missing,
        has_firefox=has_firefox,
        notes=notes,
        environments=statuses,
        target_environment=target,
        requirement_specs=requirement_specs,
    )


# --------------------------------------------------------------------------------------
# Install plans
# --------------------------------------------------------------------------------------
def _rocm_steps() -> List[str]:
    return [
        "sudo apt update",
        "sudo apt install -y wget gnupg software-properties-common",
        "wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -",
        "echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list",
        "sudo apt update",
        "sudo apt install -y rocm-dkms rocm-libs rocm-dev rocm-utils",
        "sudo usermod -a -G render,video $USER",
    ]


def build_install_plan(profile: SystemProfile, report: InstallationReport) -> List[InstallStep]:
    steps: List[InstallStep] = []

    if profile.os_name == "Linux":
        steps.append(
            InstallStep(
                name="System packages",
                commands=["sudo apt update", "sudo apt install -y python3-venv git firefox"],
                note="Adapt if you are not on an apt-based distribution.",
            )
        )
        if profile.gpu and profile.gpu.vendor == "AMD":
            steps.append(InstallStep(name="ROCm drivers", commands=_rocm_steps(), note="Reboot after installation."))
        elif profile.gpu and profile.gpu.vendor == "NVIDIA":
            steps.append(
                InstallStep(
                    name="NVIDIA drivers",
                    commands=["sudo apt install -y nvidia-driver-550 || sudo ubuntu-drivers autoinstall"],
                    note="Install CUDA toolkit separately if you need specific versions.",
                )
            )
    elif profile.os_name == "Windows":
        steps.append(
            InstallStep(
                name="Install prerequisites with Chocolatey",
                commands=[
                    "choco install -y python git firefox",
                    "choco install -y cuda --pre || echo Install CUDA manually if needed",
                ],
                note="Run from an elevated PowerShell prompt.",
            )
        )
    elif profile.os_name == "Darwin":
        steps.append(
            InstallStep(
                name="Install prerequisites with Homebrew",
                commands=["brew update", "brew install python git firefox"],
                note="Apple Silicon devices use MPS acceleration automatically when PyTorch supports it.",
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
            commands=env_setup_commands,
            note=(
                f"Target environment: {target_env.name}. Commands run directly with its python executable,"
                " so activation is not required."
            ),
        )
    )

    steps.append(
        InstallStep(
            name="Verify PyTorch hardware backend",
            commands=[
                f"{install_python} - <<'PY'\n"
                "import torch\n"
                "print('CUDA available:', torch.cuda.is_available())\n"
                "print('MPS available:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())\n"
                "PY"
            ],
            note="Ensure the reported backend matches your hardware.",
        )
    )

    return steps


# --------------------------------------------------------------------------------------
# Visual helpers
# --------------------------------------------------------------------------------------
def bar(filled: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    filled_blocks = int(width * min(filled, total) / total)
    return "█" * filled_blocks + "░" * (width - filled_blocks)


def badge(label: str, state: str) -> str:
    if state == "ok":
        return f"[{label}:✓]"
    if state == "warn":
        return f"[{label}:!]"
    return f"[{label}:x]"


def spark(values: List[int]) -> str:
    chars = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    mn, mx = min(values), max(values)
    span = max(mx - mn, 1)
    return "".join(chars[int((v - mn) / span * (len(chars) - 1))] for v in values)


def neon_grid(width: int = 36, height: int = 6) -> str:
    lines = []
    for y in range(height):
        row = []
        for x in range(width):
            phase = math.sin((x + y) * 0.35) + math.cos((x - y) * 0.25)
            row.append("╱" if phase > 0 else "╲")
        lines.append("".join(row))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------
def run_commands(commands: Sequence[str], log: Callable[[str], None], app: Application) -> None:
    def _runner() -> None:
        for command in commands:
            log(f">> {command}")
            try:
                subprocess.run(command, shell=True, check=True)
                log("OK")
            except subprocess.CalledProcessError as exc:
                log(f"Failed with code {exc.returncode}")
                break

    run_in_terminal(_runner)


def guided_setup(app: Application, profile: SystemProfile, report: InstallationReport, refresh: Callable[[], None], log: Callable[[str], None]) -> None:
    steps = build_install_plan(profile, report)
    for step in steps:
        def _ask() -> bool:
            text = (step.note + "\n\n" if step.note else "") + "\n".join(f"- {cmd}" for cmd in step.commands)
            return yes_no_dialog(title=step.name, text=text).run()

        should_run = run_in_terminal(_ask)
        if should_run:
            run_commands(step.commands, log, app)
            refresh()


def pick_model(app: Application, current: Optional[str]) -> Optional[str]:
    buttons = [(model, model) for model in SUPPORTED_MODELS]
    buttons.append(("Custom model ID", "__custom__"))

    def _choose() -> Optional[str]:
        return button_dialog(title="Select LLM", text="Tap a model to select", buttons=buttons).run()

    choice = run_in_terminal(_choose)
    if choice == "__custom__":
        def _input() -> Optional[str]:
            return input_dialog(title="Custom model", text="Enter Hugging Face model ID:").run()

        custom = run_in_terminal(_input)
        return custom or current
    return choice or current


def cache_model(app: Application, model_id: str, log: Callable[[str], None]) -> None:
    def _confirm() -> bool:
        return yes_no_dialog(
            title="Download model?",
            text=f"Download/cache model '{model_id}' to {MODEL_CACHE_DIR}?\nThis may be large.",
        ).run()

    if not run_in_terminal(_confirm):
        log("Model download skipped.")
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        log("Transformers not installed. Finish setup first.")
        return

    def _download() -> None:
        cache_dir = MODEL_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading tokenizer...")
        AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        print("Downloading model weights... (this can take a while)")
        AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        print(f"Model cached at {cache_dir}")

    log(f"Caching model {model_id} ...")
    try:
        run_in_terminal(_download)
        log("Model cached.")
    except Exception as exc:
        log(f"Model download failed: {exc}")


def launch_agent(app: Application, report: InstallationReport, selected_model: Optional[str], log: Callable[[str], None]) -> None:
    if not selected_model:
        def _alert() -> None:
            message_dialog(title="Select a model", text="Choose a model before launching.").run()

        run_in_terminal(_alert)
        return

    if not report.ready:
        def _alert_not_ready() -> None:
            message_dialog(title="Environment not ready", text="Resolve missing packages before launch.").run()

        run_in_terminal(_alert_not_ready)
        return

    def _extra_flags() -> str:
        return input_dialog(title="Launch options", text="Additional flags for autodeepseek (optional):").run() or ""

    extra_flags = run_in_terminal(_extra_flags)
    cmd: List[str] = [str(report.target_environment.python_path), "autodeepseek.py", "--model", selected_model]
    if extra_flags:
        cmd.extend(shlex.split(extra_flags))

    log(f"Launching: {' '.join(cmd)}")

    def _run() -> None:
        subprocess.run(cmd)

    run_in_terminal(_run)


# --------------------------------------------------------------------------------------
# UI builders
# --------------------------------------------------------------------------------------
def make_status_panel(state: Dict, set_target_hint: Callable[[Optional[Tuple[str, str]]], None]) -> Window:
    profile: SystemProfile = state["profile"]
    report: InstallationReport = state["report"]
    selected_model: Optional[str] = state["model"]

    total = len(report.requirement_specs) or len(REQUIRED_PACKAGES)
    ready_count = total - len(report.missing_packages)

    env_ready_counts = [len(env.present_packages) for env in report.environments]
    env_spark = spark(env_ready_counts)

    def _text() -> FormattedText:
        lines: List[Tuple[str, str]] = []
        lines.append(("class:title", "AUTO-DEEPSEEK :: CONTROL DECK\n"))
        lines.append(("", "\n"))
        lines.append(("class:status-ready" if report.ready else "class:status-warn", badge("Status", "ok" if report.ready else "warn")))
        lines.append(("class:status-ready" if report.has_firefox else "class:status-bad", f" {badge('Firefox', 'ok' if report.has_firefox else 'bad')}"))
        lines.append(("class:accent", f" [LLM: {selected_model or 'none'}]\n\n"))
        lines.append(("class:accent", f"OS: {profile.os_name} | Arch: {profile.arch} | Python: {profile.python_version}\n"))
        if profile.gpu:
            lines.append(("class:accent", f"GPU: {profile.gpu.name} ({profile.gpu.vendor})\n"))
        else:
            lines.append(("class:accent", "GPU: None detected\n"))
        lines.append(("class:accent", f"Accelerators: {', '.join(profile.accelerators)}\n"))
        lines.append(("", "\n"))
        lines.append(("class:label", f"Requirements ready {ready_count}/{total}: "))
        lines.append(("class:bar", f"{bar(ready_count, total, 30)}\n"))
        if env_spark:
            lines.append(("class:label", "Env readiness sparkline: "))
            lines.append(("class:bar", f"{env_spark}\n"))
        return FormattedText(lines)

    return Window(content=FormattedTextControl(_text), always_hide_cursor=True)


def make_env_cards(state: Dict, refresh: Callable[[], None], log: Callable[[str], None], set_target_hint: Callable[[Optional[Tuple[str, str]]], None]) -> List[Frame]:
    report: InstallationReport = state["report"]
    total_requirements = len(report.requirement_specs) or len(REQUIRED_PACKAGES)
    cards: List[Frame] = []

    for env in sorted(report.environments, key=lambda e: e.score(), reverse=True):
        is_target = env.name == report.target_environment.name and env.origin == report.target_environment.origin
        filled = len(env.present_packages)
        meter = bar(filled, total_requirements, 18)
        status_text = "READY" if not env.missing_packages and env.exists else "CREATE" if not env.exists else f"{filled}/{total_requirements}"

        detail_lines = [
            f"{env.origin} | {'current' if env.is_current else 'alt'} | python: {env.python_path}",
            f"{meter} [{status_text}]",
            f"Missing: {', '.join(env.missing_packages[:4])}" if env.missing_packages else "All requirements present",
        ]

        def handler(env=env) -> None:
            set_target_hint((env.name, env.origin))
            log(f"Target set to {env.name}")
            refresh()

        env_button = Button(text=f"{'>> ' if is_target else '   '}{env.name}", handler=handler)
        detail = TextArea(text="\n".join(detail_lines), focusable=False, height=3, style="class:env-meta")
        cards.append(Frame(HSplit([env_button, detail], padding=0), title="Target" if is_target else "Env"))

    return cards


def make_model_deck(state: Dict, choose_model: Callable[[], None], cache_selected: Callable[[], None]) -> Frame:
    selected_model: Optional[str] = state["model"]

    buttons = [Button(text=f"✨ {model}", handler=lambda m=model: (state.update({"model": m}), choose_model())) for model in SUPPORTED_MODELS]
    deck = HSplit(
        [
            TextArea(
                text=f"Selected: {selected_model or 'none'}\nTap a preset below or use actions to set/cache.",
                focusable=False,
                height=3,
                style="class:accent",
            ),
            VSplit([Box(body=b, padding=1) for b in buttons[:2]], padding=2),
            VSplit([Box(body=b, padding=1) for b in buttons[2:]], padding=2),
            Button(text="Download/cache selected", handler=cache_selected),
        ],
        padding=1,
    )
    return Frame(deck, title="LLM Deck")


def make_notes_panel(state: Dict) -> Frame:
    report: InstallationReport = state["report"]
    text = "\n".join(f"- {note}" for note in report.notes) or "No notes."
    return Frame(TextArea(text=text, focusable=False, height=3, style="class:notes"), title="Notes")


def make_actions_bar(on_inspect: Callable[[], None], on_setup: Callable[[], None], on_refresh: Callable[[], None], on_choose: Callable[[], None], on_launch: Callable[[], None], on_exit: Callable[[], None]) -> VSplit:
    buttons = [
        Button(text="Inspect", handler=on_inspect),
        Button(text="Guided setup", handler=on_setup),
        Button(text="Refresh", handler=on_refresh),
        Button(text="Choose model", handler=on_choose),
        Button(text="Launch", handler=on_launch),
        Button(text="Exit", handler=on_exit),
    ]
    bar_children = [Box(body=b, padding=1) for b in buttons]
    bar = VSplit(bar_children, padding=2)
    bar.buttons = buttons  # type: ignore[attr-defined]
    return bar


def make_env_graph(report: InstallationReport) -> Frame:
    if isinstance(report, dict):
        total = len(report.get("requirement_specs", {})) or len(REQUIRED_PACKAGES)
        envs = report.get("environments", [])
    else:
        total = len(report.requirement_specs) or len(REQUIRED_PACKAGES)
        envs = report.environments
    lines: List[str] = []
    for env in sorted(envs, key=lambda e: e.score(), reverse=True):
        filled = len(getattr(env, "present_packages", []))
        pct = int(100 * filled / max(total, 1))
        lines.append(f"{getattr(env, 'name', '')[:18]:<18} [{bar(filled, total, 18)}] {pct:>3}%")
    text = "\n".join(lines) or "No environments"
    return Frame(TextArea(text=text, focusable=False, height=min(8, max(3, len(lines) + 1)), style="class:env-meter"), title="Env Readiness")


def make_pkg_gap_graph(report: InstallationReport) -> Frame:
    if isinstance(report, dict):
        missing = report.get("missing_packages", [])
        total = len(report.get("requirement_specs", {})) or len(REQUIRED_PACKAGES)
    else:
        missing = report.missing_packages
        total = len(report.requirement_specs) or len(REQUIRED_PACKAGES)

    if not missing:
        text = "All required packages present."
    else:
        rows = []
        missing_count = len(missing)
        rows.append(f"Missing {missing_count}/{total} requirements")
        rows.append(bar(total - missing_count, total, 28))
        chunked = [", ".join(missing[i : i + 3]) for i in range(0, len(missing), 3)]
        rows.extend(chunked[:3])
        text = "\n".join(rows)
    return Frame(TextArea(text=text, focusable=False, height=6, style="class:notes"), title="Package Gap")


def make_futuristic_canvas() -> Frame:
    art = neon_grid(40, 8)
    return Frame(TextArea(text=art, focusable=False, height=8, style="class:grid"), title="Neon Lattice")


# --------------------------------------------------------------------------------------
# Application setup
# --------------------------------------------------------------------------------------
def build_app() -> Application:
    profile = detect_system_profile()
    state: Dict = {
        "profile": profile,
        "target_hint": None,
    }
    state["report"] = check_installation()
    state["model"] = _load_config().get("model")

    log_area = TextArea(text="", scrollbar=True, focusable=False, style="class:log", height=8)

    def log(message: str) -> None:
        log_area.buffer.insert_text(message + "\n")
        try:
            log_area.buffer.cursor_position = len(log_area.buffer.text)
        except Exception:
            pass

    def set_target_hint(hint: Optional[Tuple[str, str]]) -> None:
        state["target_hint"] = hint

    def refresh() -> None:
        state["report"] = check_installation(state.get("target_hint"))
        app.invalidate()

    def choose_model_action() -> None:
        selected = pick_model(app, state["model"])
        if selected:
            state["model"] = selected
            _save_config(selected)
            log(f"Model set to {selected}")
        refresh()

    def cache_selected_action() -> None:
        if not state["model"]:
            log("Select a model first.")
            return
        cache_model(app, state["model"], log)
        refresh()

    def on_inspect() -> None:
        def _show() -> None:
            message_dialog(title="System Profile", text=describe_profile(state["profile"])).run()

        run_in_terminal(_show)

    def on_guided_setup() -> None:
        guided_setup(app, state["profile"], state["report"], refresh, log)
        refresh()

    def on_refresh() -> None:
        refresh()
        log("Status refreshed.")

    def on_launch() -> None:
        launch_agent(app, state["report"], state["model"], log)
        refresh()

    header = Frame(make_status_panel(state, set_target_hint), title="Status")

    def view_dashboard() -> HSplit:
        env_cards = make_env_cards(state, refresh, log, set_target_hint)
        env_column = Frame(HSplit(env_cards, padding=1), title="Environments")
        model_deck = make_model_deck(state, choose_model_action, cache_selected_action)
        return HSplit(
            [
                VSplit([env_column, model_deck, make_futuristic_canvas()], padding=1),
                HSplit([make_env_graph(state), make_pkg_gap_graph(state)], padding=1),
            ],
            padding=1,
        )

    def view_envs() -> HSplit:
        env_cards = make_env_cards(state, refresh, log, set_target_hint)
        env_column = Frame(HSplit(env_cards, padding=1), title="Environments")
        action_row = make_actions_bar(on_inspect, on_guided_setup, on_refresh, choose_model_action, on_launch, lambda: app.exit(result=0))
        return HSplit(
            [
                env_column,
                make_env_graph(state),
                make_pkg_gap_graph(state),
                action_row,
            ],
            padding=1,
        )

    def view_llm() -> HSplit:
        deck = make_model_deck(state, choose_model_action, cache_selected_action)
        art = make_futuristic_canvas()
        mini_actions = VSplit(
            [
                Box(body=Button(text="Choose model", handler=choose_model_action), padding=1),
                Box(body=Button(text="Cache model", handler=cache_selected_action), padding=1),
                Box(body=Button(text="Launch", handler=on_launch), padding=1),
            ],
            padding=2,
        )
        return HSplit([deck, art, mini_actions], padding=1)

    def view_logs() -> HSplit:
        notes = make_notes_panel(state)
        log_frame = Frame(log_area, title="Activity Log")
        return HSplit([notes, log_frame], padding=1)

    def current_view() -> HSplit:
        page = state.get("page", "dashboard")
        if page == "envs":
            return view_envs()
        if page == "llm":
            return view_llm()
        if page == "logs":
            return view_logs()
        return view_dashboard()

    def switch_page(target: str) -> Callable[[], None]:
        def _set() -> None:
            state["page"] = target
            app.invalidate()
        return _set

    nav_children = [
        Box(body=Button(text="Dashboard", handler=switch_page("dashboard")), padding=1),
        Box(body=Button(text="Environments", handler=switch_page("envs")), padding=1),
        Box(body=Button(text="LLM", handler=switch_page("llm")), padding=1),
        Box(body=Button(text="Logs", handler=switch_page("logs")), padding=1),
    ]
    nav = VSplit(nav_children, padding=3)
    nav.buttons = [getattr(box, "body", None) for box in nav_children]  # type: ignore[attr-defined]

    layout_root = HSplit(
        [
            header,
            Frame(nav, title="Navigation"),
            DynamicContainer(current_view),
        ],
        padding=1,
    )

    kb = KeyBindings()

    @kb.add("c-c")
    @kb.add("q")
    def _(event) -> None:
        event.app.exit(result=0)

    global app
    app = Application(
        layout=Layout(layout_root, focused_element=getattr(nav, "buttons", [None])[0]),
        key_bindings=kb,
        mouse_support=True,
        full_screen=True,
        style=Style.from_dict(
            {
                "title": "fg:#ff00ff bold",
                "status-ready": "fg:#8dff7c bold",
                "status-warn": "fg:#f4d03f bold",
                "status-bad": "fg:#ff6b6b bold",
                "accent": "fg:#00ffff bold",
                "label": "fg:#cccccc bold",
                "value": "fg:#ffffff",
                "frame.border": "fg:#ff00ff",
                "log": "fg:#00ffcc",
                "env-title": "fg:#00ffff bold",
                "env-meta": "fg:#aaaaaa",
                "env-meter": "fg:#8dff7c",
                "env-miss": "fg:#ffae42",
                "env-ok": "fg:#8dff7c",
                "bar": "fg:#8dff7c",
                "notes": "fg:#ffae42",
                "grid": "fg:#ff00ff",
                "button": "bg:#222222 fg:#00ffff",
                "button.focused": "bg:#00ffff fg:#000000",
            }
        ),
    )
    return app


def main() -> int:
    application = build_app()
    result = application.run()
    return int(result or 0)


def cli_entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli_entry()
