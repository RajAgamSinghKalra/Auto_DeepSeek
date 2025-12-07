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
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Sequence

from colorama import Fore, Style, init as color_init

from system_profile import SystemProfile, describe_profile, detect_system_profile

color_init(autoreset=True)

SUPPORTED_MODELS: List[str] = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "NousResearch/Meta-Llama-3-8B-Instruct",
]

REQUIRED_PACKAGES = ["torch", "transformers", "selenium", "requests", "psutil", "GPUtil"]


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


MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def has_package(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def check_installation() -> InstallationReport:
    missing = [pkg for pkg in REQUIRED_PACKAGES if not has_package(pkg)]
    has_firefox = bool(shutil.which("firefox") or shutil.which("Mozilla Firefox"))
    notes: List[str] = []
    if not has_firefox:
        notes.append("Firefox not detected (required for browsing features)")
    return InstallationReport(ready=not missing, missing_packages=missing, has_firefox=has_firefox, notes=notes)


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


def build_install_plan(profile: SystemProfile) -> List[InstallStep]:
    steps: List[InstallStep] = []

    if profile.os_name == "Linux":
        steps.append(
            InstallStep(
                name="System packages",
                commands=["sudo apt update", "sudo apt install -y python3-venv git firefox"],
                note="Adapt these commands if you are not using an apt-based distribution.",
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
                    "choco install -y cuda --pre || echo 'Install CUDA manually if needed'",
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
        )
    )

    steps.append(
        InstallStep(
            name="Verify PyTorch hardware backend",
            commands=[
                f"{sys.executable} - <<'PY'\nimport torch;print('CUDA', torch.cuda.is_available());print('MPS', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())\nPY"
            ],
            note="Ensure the reported backend matches your hardware.",
        )
    )

    return steps


def run_commands(commands: Sequence[str]) -> None:
    for command in commands:
        print(f"\n{Fore.CYAN}→ {command}{Style.RESET_ALL}")
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"{Fore.GREEN}✓ Success{Style.RESET_ALL}")
        except subprocess.CalledProcessError as exc:
            print(f"{Fore.RED}✗ Failed with code {exc.returncode}{Style.RESET_ALL}")
            break


def show_menu(status: InstallationReport, selected_model: Optional[str]) -> None:
    status_icon = f"{Fore.GREEN}●{Style.RESET_ALL}" if status.ready else f"{Fore.YELLOW}●{Style.RESET_ALL}"
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
            """
        ).strip()
    )


def pick_model(current: Optional[str]) -> str:
    print("\nAvailable models:")
    for idx, model in enumerate(SUPPORTED_MODELS, start=1):
        print(f"  [{idx}] {model}")
    print("  [C] Custom model ID")
    choice = input("Select a model: ").strip().lower()
    if choice == "c":
        custom = input("Enter a Hugging Face model ID: ").strip()
        return custom or current or SUPPORTED_MODELS[0]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(SUPPORTED_MODELS):
            return SUPPORTED_MODELS[idx]
    except ValueError:
        pass
    print("Keeping current selection.")
    return current or SUPPORTED_MODELS[0]


def install_model(model_id: str) -> None:
    print(f"\nPreparing to download model: {model_id}")
    confirm = input("This may be large. Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Skipping model download.")
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Transformers is not installed. Please finish setup first.")
        return

    cache_dir = MODEL_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading tokenizer...")
    AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    print("Downloading model weights... (this can take a while)")
    AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    print(f"{Fore.GREEN}Model cached successfully.{Style.RESET_ALL}")


def guided_setup(profile: SystemProfile) -> None:
    steps = build_install_plan(profile)
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
        else:
            print("Skipped.")



def launch_agent(selected_model: Optional[str]) -> None:
    if not selected_model:
        print("Select a model first from the LLM menu.")
        return
    cmd = [sys.executable, "autodeepseek.py", "--model", selected_model]
    print(f"\nLaunching agent with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def main() -> int:
    profile = detect_system_profile()
    status = check_installation()
    selected_model: Optional[str] = None

    while True:
        show_menu(status, selected_model)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            print("\nDetected system profile:\n")
            print(describe_profile(profile))
            input("\nPress Enter to return to menu...")
        elif choice == "2":
            guided_setup(profile)
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
            if not status.ready:
                print("Environment not ready. Resolve missing packages first.")
                continue
            launch_agent(selected_model)
        elif choice == "0":
            print("Exiting TUI. Goodbye!")
            return 0
        else:
            print("Invalid option. Try again.")


def cli_entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli_entry()
