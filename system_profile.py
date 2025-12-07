"""System detection helpers for AutoDeepSeek TUI."""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency for detection
    torch = None

try:
    import GPUtil  # type: ignore
except Exception:  # pragma: no cover - optional dependency for detection
    GPUtil = None


@dataclass
class GPUInfo:
    name: str
    vendor: str
    memory_total_gb: Optional[float] = None


@dataclass
class SystemProfile:
    os_name: str
    arch: str
    python_version: str
    gpu: Optional[GPUInfo] = None
    accelerators: List[str] = field(default_factory=list)


def _detect_gpu_via_gputil() -> Optional[GPUInfo]:
    if not GPUtil:
        return None
    try:
        gpus = GPUtil.getGPUs()
    except Exception:
        return None
    if not gpus:
        return None
    gpu = gpus[0]
    vendor = "NVIDIA" if "NVIDIA" in gpu.name.upper() else "AMD" if "AMD" in gpu.name.upper() else "Unknown"
    return GPUInfo(name=gpu.name, vendor=vendor, memory_total_gb=getattr(gpu, "memoryTotal", None))


def _detect_gpu_via_torch() -> Optional[GPUInfo]:
    if not torch:
        return None
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return GPUInfo(name=name, vendor="NVIDIA")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return GPUInfo(name="Apple Silicon", vendor="Apple")
    return None


def detect_system_profile() -> SystemProfile:
    os_name = platform.system()
    arch = platform.machine()
    python_version = platform.python_version()

    gpu = _detect_gpu_via_torch() or _detect_gpu_via_gputil()

    accelerators: List[str] = []
    if torch:
        if torch.cuda.is_available():
            accelerators.append("CUDA")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerators.append("MPS")
        if torch.backends.openmp.is_available() if hasattr(torch.backends, "openmp") else False:
            accelerators.append("CPU-Optimized")

    if not accelerators:
        accelerators.append("CPU")

    return SystemProfile(
        os_name=os_name,
        arch=arch,
        python_version=python_version,
        gpu=gpu,
        accelerators=accelerators,
    )


def describe_profile(profile: SystemProfile) -> str:
    gpu_desc = "None detected"
    if profile.gpu:
        mem = f" ({profile.gpu.memory_total_gb:.1f} GB)" if profile.gpu.memory_total_gb else ""
        gpu_desc = f"{profile.gpu.name} [{profile.gpu.vendor}]{mem}"
    accel = ", ".join(profile.accelerators)
    return (
        f"OS: {profile.os_name}\n"
        f"Architecture: {profile.arch}\n"
        f"Python: {profile.python_version}\n"
        f"GPU: {gpu_desc}\n"
        f"Accelerators: {accel}"
    )
