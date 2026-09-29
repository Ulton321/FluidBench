"""Array backends the solver runs on.

A backend supplies the four primitives that differ between array libraries:
moving data on and off the device, rolling an array, and waiting for queued
work.  Everything else in the solver is plain arithmetic, reductions and
indexing, which numpy, cupy and torch all spell the same way.  Writing the
solver once against this interface is what lets the benchmark claim that the
CPU and the GPU are doing the same work.
"""

from __future__ import annotations

import importlib.util
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "Backend",
    "NumpyBackend",
    "CupyBackend",
    "TorchBackend",
    "available_backends",
    "all_backend_names",
    "get_backend",
    "resolve_names",
]


class Backend:
    """The minimal array interface :class:`~fluidbench.lbm.LBMSolver` needs."""

    name: str = "unknown"
    is_gpu: bool = False

    def asarray(self, a, dtype=None):
        """Copy a numpy array onto the backend's device."""
        raise NotImplementedError

    def to_numpy(self, a) -> np.ndarray:
        """Copy a backend array back to host memory as numpy."""
        raise NotImplementedError

    def roll(self, a, shifts: Sequence[int], axes: Sequence[int]):
        """Circular shift along several axes at once."""
        raise NotImplementedError

    def synchronize(self) -> None:
        """Block until queued work has finished.  A no-op on the CPU."""

    def describe(self) -> str:
        """One line naming the hardware the backend will use."""
        return self.name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}>"


class NumpyBackend(Backend):
    """Single-threaded CPU reference.  Always available."""

    name = "numpy"
    is_gpu = False

    def asarray(self, a, dtype=None):
        return np.array(a, dtype=dtype, copy=True)

    def to_numpy(self, a) -> np.ndarray:
        return np.asarray(a)

    def roll(self, a, shifts, axes):
        return np.roll(a, shifts, axis=axes)

    def describe(self) -> str:
        import platform

        return f"numpy {np.__version__} on {platform.processor() or platform.machine()}"


class CupyBackend(Backend):
    """CUDA via CuPy -- a near drop-in replacement for numpy."""

    name = "cupy"
    is_gpu = True

    def __init__(self, device: int = 0):
        import cupy as cp

        self.cp = cp
        self.device_id = device
        cp.cuda.Device(device).use()

    def asarray(self, a, dtype=None):
        return self.cp.asarray(np.asarray(a), dtype=dtype)

    def to_numpy(self, a) -> np.ndarray:
        return self.cp.asnumpy(a)

    def roll(self, a, shifts, axes):
        return self.cp.roll(a, shifts, axis=axes)

    def synchronize(self) -> None:
        self.cp.cuda.runtime.deviceSynchronize()

    def describe(self) -> str:
        props = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
        gpu = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        return f"cupy {self.cp.__version__} on {gpu}"


# Populated on first use -- importing torch is slow, so it stays lazy.
_TORCH_DTYPES: dict | None = None


def _torch_dtype(torch, dtype):
    """Map a numpy dtype onto the matching torch dtype."""
    global _TORCH_DTYPES
    if _TORCH_DTYPES is None:
        _TORCH_DTYPES = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.bool_): torch.bool,
        }
    key = np.dtype(dtype)
    if key not in _TORCH_DTYPES:
        raise TypeError(f"no torch dtype for {key}")
    return _TORCH_DTYPES[key]


class TorchBackend(Backend):
    """PyTorch, on either the CPU or a CUDA device.

    The CPU variant is worth benchmarking in its own right: it multi-threads
    the elementwise kernels that numpy runs on a single core, so it separates
    "GPU is faster" from "numpy is single-threaded".
    """

    def __init__(self, device: str = "cuda"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.is_gpu = self.device.type == "cuda"
        self.name = f"torch-{self.device.type}"

    def asarray(self, a, dtype=None):
        host = np.ascontiguousarray(a)
        tensor = self.torch.from_numpy(host).clone()
        if dtype is not None:
            tensor = tensor.to(_torch_dtype(self.torch, dtype))
        return tensor.to(self.device)

    def to_numpy(self, a) -> np.ndarray:
        return a.detach().cpu().numpy()

    def roll(self, a, shifts, axes):
        return self.torch.roll(a, tuple(shifts), dims=tuple(axes))

    def synchronize(self) -> None:
        if self.is_gpu:
            self.torch.cuda.synchronize(self.device)

    def set_num_threads(self, threads: int) -> None:
        """Pin the CPU thread count so repeated runs are comparable."""
        self.torch.set_num_threads(threads)

    def describe(self) -> str:
        version = self.torch.__version__
        if self.is_gpu:
            gpu = self.torch.cuda.get_device_name(self.device)
            cuda = self.torch.version.cuda or "?"
            return f"torch {version} (CUDA {cuda}) on {gpu}"
        return f"torch {version} on CPU ({self.torch.get_num_threads()} threads)"


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def _cuda_torch_ready() -> bool:
    if not _installed("torch"):
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - driver problems
        return False


def _cupy_ready() -> bool:
    if not _installed("cupy"):
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - driver problems
        return False


# name -> (availability probe, constructor).  Order matters: `gpu` resolves to
# the first available GPU entry, and CuPy comes first because it is the
# thinner layer over CUDA.
_REGISTRY: dict[str, tuple[Callable[[], bool], Callable[[], Backend]]] = {
    "numpy": (lambda: True, NumpyBackend),
    "cupy": (_cupy_ready, CupyBackend),
    "torch-cuda": (_cuda_torch_ready, lambda: TorchBackend("cuda")),
    "torch-cpu": (lambda: _installed("torch"), lambda: TorchBackend("cpu")),
}

_ALIASES = {"cpu": "numpy", "np": "numpy", "cuda": "gpu"}


def all_backend_names() -> list[str]:
    """Every backend FluidBench knows about, installed or not."""
    return list(_REGISTRY)


def available_backends() -> list[str]:
    """The backends that can actually run on this machine."""
    return [name for name, (probe, _) in _REGISTRY.items() if probe()]


def _first_gpu() -> str:
    for name in _REGISTRY:
        probe, _ = _REGISTRY[name]
        if name != "numpy" and probe() and name != "torch-cpu":
            return name
    raise RuntimeError(
        "no GPU backend available -- install cupy, or a CUDA build of torch "
        "(https://pytorch.org/get-started/locally/)"
    )


def get_backend(name: str) -> Backend:
    """Build a backend by name.  Accepts `cpu`, `gpu` and a few aliases."""
    key = _ALIASES.get(name.lower(), name.lower())
    if key == "gpu":
        key = _first_gpu()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown backend {name!r}; choose from {', '.join(all_backend_names())}"
        )
    probe, build = _REGISTRY[key]
    if not probe():
        raise RuntimeError(f"backend {key!r} is not available on this machine")
    return build()


def resolve_names(names: Sequence[str]) -> list[str]:
    """Expand a CLI backend list, where `all` means everything available."""
    resolved: list[str] = []
    for name in names:
        if name.lower() == "all":
            resolved.extend(available_backends())
        else:
            key = _ALIASES.get(name.lower(), name.lower())
            resolved.append(_first_gpu() if key == "gpu" else key)
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(resolved))
