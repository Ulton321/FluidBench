"""FluidBench -- a D2Q9 lattice Boltzmann benchmark for CPU vs GPU.

The same solver runs on every backend, so the timings measure the hardware
and the array library rather than two different algorithms.
"""

from .backends import Backend, available_backends, get_backend
from .benchmark import BenchmarkResult, compare_backends, run_benchmark, sweep
from .lbm import LBMConfig, LBMSolver

__version__ = "1.0.0"

__all__ = [
    "Backend",
    "BenchmarkResult",
    "LBMConfig",
    "LBMSolver",
    "available_backends",
    "compare_backends",
    "get_backend",
    "run_benchmark",
    "sweep",
]
