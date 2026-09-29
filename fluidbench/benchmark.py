"""Timing harness and cross-backend correctness checks.

Three things make the numbers trustworthy, and all three are easy to get
wrong:

* **Warm up first.**  The first GPU steps pay for context creation, kernel
  loading and allocator growth, and the clocks have not ramped yet.
* **Synchronize before stopping the clock.**  CUDA calls are asynchronous, so
  a naive timer measures how fast Python can queue work, not how fast the GPU
  finishes it.
* **Start from identical data.**  The initial condition is generated once on
  the host from a seeded generator, so every backend gets the same bytes.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .backends import Backend, get_backend
from .lbm import NL, LBMConfig, LBMSolver

__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "run_benchmark",
    "sweep",
    "compare_backends",
    "working_set_mb",
]


def working_set_mb(config: LBMConfig) -> float:
    """Rough peak device memory for one solver, in MiB.

    The state itself is one (ny, nx, 9) array, but a step materialises the
    equilibrium and a couple of other full-size temporaries, so budget for
    about four of them plus change.
    """
    itemsize = np.dtype(config.dtype).itemsize
    full = config.cells * NL * itemsize
    planar = config.cells * itemsize
    return (4 * full + 6 * planar) / (1024**2)


@dataclass
class BenchmarkResult:
    """One backend, one grid size, one set of timings."""

    backend: str
    device: str
    dtype: str
    nx: int
    ny: int
    steps: int
    warmup: int
    repeats: int
    seconds: float
    """Best wall-clock time over the repeats."""

    seconds_all: list[float] = field(default_factory=list)
    finite: bool = True
    """False if the run produced NaN or inf -- the timing is still valid, but
    the physics is not."""

    @property
    def cells(self) -> int:
        return self.nx * self.ny

    @property
    def mlups(self) -> float:
        """Million lattice updates per second, the standard LBM metric."""
        return self.cells * self.steps / (self.seconds * 1e6)

    @property
    def step_ms(self) -> float:
        return self.seconds * 1e3 / self.steps

    @property
    def bandwidth_gbs(self) -> float:
        """Effective memory bandwidth under a minimum-traffic model.

        Streaming has to read and write all nine populations per cell per
        step, so this is a lower bound; the real traffic is higher because
        this implementation materialises temporaries.
        """
        itemsize = np.dtype(self.dtype).itemsize
        moved = 2 * NL * itemsize * self.cells * self.steps
        return moved / self.seconds / 1e9

    @property
    def spread(self) -> float:
        """Slowest/fastest repeat.  Well above 1.0 means noisy timings."""
        if not self.seconds_all:
            return 1.0
        return max(self.seconds_all) / min(self.seconds_all)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            mlups=self.mlups,
            step_ms=self.step_ms,
            bandwidth_gbs=self.bandwidth_gbs,
            spread=self.spread,
        )
        return data


def run_benchmark(
    backend: Backend | str,
    config: LBMConfig,
    steps: int = 200,
    warmup: int = 20,
    repeats: int = 3,
) -> BenchmarkResult:
    """Time `steps` lattice steps, best of `repeats`."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    if isinstance(backend, str):
        backend = get_backend(backend)

    solver = LBMSolver(config, backend)

    if warmup > 0:
        solver.run(warmup)

    times: list[float] = []
    for _ in range(repeats):
        backend.synchronize()
        start = time.perf_counter()
        solver.run(steps, synchronize=False)
        backend.synchronize()
        times.append(time.perf_counter() - start)

    return BenchmarkResult(
        backend=backend.name,
        device=backend.describe(),
        dtype=config.dtype,
        nx=config.nx,
        ny=config.ny,
        steps=steps,
        warmup=warmup,
        repeats=repeats,
        seconds=min(times),
        seconds_all=times,
        finite=solver.is_finite(),
    )


def sweep(
    backend_names: list[str],
    sizes: list[tuple[int, int]],
    base: LBMConfig | None = None,
    steps: int = 100,
    warmup: int = 10,
    repeats: int = 3,
    on_result=None,
) -> list[BenchmarkResult]:
    """Benchmark every backend at every grid size.

    Backends are built once and reused across sizes, so CUDA context creation
    is not charged to the first measurement.  `on_result` is called with each
    result as it lands, which lets the CLI stream a table instead of going
    quiet for a minute.
    """
    base = base or LBMConfig()
    backends = {name: get_backend(name) for name in backend_names}

    results: list[BenchmarkResult] = []
    for nx, ny in sizes:
        config = LBMConfig(
            nx=nx,
            ny=ny,
            tau=base.tau,
            radius=base.radius,
            inflow=base.inflow,
            rho0=base.rho0,
            noise=base.noise,
            seed=base.seed,
            dtype=base.dtype,
        )
        for name, backend in backends.items():
            result = run_benchmark(backend, config, steps, warmup, repeats)
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results


@dataclass
class ComparisonResult:
    """How far one backend drifted from the reference."""

    backend: str
    reference: str
    steps: int
    dtype: str
    max_abs_rho: float
    max_rel_rho: float
    max_abs_u: float
    tolerance: float
    finite: bool

    @property
    def passed(self) -> bool:
        return self.finite and self.max_rel_rho <= self.tolerance


def compare_backends(
    backend_names: list[str],
    config: LBMConfig,
    steps: int = 100,
    reference: str = "numpy",
    tolerance: float = 1e-6,
) -> list[ComparisonResult]:
    """Check that the other backends track the reference one.

    Exact agreement is not on offer: floating-point reductions are not
    associative, and a GPU sums them in a different order.  What matters is
    that the gap stays at rounding level rather than growing into different
    physics -- so run this in float64, where the tolerance can be tight.
    """
    ref_backend = get_backend(reference)
    ref_solver = LBMSolver(config, ref_backend)
    ref_solver.run(steps)
    ref_rho, ref_ux, ref_uy = ref_solver.macroscopic()

    scale = np.abs(ref_rho).max() or 1.0

    results: list[ComparisonResult] = []
    for name in backend_names:
        if name == reference:
            continue
        solver = LBMSolver(config, get_backend(name))
        solver.run(steps)
        rho, ux, uy = solver.macroscopic()

        abs_rho = float(np.abs(rho - ref_rho).max())
        abs_u = float(
            max(np.abs(ux - ref_ux).max(), np.abs(uy - ref_uy).max())
        )
        results.append(
            ComparisonResult(
                backend=name,
                reference=reference,
                steps=steps,
                dtype=config.dtype,
                max_abs_rho=abs_rho,
                max_rel_rho=abs_rho / scale,
                max_abs_u=abs_u,
                tolerance=tolerance,
                finite=bool(np.isfinite(rho).all() and np.isfinite(ux).all()),
            )
        )
    return results
