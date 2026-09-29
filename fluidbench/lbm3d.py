r"""D3Q19 lattice Boltzmann: the 3D solver, and flow past a sphere.

The 2D solver in :mod:`fluidbench.lbm` is the benchmark's reference workload
and does not change.  This is its three-dimensional sibling, written for the
cases 2D cannot represent -- anything whose interesting structure is a
streamwise vortex.

Two things are deliberately different from the 2D code, and both were settled
by measurement rather than taste (see ``docs/3d-plan.md``):

**State is structure-of-arrays.**  ``f`` has shape ``(19, nz, ny, nx)``, so
each direction is its own contiguous block.  The 2D solver keeps the direction
last, where ``f[..., i]`` reads one float every 9 and coalescing is lost; at
19 directions that stride costs 2.8x on a CUDA device.  In 2D the same change
would be a *pessimisation* below about half a million cells, because the
per-direction loop issues far more kernels and small grids are launch-bound --
which is the benchmark's own thesis turning up in its own implementation.  3D
grids are never that small.

**The equilibrium is never materialised.**  The 2D step builds ``feq`` as one
full ``(ny, nx, 9)`` array.  The same shape in 3D costs 554 bytes per cell and
caps an 8 GB card at 13 M cells, which is too coarse to be worth running.
Looping over the 19 directions and keeping every temporary planar costs 196
B/cell instead and nearly triples the domain.

Lattice layout (D3Q19): the rest vector, six axial neighbours, and the twelve
face diagonals.  Opposites are adjacent pairs, so ``OPPOSITE`` is just a swap
of each pair and bounce-back indexing stays trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import Backend

__all__ = [
    "LBM3DConfig",
    "LBM3DSolver",
    "equilibrium3d",
    "build_sphere",
    "taylor_green_velocity",
    "NL3",
    "C3",
    "W3",
    "OPPOSITE3",
]

#: Number of discrete velocities.
NL3 = 19

#: Lattice velocities as (cx, cy, cz).  Rest, then six axial, then twelve
#: face diagonals -- each listed immediately beside its own reverse.
C3 = np.array(
    [
        (0, 0, 0),
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (-1, -1, 0),
        (1, -1, 0), (-1, 1, 0),
        (1, 0, 1), (-1, 0, -1),
        (1, 0, -1), (-1, 0, 1),
        (0, 1, 1), (0, -1, -1),
        (0, 1, -1), (0, -1, 1),
    ],
    dtype=np.int64,
)

#: Weights: 1/3 for rest, 1/18 for the axial pairs, 1/36 for the diagonals.
W3 = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)

#: Index of the reversed direction.  Pairs are adjacent by construction, so
#: this is 0, then each (odd, even) pair swapped.
OPPOSITE3 = np.array([0] + [i + 1 if i % 2 else i - 1 for i in range(1, NL3)])

#: Speed of sound squared, in lattice units.
CS2 = 1.0 / 3.0

#: Sphere radius as a fraction of the smaller cross-stream dimension, chosen
#: to match the 2D cylinder's blockage so the two cases stay comparable.
RADIUS_FRACTION = 0.13


@dataclass
class LBM3DConfig:
    """Everything that defines a 3D run.

    ``nx`` is the streamwise direction, ``ny`` and ``nz`` the cross-stream
    ones.  Two configs that compare equal produce identical starting data on
    every backend.
    """

    nx: int = 128
    ny: int = 64
    nz: int = 64
    tau: float = 0.6
    """Relaxation time.  Kinematic viscosity is (tau - 1/2)/3, so tau must
    stay above 0.5 or the scheme is unstable.  The default is looser than the
    2D solver's 0.53: 3D grids are coarser per unit length, and BGK near the
    stability limit has less margin when there are 19 directions feeding a
    single relaxation."""

    radius: float | None = None
    """Sphere radius in cells.  Defaults to 13% of the smaller cross-stream
    dimension, matching the 2D cylinder's blockage."""

    inflow: float = 0.05
    """Free-stream velocity in lattice units.  Lower than the 2D default:
    the same Mach argument applies, and 3D runs are long enough that the
    accumulated compressibility error is worth keeping small."""

    rho0: float = 1.0
    noise: float = 0.01
    seed: int = 42
    dtype: str = "float32"
    obstacle: bool = True
    """False gives an empty periodic box -- what the validation cases want."""

    MAX_MACH = 0.3

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 8:
            raise ValueError("grid must be at least 8x8x8")
        if self.tau <= 0.5:
            raise ValueError(f"tau must exceed 0.5 for stability, got {self.tau}")
        if not 0.0 <= self.mach <= self.MAX_MACH:
            raise ValueError(
                f"inflow {self.inflow} is Mach {self.mach:.2f}; keep it under "
                f"{self.MAX_MACH} or the simulation will not stay stable"
            )
        if np.dtype(self.dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be float32 or float64")

    @property
    def sphere_radius(self) -> float:
        if self.radius is not None:
            return self.radius
        return RADIUS_FRACTION * min(self.ny, self.nz)

    @property
    def viscosity(self) -> float:
        return (self.tau - 0.5) * CS2

    @property
    def mach(self) -> float:
        return abs(self.inflow) / np.sqrt(CS2)

    @property
    def reynolds(self) -> float:
        """Based on the sphere diameter.  The wake goes unsteady above ~270."""
        return self.inflow * 2 * self.sphere_radius / self.viscosity

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    def label(self) -> str:
        return f"{self.nx}x{self.ny}x{self.nz}"


def working_set_mb(config: LBM3DConfig) -> float:
    """Rough peak device memory for one solver, in MiB.

    Two full ``(19, nz, ny, nx)`` buffers for the ping-pong, plus roughly
    seven planar temporaries live at once during a step.  Measured at 196
    bytes per cell in float32, which this model reproduces.
    """
    itemsize = np.dtype(config.dtype).itemsize
    planar = config.cells * itemsize
    return (2 * NL3 * planar + 7 * planar) / (1024**2)


def build_sphere(config: LBM3DConfig) -> np.ndarray:
    """Boolean mask marking the sphere, a quarter of the way downstream."""
    z, y, x = np.ogrid[: config.nz, : config.ny, : config.nx]
    cx, cy, cz = config.nx // 4, config.ny // 2, config.nz // 2
    return ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) < config.sphere_radius**2


def equilibrium3d(rho, ux, uy, uz, index: int):
    """Second-order equilibrium for one direction.

    One direction at a time, on purpose: the caller loops, so every temporary
    stays planar.  Pure broadcast arithmetic, so it runs unchanged on a numpy
    array or a CUDA tensor.
    """
    cx, cy, cz = (float(v) for v in C3[index])
    cu = ux * cx + uy * cy + uz * cz
    usq = ux * ux + uy * uy + uz * uz
    return rho * float(W3[index]) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def taylor_green_velocity(config: LBM3DConfig, amplitude: float | None = None):
    """The decaying Taylor-Green vortex, uniform in z.

    The 2D form extended through the third dimension, which is the useful
    shape for checking a 3D code: it has a closed-form solution, so the error
    is measurable rather than eyeballed, *and* the field must stay uniform in
    z -- any drift there is a bug in the z half of the streaming.

        ux = -U cos(kx x) sin(ky y) exp(-2 nu k^2 t)
        uy =  U sin(kx x) cos(ky y) exp(-2 nu k^2 t)

    Returns (ux, uy, uz, rho) at t = 0, and the decay rate.
    """
    amplitude = config.inflow if amplitude is None else amplitude
    kx = 2.0 * np.pi / config.nx
    ky = 2.0 * np.pi / config.ny
    z, y, x = np.ogrid[: config.nz, : config.ny, : config.nx]

    ux = -amplitude * np.cos(kx * x) * np.sin(ky * y) * np.ones_like(z, dtype=float)
    uy = amplitude * np.sin(kx * x) * np.cos(ky * y) * np.ones_like(z, dtype=float)
    uz = np.zeros_like(ux)

    # The pressure that goes with it, to the same order as the equilibrium.
    rho = config.rho0 - (
        0.75 * amplitude**2 * config.rho0
        * (np.cos(2 * kx * x) + np.cos(2 * ky * y))
        * np.ones_like(z, dtype=float)
    )
    decay = config.viscosity * (kx**2 + ky**2)
    return ux, uy, uz, rho, decay


def initial_distribution(config: LBM3DConfig) -> np.ndarray:
    """Starting populations at local equilibrium, shaped (19, nz, ny, nx).

    Starting *at* equilibrium rather than at ``f = 1 + noise`` for the same
    reason the 2D solver does: the collision over-relaxes, and a state far
    from equilibrium overshoots into negative populations on the first step.
    """
    dtype = np.dtype(config.dtype)
    rng = np.random.default_rng(config.seed)
    shape = (config.nz, config.ny, config.nx)

    ux = config.inflow * (1.0 + config.noise * rng.standard_normal(shape))
    uy = config.inflow * config.noise * rng.standard_normal(shape)
    uz = config.inflow * config.noise * rng.standard_normal(shape)
    rho = np.full(shape, float(config.rho0))

    return populations_from(rho, ux, uy, uz, dtype)


def populations_from(rho, ux, uy, uz, dtype) -> np.ndarray:
    """Equilibrium populations for a given macroscopic field, on the host."""
    f = np.empty((NL3, *rho.shape), dtype=dtype)
    for i in range(NL3):
        f[i] = equilibrium3d(rho, ux, uy, uz, i)
    return f


class LBM3DSolver:
    """A single 3D simulation bound to one backend.

    State lives on the backend's device for the whole run; nothing is copied
    back to the host until you ask for it.
    """

    def __init__(self, config: LBM3DConfig, backend: Backend,
                 populations: np.ndarray | None = None):
        self.config = config
        self.backend = backend
        dtype = np.dtype(config.dtype)

        start = initial_distribution(config) if populations is None \
            else np.ascontiguousarray(populations, dtype=dtype)
        if start.shape != (NL3, config.nz, config.ny, config.nx):
            raise ValueError(
                f"populations must be {(NL3, config.nz, config.ny, config.nx)}, "
                f"got {start.shape}"
            )

        self.f = backend.asarray(start, dtype)
        self.f2 = backend.asarray(np.empty_like(start), dtype)

        if config.obstacle:
            mask = build_sphere(config)
            self.obstacle = backend.asarray(mask, np.bool_)
            # Integer indices rather than a boolean mask, for the reason the
            # 2D solver documents: masked indexing has to count the selected
            # elements before it can size its output, which on CUDA is a
            # device-to-host sync on every step.
            oz, oy, ox = np.nonzero(mask)
            self.oz = backend.asarray(oz, np.int64)
            self.oy = backend.asarray(oy, np.int64)
            self.ox = backend.asarray(ox, np.int64)
            self.has_obstacle = oz.size > 0
        else:
            self.obstacle = None
            self.has_obstacle = False

        self.inv_tau = 1.0 / config.tau
        self.steps_taken = 0

        # Hoisted out of the hot path: the roll for each direction, and which
        # directions carry a component along each axis.  cx is always 0 or
        # +-1, so the momentum sums are adds and subtracts, never multiplies.
        self._shifts = [(int(c[2]), int(c[1]), int(c[0])) for c in C3]
        self._plus = [[i for i in range(NL3) if C3[i, d] > 0] for d in range(3)]
        self._minus = [[i for i in range(NL3) if C3[i, d] < 0] for d in range(3)]
        self._opposite = [int(v) for v in OPPOSITE3]

    # -- the step ---------------------------------------------------------

    def _axis_momentum(self, f, axis: int):
        """Sum of populations along one axis, as adds and subtracts."""
        total = f[self._plus[axis][0]] - f[self._minus[axis][0]]
        for i in self._plus[axis][1:]:
            total = total + f[i]
        for i in self._minus[axis][1:]:
            total = total - f[i]
        return total

    def step(self) -> None:
        """Advance one lattice time step: stream, collide, bounce back."""
        be = self.backend
        src, dst = self.f, self.f2

        # 1. Streaming -- every population hops one cell along its direction.
        #    Periodic in all three axes, as np.roll is.
        for i in range(NL3):
            dst[i] = be.roll(src[i], self._shifts[i], (0, 1, 2))

        # 2. Grab the populations that streamed into the sphere and reverse
        #    them now, before collision touches them.
        bounced = None
        if self.has_obstacle:
            oz, oy, ox = self.oz, self.oy, self.ox
            bounced = [dst[self._opposite[i]][oz, oy, ox] for i in range(NL3)]

        # 3. Macroscopic density and velocity, planar temporaries only.
        rho = dst[0] + dst[1]
        for i in range(2, NL3):
            rho = rho + dst[i]
        ux = self._axis_momentum(dst, 0) / rho
        uy = self._axis_momentum(dst, 1) / rho
        uz = self._axis_momentum(dst, 2) / rho

        # 4. BGK collision, one direction at a time so feq is never a full
        #    (19, nz, ny, nx) array.  An MRT or regularized operator would
        #    replace exactly this loop and nothing else.
        usq = ux * ux + uy * uy + uz * uz
        for i in range(NL3):
            cx, cy, cz = (float(v) for v in C3[i])
            cu = ux * cx + uy * cy + uz * cz
            feq = rho * float(W3[i]) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
            dst[i] = dst[i] - self.inv_tau * (dst[i] - feq)

        # 5. No-slip wall: write the reversed populations back.
        if bounced is not None:
            for i in range(NL3):
                dst[i][self.oz, self.oy, self.ox] = bounced[i]

        self.f, self.f2 = dst, src
        self.steps_taken += 1

    def run(self, steps: int, synchronize: bool = True) -> None:
        """Advance `steps` time steps, waiting for the device by default."""
        for _ in range(steps):
            self.step()
        if synchronize:
            self.backend.synchronize()

    # -- reading the state -------------------------------------------------

    def macroscopic(self):
        """Density and velocity as host arrays, with the sphere zeroed."""
        be, f = self.backend, self.f
        rho = f[0] + f[1]
        for i in range(2, NL3):
            rho = rho + f[i]
        ux = self._axis_momentum(f, 0) / rho
        uy = self._axis_momentum(f, 1) / rho
        uz = self._axis_momentum(f, 2) / rho

        rho_h = be.to_numpy(rho)
        ux_h, uy_h, uz_h = be.to_numpy(ux), be.to_numpy(uy), be.to_numpy(uz)
        if self.obstacle is not None:
            mask = be.to_numpy(self.obstacle)
            ux_h = np.where(mask, 0.0, ux_h)
            uy_h = np.where(mask, 0.0, uy_h)
            uz_h = np.where(mask, 0.0, uz_h)
        return rho_h, ux_h, uy_h, uz_h

    def vorticity_magnitude(self) -> np.ndarray:
        """|curl u|, by central differences.  The field worth looking at in
        3D -- a single component means little once the wake is not planar."""
        _, ux, uy, uz = self.macroscopic()
        duz_dy = (np.roll(uz, -1, axis=1) - np.roll(uz, 1, axis=1)) * 0.5
        duy_dz = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) * 0.5
        dux_dz = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) * 0.5
        duz_dx = (np.roll(uz, -1, axis=2) - np.roll(uz, 1, axis=2)) * 0.5
        duy_dx = (np.roll(uy, -1, axis=2) - np.roll(uy, 1, axis=2)) * 0.5
        dux_dy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) * 0.5
        wx, wy, wz = duz_dy - duy_dz, dux_dz - duz_dx, duy_dx - dux_dy
        mag = np.sqrt(wx * wx + wy * wy + wz * wz)
        if self.obstacle is not None:
            mag = np.where(self.backend.to_numpy(self.obstacle), np.nan, mag)
        return mag

    @property
    def obstacle_mask(self) -> np.ndarray:
        return self.backend.to_numpy(self.obstacle)

    def total_mass(self) -> float:
        """Summed density -- conserved by the scheme, so a cheap check."""
        be = self.backend
        return float(sum(be.to_numpy(self.f[i]).sum() for i in range(NL3)))

    def is_finite(self) -> bool:
        """False once the simulation has blown up."""
        be = self.backend
        return all(bool(np.isfinite(be.to_numpy(self.f[i])).all()) for i in range(NL3))
