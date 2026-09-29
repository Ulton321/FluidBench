r"""D2Q9 lattice Boltzmann solver: flow past a cylinder.

The classic von Karman vortex street.  It is a good benchmark kernel because
it is memory bound, entirely elementwise plus nine circular shifts, and has no
data-dependent branching -- so a GPU should win once the grid is large enough
to hide kernel launch latency, and the crossover point is the interesting
number.

Lattice layout (D2Q9), index -> direction:

    8   1   2        NW  N  NE
      \ | /
    7 - 0 - 3        W   .  E
      / | \
    6   5   4        SW  S  SE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import Backend

__all__ = [
    "LBMConfig",
    "LBMSolver",
    "equilibrium",
    "NL",
    "CXS",
    "CYS",
    "WEIGHTS",
    "OPPOSITE",
]

#: Number of discrete velocities.
NL = 9

#: Lattice velocities, x then y, in the index order drawn above.
CXS = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
CYS = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])

#: Lattice weights; 4/9 for rest, 1/9 for axial, 1/36 for diagonal.
WEIGHTS = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

#: Index of the reversed direction, used for bounce-back off the obstacle.
OPPOSITE = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])

# The original 100-row reference case used a 13-cell cylinder.  Expressing the
# radius as a fraction of the height keeps the geometry similar when the grid
# is scaled up for throughput measurements.
RADIUS_FRACTION = 0.13


@dataclass
class LBMConfig:
    """Everything that defines a run.  Two configs that compare equal produce
    identical starting data on every backend."""

    nx: int = 400
    ny: int = 100
    tau: float = 0.53
    """Relaxation time.  Kinematic viscosity is (tau - 1/2) / 3, so tau must
    stay above 0.5 or the scheme is unstable."""

    radius: float | None = None
    """Cylinder radius in cells.  Defaults to 13% of the grid height."""

    inflow: float = 0.1
    """Free-stream velocity in lattice units.  The speed of sound is 1/sqrt(3),
    so this is a Mach number of about 0.17 -- the expansion the collision term
    uses is only valid well below 1."""

    rho0: float = 100.0

    noise: float = 0.01
    """Relative velocity jitter applied to the free stream.  Without it the
    wake stays symmetric for a long time and no vortices shed."""

    seed: int = 42
    dtype: str = "float32"

    #: Above this Mach number the second-order equilibrium stops being valid.
    MAX_MACH = 0.3

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("grid must be at least 8x8")
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
    def cylinder_radius(self) -> float:
        return self.radius if self.radius is not None else RADIUS_FRACTION * self.ny

    @property
    def viscosity(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def mach(self) -> float:
        return abs(self.inflow) * np.sqrt(3.0)

    @property
    def reynolds(self) -> float:
        """Based on the cylinder diameter.  Vortices shed above roughly 47."""
        return self.inflow * 2 * self.cylinder_radius / self.viscosity

    @property
    def cells(self) -> int:
        return self.nx * self.ny

    def label(self) -> str:
        return f"{self.nx}x{self.ny}"


def build_obstacle(config: LBMConfig) -> np.ndarray:
    """Boolean mask marking the cylinder, a quarter of the way downstream."""
    y, x = np.ogrid[: config.ny, : config.nx]
    cx, cy = config.nx // 4, config.ny // 2
    return ((x - cx) ** 2 + (y - cy) ** 2) < config.cylinder_radius**2


def equilibrium(rho, ux, uy, cxs, cys, weights):
    """Second-order Maxwell-Boltzmann equilibrium populations.

    Pure broadcast arithmetic, so this runs unchanged on a numpy array or a
    CUDA tensor -- which is why the initial condition and the collision step
    can share it.
    """
    cu = ux[..., None] * cxs + uy[..., None] * cys
    usq = (ux * ux + uy * uy)[..., None]
    return rho[..., None] * weights * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def initial_distribution(config: LBMConfig) -> np.ndarray:
    """Starting populations, built on the host so every backend agrees bit for
    bit before the first step.

    The field starts *at* local equilibrium.  Seeding it with uniform
    populations instead -- as the usual tutorial version does -- puts the state
    far from equilibrium, and since the collision over-relaxes (omega = 1/tau
    is close to 2) the first step overshoots straight into negative
    populations and the run blows up within a few dozen steps.
    """
    dtype = np.dtype(config.dtype)
    rng = np.random.default_rng(config.seed)
    shape = (config.ny, config.nx)

    # Uniform stream in +x, jittered just enough to break the symmetry of the
    # wake so vortices actually shed.
    ux = config.inflow * (1.0 + config.noise * rng.standard_normal(shape))
    uy = config.inflow * config.noise * rng.standard_normal(shape)
    rho = np.full(shape, config.rho0)

    f = equilibrium(rho, ux, uy, CXS.astype(float), CYS.astype(float), WEIGHTS)
    return f.astype(dtype)


class LBMSolver:
    """A single simulation bound to one backend.

    State lives on the backend's device for the whole run; nothing is copied
    back to the host until you ask for it.
    """

    def __init__(self, config: LBMConfig, backend: Backend):
        self.config = config
        self.backend = backend
        dtype = np.dtype(config.dtype)

        self.cxs = backend.asarray(CXS, dtype)
        self.cys = backend.asarray(CYS, dtype)
        self.weights = backend.asarray(WEIGHTS, dtype)
        self.opposite = backend.asarray(OPPOSITE, np.int64)
        self.obstacle = backend.asarray(build_obstacle(config), np.bool_)
        self.f = backend.asarray(initial_distribution(config), dtype)

        # Address the obstacle cells by integer index rather than by boolean
        # mask.  Masked indexing has to count the selected elements before it
        # can size the result, which on CUDA means a device-to-host sync on
        # every single step -- it serialises the whole pipeline and dominates
        # the step time.  These indices are computed once on the host instead.
        obs_y, obs_x = np.nonzero(build_obstacle(config))
        self.obs_y = backend.asarray(obs_y, np.int64)
        self.obs_x = backend.asarray(obs_x, np.int64)

        # Plain Python floats: they promote weakly against float32 arrays in
        # both numpy and torch, so float32 runs stay float32.
        self.inv_tau = 1.0 / config.tau
        self.steps_taken = 0

        # Shift the whole nine-direction loop out of the hot path.
        self._shifts = [(int(CYS[i]), int(CXS[i])) for i in range(NL)]

    def step(self) -> None:
        """Advance one lattice time step: stream, collide, bounce back."""
        backend, f = self.backend, self.f

        # 1. Streaming -- every population hops one cell along its direction.
        for i, shift in enumerate(self._shifts):
            f[:, :, i] = backend.roll(f[:, :, i], shift, (0, 1))

        # 2. Grab the populations that streamed into the cylinder and reverse
        #    them now, before collision touches them.
        bounced = f[self.obs_y, self.obs_x, :][:, self.opposite]

        # 3. Macroscopic density and velocity.
        rho = f.sum(2)
        ux = (f * self.cxs).sum(2) / rho
        uy = (f * self.cys).sum(2) / rho

        # 4. BGK collision -- relax towards the local Maxwellian.
        feq = equilibrium(rho, ux, uy, self.cxs, self.cys, self.weights)
        f -= self.inv_tau * (f - feq)

        # 5. No-slip wall: write the reversed populations back.
        f[self.obs_y, self.obs_x, :] = bounced

        self.steps_taken += 1

    def run(self, steps: int, synchronize: bool = True) -> None:
        """Advance `steps` time steps, waiting for the device by default."""
        for _ in range(steps):
            self.step()
        if synchronize:
            self.backend.synchronize()

    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Density and velocity as host arrays, with the cylinder zeroed."""
        f = self.f
        rho = f.sum(2)
        ux = (f * self.cxs).sum(2) / rho
        uy = (f * self.cys).sum(2) / rho

        be = self.backend
        rho_h = be.to_numpy(rho)
        ux_h, uy_h = be.to_numpy(ux), be.to_numpy(uy)
        mask = be.to_numpy(self.obstacle)
        ux_h = np.where(mask, 0.0, ux_h)
        uy_h = np.where(mask, 0.0, uy_h)
        return rho_h, ux_h, uy_h

    def vorticity(self) -> np.ndarray:
        """Out-of-plane curl of the velocity field, by central differences."""
        _, ux, uy = self.macroscopic()
        dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) * 0.5
        duy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) * 0.5
        vort = duy_dx - dux_dy
        return np.where(self.obstacle_mask, np.nan, vort)

    @property
    def obstacle_mask(self) -> np.ndarray:
        """The cylinder as a host-side boolean array."""
        return self.backend.to_numpy(self.obstacle)

    def total_mass(self) -> float:
        """Summed density -- conserved by the scheme, so a cheap sanity check."""
        return float(self.backend.to_numpy(self.f.sum()))

    def is_finite(self) -> bool:
        """False once the simulation has blown up."""
        return bool(np.isfinite(self.backend.to_numpy(self.f)).all())
