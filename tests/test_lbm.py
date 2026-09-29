"""Correctness checks for the solver and the benchmark harness.

Run from the FluidBench directory:

    python -m pytest tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fluidbench.backends import available_backends, get_backend  # noqa: E402
from fluidbench.benchmark import compare_backends, run_benchmark, working_set_mb  # noqa: E402
from fluidbench.lbm import (  # noqa: E402
    CXS,
    CYS,
    NL,
    OPPOSITE,
    WEIGHTS,
    LBMConfig,
    LBMSolver,
    build_obstacle,
    equilibrium,
)

SMALL = LBMConfig(nx=80, ny=40, dtype="float64")

# Every backend except numpy, which is the reference the others are checked against.
OTHER_BACKENDS = [name for name in available_backends() if name != "numpy"]


# --- lattice constants -------------------------------------------------------


def test_weights_sum_to_one():
    assert WEIGHTS.sum() == pytest.approx(1.0)

def test_lattice_is_symmetric():
    # Zero net momentum at rest, and a unit second moment: both are required
    # for the scheme to recover Navier-Stokes.
    assert (WEIGHTS * CXS).sum() == pytest.approx(0.0)
    assert (WEIGHTS * CYS).sum() == pytest.approx(0.0)
    assert (WEIGHTS * CXS * CXS).sum() == pytest.approx(1 / 3)
    assert (WEIGHTS * CYS * CYS).sum() == pytest.approx(1 / 3)
    assert (WEIGHTS * CXS * CYS).sum() == pytest.approx(0.0)


def test_opposite_directions_reverse_velocity():
    assert (CXS[OPPOSITE] == -CXS).all()
    assert (CYS[OPPOSITE] == -CYS).all()
    assert (OPPOSITE[OPPOSITE] == np.arange(NL)).all()


# --- configuration -----------------------------------------------------------


def test_unstable_tau_is_rejected():
    with pytest.raises(ValueError, match="tau"):
        LBMConfig(tau=0.5)


def test_tiny_grid_is_rejected():
    with pytest.raises(ValueError, match="8x8"):
        LBMConfig(nx=4, ny=4)


def test_viscosity_follows_tau():
    assert LBMConfig(tau=0.53).viscosity == pytest.approx(0.01)


def test_supersonic_inflow_is_rejected():
    with pytest.raises(ValueError, match="Mach"):
        LBMConfig(inflow=0.9)


def test_default_case_sheds_vortices():
    # Well past the shedding threshold of Re ~ 47, and below the Mach limit.
    config = LBMConfig()
    assert config.reynolds > 47
    assert config.mach < config.MAX_MACH


def test_obstacle_sits_a_quarter_downstream():
    config = LBMConfig(nx=400, ny=100)
    mask = build_obstacle(config)
    assert mask.shape == (100, 400)
    assert mask[50, 100]  # centre of the cylinder
    assert not mask[50, 300]  # well downstream
    # Area should be close to pi r^2 for a 13-cell radius.
    assert mask.sum() == pytest.approx(np.pi * 13**2, rel=0.05)


def test_working_set_estimate_scales_with_cells():
    small = working_set_mb(LBMConfig(nx=100, ny=100))
    big = working_set_mb(LBMConfig(nx=200, ny=200))
    assert big == pytest.approx(4 * small)


# --- solver ------------------------------------------------------------------


def test_initial_state_matches_requested_density():
    solver = LBMSolver(SMALL, get_backend("numpy"))
    rho, _, _ = solver.macroscopic()
    assert rho == pytest.approx(SMALL.rho0, rel=1e-9)


def test_mass_is_conserved():
    solver = LBMSolver(SMALL, get_backend("numpy"))
    before = solver.total_mass()
    solver.run(100)
    assert solver.total_mass() == pytest.approx(before, rel=1e-9)


def test_reference_case_stays_finite():
    solver = LBMSolver(LBMConfig(nx=400, ny=100), get_backend("numpy"))
    solver.run(500)
    assert solver.is_finite()
    _, ux, _ = solver.macroscopic()
    # A stable run stays near the free-stream speed; divergence shows up here
    # as a velocity far above the speed of sound long before it reaches inf.
    assert np.abs(ux).max() < 0.5


def test_populations_stay_positive():
    # The regression guard for the original bug: seeding the field away from
    # equilibrium made the first over-relaxed collision drive populations
    # negative, and the run blew up at the cylinder within ~60 steps.
    solver = LBMSolver(LBMConfig(nx=400, ny=100, dtype="float64"), get_backend("numpy"))
    for _ in range(100):
        solver.step()
        assert np.asarray(solver.f).min() > 0


def test_initial_state_is_exactly_at_equilibrium():
    # Equilibrium is a fixed point of the collision operator, so starting there
    # means the first step relaxes nothing and cannot overshoot.
    solver = LBMSolver(LBMConfig(nx=80, ny=40, dtype="float64"), get_backend("numpy"))
    f = np.asarray(solver.f)
    rho = f.sum(2)
    ux = (f * CXS).sum(2) / rho
    uy = (f * CYS).sum(2) / rho
    feq = equilibrium(rho, ux, uy, CXS.astype(float), CYS.astype(float), WEIGHTS)
    assert np.allclose(f, feq, rtol=1e-12)


def test_same_seed_reproduces_the_run():
    a = LBMSolver(SMALL, get_backend("numpy"))
    b = LBMSolver(SMALL, get_backend("numpy"))
    a.run(25)
    b.run(25)
    assert np.array_equal(a.f, b.f)


def test_different_seeds_diverge():
    a = LBMSolver(SMALL, get_backend("numpy"))
    b = LBMSolver(LBMConfig(nx=80, ny=40, dtype="float64", seed=7), get_backend("numpy"))
    a.run(25)
    b.run(25)
    assert not np.array_equal(a.f, b.f)


def test_flow_moves_downstream():
    solver = LBMSolver(SMALL, get_backend("numpy"))
    solver.run(50)
    _, ux, _ = solver.macroscopic()
    assert ux.mean() > 0


def test_obstacle_is_excluded_from_the_velocity_field():
    solver = LBMSolver(SMALL, get_backend("numpy"))
    solver.run(10)
    _, ux, uy = solver.macroscopic()
    mask = solver.obstacle_mask
    assert (ux[mask] == 0).all()
    assert (uy[mask] == 0).all()


def test_vorticity_masks_the_cylinder():
    solver = LBMSolver(SMALL, get_backend("numpy"))
    solver.run(10)
    vort = solver.vorticity()
    assert vort.shape == (SMALL.ny, SMALL.nx)
    assert np.isnan(vort[solver.obstacle_mask]).all()
    assert np.isfinite(vort[~solver.obstacle_mask]).all()


def test_float32_state_stays_float32():
    solver = LBMSolver(LBMConfig(nx=80, ny=40, dtype="float32"), get_backend("numpy"))
    solver.run(5)
    assert np.asarray(solver.f).dtype == np.float32


def test_step_count_tracks_the_run():
    solver = LBMSolver(SMALL, get_backend("numpy"))
    solver.run(12)
    solver.step()
    assert solver.steps_taken == 13


# --- backends ----------------------------------------------------------------


def test_numpy_is_always_available():
    assert "numpy" in available_backends()


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nonsense")


@pytest.mark.parametrize("name", OTHER_BACKENDS)
def test_backend_roundtrips_arrays(name):
    backend = get_backend(name)
    source = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    assert np.array_equal(backend.to_numpy(backend.asarray(source, np.float64)), source)


@pytest.mark.parametrize("name", OTHER_BACKENDS)
def test_backend_roll_matches_numpy(name):
    backend = get_backend(name)
    source = np.arange(20, dtype=np.float64).reshape(4, 5)
    rolled = backend.to_numpy(backend.roll(backend.asarray(source, np.float64), (1, -2), (0, 1)))
    assert np.array_equal(rolled, np.roll(source, (1, -2), axis=(0, 1)))


@pytest.mark.parametrize("name", OTHER_BACKENDS)
def test_backend_agrees_with_numpy(name):
    # float64 keeps the only differences at reduction-order rounding level.
    results = compare_backends([name], SMALL, steps=60, tolerance=1e-9)
    assert results[0].passed, f"{name} drifted by {results[0].max_rel_rho:.2e}"


# --- benchmark harness -------------------------------------------------------


def test_benchmark_reports_sane_numbers():
    config = LBMConfig(nx=80, ny=40)
    result = run_benchmark("numpy", config, steps=10, warmup=2, repeats=2)
    assert result.backend == "numpy"
    assert result.seconds > 0
    assert result.mlups == pytest.approx(config.cells * 10 / result.seconds / 1e6)
    assert result.step_ms == pytest.approx(result.seconds * 100)
    assert result.finite
    assert len(result.seconds_all) == 2
    assert result.seconds == min(result.seconds_all)


def test_benchmark_rejects_empty_runs():
    with pytest.raises(ValueError):
        run_benchmark("numpy", LBMConfig(nx=80, ny=40), steps=0)


def test_benchmark_serialises():
    result = run_benchmark("numpy", LBMConfig(nx=80, ny=40), steps=5, warmup=0, repeats=1)
    data = result.to_dict()
    assert {"backend", "mlups", "bandwidth_gbs", "step_ms", "nx", "ny"} <= data.keys()
