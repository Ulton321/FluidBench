"""Correctness checks for the D3Q19 solver.

Run from the FluidBench directory:

    python -m pytest tests -v

The 2D suite proves the backends agree with each other.  That is necessary
and not sufficient: three implementations can agree on the same wrong answer.
These tests add cases with a closed form, so the solver is checked against
the physics rather than against itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fluidbench.backends import available_backends, get_backend  # noqa: E402
from fluidbench.lbm3d import (  # noqa: E402
    C3,
    CS2,
    NL3,
    OPPOSITE3,
    W3,
    LBM3DConfig,
    LBM3DSolver,
    build_sphere,
    equilibrium3d,
    populations_from,
    taylor_green_velocity,
    working_set_mb,
)

SMALL = LBM3DConfig(nx=48, ny=24, nz=24, dtype="float64")

OTHER_BACKENDS = [name for name in available_backends() if name != "numpy"]


# --- lattice constants -------------------------------------------------------


def test_nineteen_directions_with_unit_weight():
    assert len(C3) == NL3 == len(W3) == 19
    assert W3.sum() == pytest.approx(1.0, abs=1e-14)


def test_weights_reproduce_the_second_moment():
    """sum_i w_i c_i c_i must be cs^2 I, or the equilibrium is not
    Navier-Stokes to second order."""
    moment = np.einsum("i,ia,ib->ab", W3, C3.astype(float), C3.astype(float))
    assert np.allclose(moment, np.eye(3) * CS2)


def test_odd_moments_vanish():
    """First and third moments must be zero -- the lattice has no preferred
    direction, and a non-zero third moment shows up as spurious anisotropy."""
    assert np.allclose(np.einsum("i,ia->a", W3, C3.astype(float)), 0)
    assert np.allclose(
        np.einsum("i,ia,ib,ic->abc", W3, *([C3.astype(float)] * 3)), 0
    )


def test_opposite_directions_reverse_velocity():
    assert np.array_equal(C3[OPPOSITE3], -C3)
    assert OPPOSITE3[0] == 0


def test_speeds_are_rest_axial_or_diagonal():
    speeds = np.abs(C3).sum(axis=1)
    assert sorted(np.bincount(speeds).tolist()) == [1, 6, 12]


# --- configuration -----------------------------------------------------------


def test_unstable_tau_is_rejected():
    with pytest.raises(ValueError, match="tau"):
        LBM3DConfig(tau=0.5)


def test_tiny_grid_is_rejected():
    with pytest.raises(ValueError, match="8x8x8"):
        LBM3DConfig(nx=4, ny=4, nz=4)


def test_supersonic_inflow_is_rejected():
    with pytest.raises(ValueError, match="Mach"):
        LBM3DConfig(inflow=0.9)


def test_viscosity_follows_tau():
    assert LBM3DConfig(tau=0.8).viscosity == pytest.approx(0.1)


def test_working_set_scales_with_cells():
    small = working_set_mb(LBM3DConfig(nx=32, ny=32, nz=32))
    big = working_set_mb(LBM3DConfig(nx=64, ny=32, nz=32))
    assert big == pytest.approx(2 * small, rel=1e-6)


# --- geometry ----------------------------------------------------------------


def test_sphere_volume_matches_the_analytic_one():
    config = LBM3DConfig(nx=96, ny=48, nz=48)
    mask = build_sphere(config)
    assert mask.sum() == pytest.approx(
        4 / 3 * np.pi * config.sphere_radius**3, rel=0.05
    )


def test_sphere_sits_a_quarter_downstream_and_centred():
    config = LBM3DConfig(nx=96, ny=48, nz=48)
    z, y, x = np.nonzero(build_sphere(config))
    assert x.mean() == pytest.approx(config.nx // 4, abs=1.0)
    assert y.mean() == pytest.approx(config.ny // 2, abs=1.0)
    assert z.mean() == pytest.approx(config.nz // 2, abs=1.0)


# --- the physics -------------------------------------------------------------


def test_initial_state_is_exactly_at_equilibrium():
    """Seeding away from equilibrium is what makes the tutorial version blow
    up; the 3D solver inherits the fix and this guards it.

    Checked without the sphere, because `macroscopic()` deliberately zeroes
    the velocity inside the obstacle -- rebuilding the equilibrium from a
    masked field would not reproduce what is actually stored there.
    """
    config = LBM3DConfig(nx=48, ny=24, nz=24, dtype="float64", obstacle=False)
    solver = LBM3DSolver(config, get_backend("numpy"))
    rho, ux, uy, uz = solver.macroscopic()
    for i in range(NL3):
        assert np.allclose(
            solver.backend.to_numpy(solver.f[i]),
            equilibrium3d(rho, ux, uy, uz, i),
            atol=1e-12,
        )


def test_mass_is_conserved():
    solver = LBM3DSolver(SMALL, get_backend("numpy"))
    before = solver.total_mass()
    solver.run(40)
    assert solver.total_mass() == pytest.approx(before, rel=1e-12)


def test_reference_case_stays_finite():
    solver = LBM3DSolver(SMALL, get_backend("numpy"))
    solver.run(60)
    assert solver.is_finite()


def test_populations_stay_positive():
    solver = LBM3DSolver(SMALL, get_backend("numpy"))
    solver.run(60)
    for i in range(NL3):
        assert solver.backend.to_numpy(solver.f[i]).min() > 0


def test_flow_moves_downstream():
    solver = LBM3DSolver(SMALL, get_backend("numpy"))
    solver.run(20)
    _, ux, _, _ = solver.macroscopic()
    assert ux.mean() > 0


def test_sphere_is_excluded_from_the_velocity_field():
    solver = LBM3DSolver(SMALL, get_backend("numpy"))
    solver.run(10)
    _, ux, uy, uz = solver.macroscopic()
    mask = solver.obstacle_mask
    assert np.all(ux[mask] == 0) and np.all(uy[mask] == 0) and np.all(uz[mask] == 0)


def test_vorticity_masks_the_sphere():
    solver = LBM3DSolver(SMALL, get_backend("numpy"))
    solver.run(10)
    mag = solver.vorticity_magnitude()
    assert np.all(np.isnan(mag[solver.obstacle_mask]))
    assert np.isfinite(mag[~solver.obstacle_mask]).all()


def test_same_seed_reproduces_the_run():
    a = LBM3DSolver(SMALL, get_backend("numpy")); a.run(15)
    b = LBM3DSolver(SMALL, get_backend("numpy")); b.run(15)
    assert np.array_equal(a.macroscopic()[0], b.macroscopic()[0])


# --- against a closed form ---------------------------------------------------


TGV = LBM3DConfig(nx=64, ny=64, nz=16, tau=0.8, inflow=0.04,
                  dtype="float64", obstacle=False)


def _taylor_green_solver(backend_name="numpy"):
    ux, uy, uz, rho, decay = taylor_green_velocity(TGV)
    f0 = populations_from(rho, ux, uy, uz, np.dtype(TGV.dtype))
    return LBM3DSolver(TGV, get_backend(backend_name), populations=f0), ux, decay


def test_taylor_green_decays_at_the_analytic_rate():
    """The vortex decays as exp(-2 nu k^2 t).  This is the test that would
    catch a wrong relaxation time or a mis-scaled equilibrium -- neither of
    which cross-backend agreement can see, since every backend would be
    wrong together."""
    solver, ux0, decay = _taylor_green_solver()
    peak0 = np.abs(ux0).max()
    for step in (100, 200, 400):
        solver.run(step - solver.steps_taken)
        _, ux, _, _ = solver.macroscopic()
        expected = peak0 * np.exp(-decay * solver.steps_taken)
        assert np.abs(ux).max() == pytest.approx(expected, rel=5e-3)


def test_taylor_green_stays_uniform_along_z():
    """The initial condition has no z dependence and the equations introduce
    none, so any drift along z is a bug in the third axis of the streaming --
    the one thing 2D could never have tested."""
    solver, _, _ = _taylor_green_solver()
    solver.run(200)
    _, ux, uy, uz = solver.macroscopic()
    assert np.abs(ux - ux[0]).max() < 1e-14
    assert np.abs(uy - uy[0]).max() < 1e-14
    assert np.abs(uz).max() < 1e-14


def test_taylor_green_conserves_mass():
    solver, _, _ = _taylor_green_solver()
    before = solver.total_mass()
    solver.run(200)
    assert solver.total_mass() == pytest.approx(before, rel=1e-12)


# --- backends ----------------------------------------------------------------


@pytest.mark.parametrize("name", OTHER_BACKENDS)
def test_backends_track_numpy(name):
    """Every backend runs the same sequence of operations on the same bytes.

    Unlike the 2D solver this comes out bit-identical: the per-direction loop
    sums the moments in a fixed order, where `f.sum(2)` lets each library
    pick its own reduction tree.
    """
    reference = LBM3DSolver(SMALL, get_backend("numpy"))
    reference.run(30)
    rho0, ux0, uy0, uz0 = reference.macroscopic()

    other = LBM3DSolver(SMALL, get_backend(name))
    other.run(30)
    rho, ux, uy, uz = other.macroscopic()

    scale = np.abs(rho0).max()
    assert np.abs(rho - rho0).max() / scale < 1e-9
    assert np.abs(ux - ux0).max() < 1e-9
    assert np.abs(uy - uy0).max() < 1e-9
    assert np.abs(uz - uz0).max() < 1e-9


@pytest.mark.parametrize("name", OTHER_BACKENDS)
def test_backends_agree_on_taylor_green(name):
    solver, ux0, decay = _taylor_green_solver(name)
    peak0 = np.abs(ux0).max()
    solver.run(200)
    _, ux, _, _ = solver.macroscopic()
    expected = peak0 * np.exp(-decay * solver.steps_taken)
    assert np.abs(ux).max() == pytest.approx(expected, rel=5e-3)
