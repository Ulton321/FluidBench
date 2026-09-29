# D3Q19: scoping a 3D solver, and an F1 case on top of it

Status: **phases 1 and 2 built and passing** (`fluidbench/lbm3d.py`,
`tests/test_lbm3d.py`, 27 tests). Phases 3-5 not started. Numbers are measured
on this machine (RTX 2070 8 GB, torch 2.7.1+cu118, float32), not estimated.

| phase | state |
|---|---|
| 1 D3Q19 solver, SoA, per-direction step | **done** |
| 2 validation against closed forms | **done** -- Taylor-Green, lattice moments, mass, sphere volume |
| 3 inlet/outlet, moving ground, forces, sphere Cd | not started |
| 4 fused CUDA kernel | not started |
| 5 car geometry, Smagorinsky, Q-criterion rendering | not started |

There is still **no car and no 3D view in the dashboard.** The solver exists and
is checked; nothing renders it yet.

## Why 3D at all

Two reasons, and the second is the better one.

1. **F1 aerodynamics is three-dimensional or it is nothing.** Almost everything
   that matters in that sport is a streamwise vortex: the Y250 off the front
   wing, tip vortices, the floor-edge and diffuser vortices that do most of the
   downforce work, the wheel wake. A 2D slice contains none of them. A 2D car
   silhouette would give a believable picture of stagnation and separation, and
   a completely fictional drag number.

2. **The benchmark gets more interesting.** In 2D the GPU saturates near 90
   MLUPS by 2.6 M cells and the speedup stops growing at ~32x, because both
   sides end up bandwidth bound. A 3D domain is tens of millions of cells with
   19 populations each.

   **Measured correction.** This claim was half wrong, and the measurement is
   worth keeping. 3D does *not* give a bigger ratio: it peaks at **33.5x** at
   1.8 M cells, against 2D's 32.3x. The launch-bound region does not disappear
   either -- at 0.07 M cells the GPU still loses (0.8x), the same story as 2D.
   The reason is that the per-direction step is launch-overhead dominated on
   both sides: 66 MLUPS is 2% of what the card's bandwidth allows. The extra
   headroom 3D promises is real but it is behind the fused kernel (phase 4),
   not in front of it. Until then, 3D's argument for itself is the physics,
   not the speedup.

## Feasibility: what this card can actually do

A throwaway D3Q19 probe, same arithmetic as `lbm.py`, three implementations:

| layout | step shape | MLUPS | bytes/cell | max grid on ~7 GiB |
|---|---|---:|---:|---:|
| AoS `f[z,y,x,i]` | equilibrium materialised (what `lbm.py` does) | 49 | 554 | 13 M |
| AoS `f[z,y,x,i]` | per-direction loop, planar temporaries | 24 | 196 | 38 M |
| **SoA `f[i,z,y,x]`** | **per-direction loop, planar temporaries** | **66** | **196** | **38 M** |

Three things follow.

**The equilibrium cannot be materialised.** Building `feq` as a full
`(nz, ny, nx, 19)` array costs 554 bytes per cell and caps the domain at 13 M
cells, which is too coarse for a car. Looping over the 19 directions and keeping
every temporary planar drops that to 196 B/cell and nearly triples the domain.
This is a real change to how a step is written, and it is not optional.

**The layout has to flip.** `f[..., i]` on an array-of-structures reads one
float every 76 bytes, so coalescing is gone. Moving the direction to the *outer*
axis makes each direction a contiguous block and is worth **2.8x** at identical
memory. That is the single largest free win available.

**The array-library approach leaves 98% of the card on the table.** 66 MLUPS
against the minimum-traffic model (2 x 19 x 4 = 152 B/cell/step) is about
10 GB/s on a card that does 448 GB/s. The step is ~150 separate full-array
passes when it could be one. A fused CUDA kernel is where the remaining 20-40x
lives.

### Time to solution, which is the uncomfortable part

At 38 M cells and 66 MLUPS: **1.7 steps per second**. A car case wants on the
order of 10^5 steps to develop and then time-average, so:

- 100k steps at 38 M cells: **~16 hours**
- the same with a fused kernel at a plausible 1.5 GLUPS: **~45 minutes**

So for 3D the custom CUDA kernel stops being "the natural next backend" and
becomes load-bearing. That reframes the project: the kernel is no longer a neat
demonstration of the upper bound, it is the difference between a run you can
iterate on and a run you start before going to bed.

## What has to be built

### 1. The solver (`lbm3d.py`)

D3Q19: rest + 6 axial (w = 1/18) + 12 face-diagonal (w = 1/36), `cs^2 = 1/3`.
The lattice identity `sum_i w_i c_i c_i = I/3` is checked in the probe and
holds. Opposites are adjacent pairs, so bounce-back indexing stays trivial.

State is `(19, nz, ny, nx)`, ping-ponged between two buffers. The backend
interface (`asarray`, `to_numpy`, `roll`, `synchronize`) needs nothing new --
`roll` just takes three axes -- so numpy, CuPy and torch all still run the same
code, and `validate` works unchanged.

### 2. Boundary conditions (the part 2D let us skip)

The current solver is periodic in both directions, which is fine for a cylinder
in a box and useless for a car: the wake wraps around into the nose.

- **Inlet**: fixed velocity (Zou/He, or equilibrium plus non-equilibrium
  extrapolation).
- **Outlet**: convective or zero-gradient. Getting this wrong reflects the wake
  back upstream and quietly poisons everything upstream of it.
- **Moving ground**: the single most F1-specific piece. In the car's frame the
  road moves at freestream. A stationary floor grows a boundary layer that does
  not exist and gets ground effect badly wrong. Momentum-offset bounce-back:
  `f_opp = f_i - 6 w_i rho_w (c_i . u_wall)`. About five lines.
- **Rotating wheels**: the same trick with a tangential wall velocity.

### 3. Forces

Otherwise it is a picture, not a simulator. Momentum exchange over the boundary
links gives drag and lift directly, and the populations needed are already
captured in the step as the bounced set. Sum `c_i (f_i + f_opp)` over links that
cross the interface. Roughly 15 lines, backend-agnostic. Report `Cd`, `Cl`, and
their running time-average.

### 4. Turbulence, or an honest admission

Real F1 Reynolds number is ~5x10^6. BGK at the viscosity that implies is
unconditionally unstable, and no grid we can afford resolves it. Options in
increasing order of work:

- Cap Re around 10^4 and label the result a laminar analogue. Honest, cheap, and
  genuinely instructive about separation and wake structure. Not F1.
- **Smagorinsky** subgrid viscosity from the non-equilibrium part of `f`
  (~10 lines) -- buys an order of magnitude of stability, still not LES quality.
- **MRT or regularized collision** -- markedly more stable at low viscosity than
  BGK, and a bigger change to the step.

Whatever is chosen goes on the page next to the number, the way the WebGL flow
is labelled today.

### 5. Seeing it

3D fields do not render themselves, and this is easy to underestimate.

- **Slices** through the centreline and a few streamwise planes -- cheapest, and
  the most diagnostic while developing.
- **Q-criterion or lambda-2 iso-surfaces** -- the standard way to show vortex
  cores, and the only view in which the F1 structures are recognisable. Needs
  marching cubes.
- **Volume ray-marching** of vorticity magnitude in WebGL2 -- a 3D texture and
  ~150 lines of shader. This is the one that looks spectacular.

Important: at 38 M cells the simulation cannot run in the browser. The web view
would ray-march a *saved, downsampled* field from a Python run rather than
simulate live. That is a different data path from the current Flow tab and
should be designed as one.

## Validation: how we would know it is right

The existing `validate` (cross-backend drift) still applies, but it only proves
the backends agree, not that any of them is correct. For 3D:

- **Taylor-Green vortex** -- analytic decay at early times. The first test.
- **Poiseuille flow in a duct** -- analytic parabolic profile; checks the wall
  treatment specifically.
- **3D lid-driven cavity** at Re 100/400/1000 -- published reference profiles.
- **Flow past a sphere** -- the 3D analogue of the current cylinder, with
  well-established `Cd(Re)` correlations. This is the bridge case: it exercises
  the force measurement on a shape whose answer is already known, before any car
  is involved.
- Mass conservation and positivity, as in the 2D tests.

## Phasing

| phase | deliverable | why it is separable |
|---|---|---|
| 1 | D3Q19 solver, SoA, per-direction step, plus `bench` / `sweep` in 3D | Immediately improves the benchmark story. No geometry, no BCs. |
| 2 | Taylor-Green, Poiseuille and cavity validation | Nothing downstream is trustworthy without it. |
| 3 | Inlet/outlet, moving ground, forces; sphere `Cd` against published data | The physics that makes a car case meaningful, tested on a shape with a known answer. |
| 4 | Fused CUDA kernel backend | By here a reference implementation exists to check it against. |
| 5 | Car geometry, Smagorinsky, Q-criterion rendering | Only now is this worth doing. |

Phase 1 alone is worth having. Phase 5 without phases 2 and 3 would produce a
convincing-looking picture with no claim to correctness, which is the failure
mode this project exists to argue against.

## Open decisions

1. **Does the 2D benchmark change?** The layout finding applies to it too, but
   the answer is size-dependent: SoA is 0.32x at 40 k cells and 1.42x at 2.6 M,
   because the per-direction loop issues many more kernels and small grids are
   launch-bound -- the project's own thesis showing up in its own
   implementation. Switching would invalidate the published results table.
   Recommendation: leave 2D alone and report the layout comparison as a finding.
2. **Which collision operator**, given that BGK caps the achievable Reynolds
   number hard.
3. **How much of the F1 claim to make.** Recommendation: call it "flow past a
   car-shaped body", and let the pictures speak rather than quote a drag
   coefficient anyone might take seriously.

## What phase 1 actually produced

`fluidbench/lbm3d.py`. State is `(19, nz, ny, nx)`, ping-ponged, BGK collision
isolated in one loop so MRT can replace it without touching anything else. The
2D solver is untouched, so the published results table still stands.

Measured on this machine, float32, sphere case:

```
grid              cells     MiB    numpy   torch-cuda   speedup
64x32x32          0.07 M     11      5.2          4.4      0.8x   <- GPU loses
128x64x64         0.52 M     90      2.0         35.8     17.9x
192x96x96         1.77 M    304      1.9         64.2     33.5x
256x128x128       4.19 M    720      2.4         66.7     27.8x
```

Validation, all in `tests/test_lbm3d.py`:

- Lattice moments: weights sum to 1, first and third moments vanish, second
  moment is exactly `cs^2 I`, opposites reverse. All exact.
- **Taylor-Green vortex** against its closed form: tracks the analytic decay
  `exp(-2 nu k^2 t)` to **1e-3 relative** over 400 steps, which is the expected
  second-order error at this resolution. This is the test that would catch a
  wrong relaxation time or a mis-scaled equilibrium -- cross-backend agreement
  cannot, because every backend would be wrong together.
- **z-uniformity**: the Taylor-Green field has no z dependence and must not
  develop one. Drift after 200 steps is `0.0` exactly, and `max|uz|` is 7e-17.
  This is the only real check that the third axis of the streaming is right,
  and 2D could never have run it.
- Mass conserved to 3e-14 over 400 steps; populations stay strictly positive.
- Sphere volume within 0.3% of `4/3 pi r^3`.

One unexpected result: **cross-backend agreement is bit-identical** (numpy vs
torch-cpu vs torch-cuda, drift exactly 0.0), where the 2D solver drifts at
~3e-13. The per-direction loop sums the moments in a fixed order on every
backend, while `f.sum(2)` lets each library choose its own reduction tree.
Worth knowing: it means `validate` in 3D can use a far tighter tolerance.

## Measured appendix

```
D3Q19 float32, structure-of-arrays, per-direction step, RTX 2070:
  64x64x128     0.5 M cells    28.3 MLUPS   196 B/cell
  96x96x192     1.8 M cells    61.6 MLUPS   196 B/cell
  128x128x256   4.2 M cells    60.8 MLUPS   196 B/cell
  160x160x320   8.2 M cells    65.8 MLUPS   197 B/cell
  192x192x384  14.2 M cells    66.1 MLUPS   196 B/cell

D3Q19 float32, same step, array-of-structures:
  128x128x256   4.2 M cells    23.8 MLUPS   198 B/cell
D3Q19 float32, equilibrium materialised (the shape lbm.py uses today):
  128x128x256   4.2 M cells    49.6 MLUPS   554 B/cell

D2Q9 float32, current layout vs SoA, RTX 2070:
  400x100     40 k     AoS  25.0   SoA    8.0   SoA 0.32x
  800x200    160 k     AoS  95.9   SoA   31.5   SoA 0.33x
  1600x400   640 k     AoS 108.0   SoA  133.6   SoA 1.24x
  3200x800   2.56 M    AoS 110.4   SoA  156.6   SoA 1.42x
```
