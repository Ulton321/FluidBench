"""Rendering the flow, so you can see that the benchmark simulates something real.

Nothing here is timed.  Pulling a frame back to the host is a device-to-host
copy, which would dominate the measurement at small grid sizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends import Backend
from .lbm import LBMConfig, LBMSolver

__all__ = ["render_vorticity", "animate"]


def _symmetric_limit(field: np.ndarray, percentile: float = 99.5) -> float:
    """A colour limit that ignores the few extreme cells near the cylinder."""
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    return limit if limit > 0 else 1.0


def render_vorticity(
    solver: LBMSolver,
    path: str | Path,
    limit: float | None = None,
    cmap: str = "RdBu_r",
    dpi: int = 120,
) -> Path:
    """Write a single vorticity frame to an image file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vort = solver.vorticity()
    limit = limit or _symmetric_limit(vort)

    fig, ax = plt.subplots(figsize=(10, 10 * solver.config.ny / solver.config.nx))
    _draw(ax, vort, limit, cmap, solver)
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw(ax, vort: np.ndarray, limit: float, cmap: str, solver: LBMSolver):
    import matplotlib.pyplot as plt

    colours = plt.get_cmap(cmap).copy()
    colours.set_bad("0.35")  # the cylinder, which carries NaN
    image = ax.imshow(vort, cmap=colours, vmin=-limit, vmax=limit, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"vorticity -- {solver.config.label()}, {solver.backend.name}, "
        f"step {solver.steps_taken}"
    )
    return image


def animate(
    config: LBMConfig,
    backend: Backend,
    steps: int = 3000,
    every: int = 25,
    save: str | Path | None = None,
    fps: int = 20,
    limit: float | None = None,
    cmap: str = "RdBu_r",
    dpi: int = 100,
    progress=None,
    spinup: int = 0,
) -> LBMSolver:
    """Run the simulation, drawing a frame every `every` steps.

    With `save` the frames go to a .gif or .mp4 instead of a window.

    `spinup` steps run before any drawing starts.  The wake needs a few
    thousand steps before it rolls up, and watching a near-uniform field
    inch along for the first minute looks exactly like a program that has
    hung, so by default the boring part is fast-forwarded.

    `progress` is called with (step, total, phase) so the caller can show that
    something is happening.
    """
    import matplotlib

    if save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    solver = LBMSolver(config, backend)

    for done in range(0, spinup, every):
        solver.run(min(every, spinup - done))
        if progress is not None:
            progress(solver.steps_taken, spinup, "spin-up")

    fig, ax = plt.subplots(figsize=(10, 10 * config.ny / config.nx))
    vort = solver.vorticity()

    # The colour scale cannot be fixed from a frame taken before the flow has
    # developed: at step 0 the field is only the initial jitter, roughly seven
    # times weaker than the developed wake, so holding that scale saturates
    # every later frame into solid red and blue.  After a spin-up the very
    # first frame is already representative; without one, keep tracking the
    # scale until the wake has grown, then freeze it so it does not flicker.
    fixed_limit = limit is not None
    limit = limit if fixed_limit else _symmetric_limit(vort)
    if fixed_limit or spinup >= 500:
        calibrate_until = solver.steps_taken
    else:
        calibrate_until = solver.steps_taken + max(every, min(500, steps // 10))

    image = _draw(ax, vort, limit, cmap, solver)
    fig.tight_layout()

    writer = None
    if save is not None:
        save = Path(save)
        suffix = save.suffix.lower()
        if suffix == ".gif":
            writer = PillowWriter(fps=fps)
        elif suffix in (".mp4", ".mov", ".avi"):
            writer = FFMpegWriter(fps=fps)
        else:
            raise ValueError(f"unsupported animation format {suffix!r}; use .gif or .mp4")

    def frame() -> None:
        field = solver.vorticity()
        image.set_data(field)
        if solver.steps_taken <= calibrate_until:
            scale = _symmetric_limit(field)
            image.set_clim(-scale, scale)
        ax.set_title(
            f"vorticity -- {config.label()}, {backend.name}, step {solver.steps_taken}"
        )
        if writer is not None:
            writer.grab_frame()
        else:
            # draw_idle + flush_events rather than plt.pause(): a short pause
            # leaves the redraw queued and the window can go several frames
            # without actually repainting, which looks like a frozen picture.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def loop() -> None:
        target = solver.steps_taken + steps
        while solver.steps_taken < target:
            solver.run(min(every, target - solver.steps_taken))
            if not solver.is_finite():
                raise RuntimeError(
                    f"simulation diverged at step {solver.steps_taken}; "
                    "lower the inflow velocity or raise tau"
                )
            frame()
            if progress is not None:
                progress(solver.steps_taken - (target - steps), steps, "render")

    if writer is not None:
        with writer.saving(fig, str(save), dpi):
            writer.grab_frame()
            loop()
        plt.close(fig)
    else:
        plt.show(block=False)
        fig.canvas.draw()
        loop()
        plt.show()  # keep the finished flow on screen until it is closed

    return solver
