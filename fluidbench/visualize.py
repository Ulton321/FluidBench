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
) -> LBMSolver:
    """Run the simulation, drawing a frame every `every` steps.

    With `save` the frames go to a .gif or .mp4 instead of a window.
    """
    import matplotlib

    if save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    solver = LBMSolver(config, backend)

    fig, ax = plt.subplots(figsize=(10, 10 * config.ny / config.nx))
    vort = solver.vorticity()
    limit = limit or _symmetric_limit(vort)
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
        ax.set_title(
            f"vorticity -- {config.label()}, {backend.name}, step {solver.steps_taken}"
        )
        if writer is not None:
            writer.grab_frame()
        else:
            plt.pause(0.001)

    def loop() -> None:
        for _ in range(0, steps, every):
            solver.run(every)
            if not solver.is_finite():
                raise RuntimeError(
                    f"simulation diverged at step {solver.steps_taken}; "
                    "lower the inflow velocity or raise tau"
                )
            frame()

    if writer is not None:
        with writer.saving(fig, str(save), dpi):
            writer.grab_frame()
            loop()
        plt.close(fig)
    else:
        plt.show(block=False)
        loop()
        plt.show()

    return solver
