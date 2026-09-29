"""Command line front end.

    python main.py list                 what this machine can run
    python main.py bench                time every backend at one grid size
    python main.py sweep                find the size where the GPU takes over
    python main.py validate             check the backends agree
    python main.py sim --save out.gif   render the vortex street
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import __version__
from .backends import (
    TorchBackend,
    all_backend_names,
    available_backends,
    get_backend,
    resolve_names,
)
from .benchmark import (
    BenchmarkResult,
    compare_backends,
    run_benchmark,
    sweep,
    working_set_mb,
)
from .lbm import LBMConfig

#: Grid sizes for `sweep`, spanning launch-latency-bound to bandwidth-bound.
DEFAULT_SWEEP = [(200, 50), (400, 100), (800, 200), (1600, 400), (3200, 800)]

#: Above this the sweep warns rather than silently filling up VRAM.
MEMORY_WARN_MB = 3072


def _parse_size(text: str) -> tuple[int, int]:
    """Parse `NXxNY`, e.g. 400x100."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected NXxNY, got {text!r}")
    try:
        nx, ny = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integers in {text!r}") from None
    if nx < 8 or ny < 8:
        raise argparse.ArgumentTypeError("grid must be at least 8x8")
    return nx, ny


def _parse_sizes(text: str) -> list[tuple[int, int]]:
    return [_parse_size(token) for token in text.split(",") if token.strip()]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Fixed-width table; first column left aligned, the rest right."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        out = [cells[0].ljust(widths[0])]
        out += [cell.rjust(widths[i + 1]) for i, cell in enumerate(cells[1:])]
        return "  ".join(out).rstrip()

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt(row) for row in rows]
    return "\n".join(lines)


def _config_from_args(args) -> LBMConfig:
    nx, ny = args.size
    return LBMConfig(nx=nx, ny=ny, tau=args.tau, dtype=args.dtype, seed=args.seed)


def _build(name: str, threads: int | None):
    backend = get_backend(name)
    if threads and isinstance(backend, TorchBackend) and not backend.is_gpu:
        backend.set_num_threads(threads)
    return backend


def _baseline(results: list[BenchmarkResult]) -> BenchmarkResult | None:
    """The CPU result speedups are quoted against."""
    for result in results:
        if result.backend == "numpy":
            return result
    return results[0] if results else None


def _result_rows(results: list[BenchmarkResult]) -> list[list[str]]:
    base = _baseline(results)
    rows = []
    for r in results:
        speedup = f"{r.mlups / base.mlups:.2f}x" if base else "-"
        rows.append(
            [
                r.backend,
                f"{r.nx}x{r.ny}",
                r.dtype,
                str(r.steps),
                f"{r.seconds:.4f}",
                f"{r.step_ms:.3f}",
                f"{r.mlups:.1f}",
                f"{r.bandwidth_gbs:.1f}",
                speedup,
                "" if r.finite else "diverged",
            ]
        )
    return rows


RESULT_HEADERS = [
    "backend",
    "grid",
    "dtype",
    "steps",
    "best(s)",
    "ms/step",
    "MLUPS",
    "GB/s",
    "speedup",
    "note",
]


def _export(results: list[BenchmarkResult], json_path, csv_path) -> None:
    payload = [r.to_dict() for r in results]
    if json_path:
        Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {json_path}")
    if csv_path and payload:
        fields = [k for k in payload[0] if k != "seconds_all"]
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(payload)
        print(f"wrote {csv_path}")


def cmd_list(args) -> int:
    available = available_backends()
    rows = []
    for name in all_backend_names():
        if name in available:
            try:
                rows.append([name, "yes", _build(name, args.threads).describe()])
            except Exception as exc:  # pragma: no cover - driver problems
                rows.append([name, "error", str(exc)])
        else:
            rows.append([name, "no", "not installed / no device"])
    print(_table(["backend", "ready", "device"], rows))
    if not any(get_backend(n).is_gpu for n in available if n != "numpy" and n != "torch-cpu"):
        print("\nNo GPU backend found. Install cupy or a CUDA build of torch to compare.")
    return 0


def cmd_bench(args) -> int:
    config = _config_from_args(args)
    names = resolve_names(args.backends)
    if not names:
        print("no backends selected", file=sys.stderr)
        return 1

    print(
        f"grid {config.label()}  tau {config.tau}  nu {config.viscosity:.4f}  "
        f"{config.dtype}  ~{working_set_mb(config):.0f} MiB/backend"
    )
    print(f"{args.steps} timed steps, {args.warmup} warmup, best of {args.repeats}\n")

    results = []
    for name in names:
        backend = _build(name, args.threads)
        print(f"  running {name:<12} {backend.describe()}", flush=True)
        results.append(
            run_benchmark(backend, config, args.steps, args.warmup, args.repeats)
        )

    print()
    print(_table(RESULT_HEADERS, _result_rows(results)))

    noisy = [r for r in results if r.spread > 1.25]
    for r in noisy:
        print(f"\nnote: {r.backend} timings varied by {r.spread:.2f}x across repeats")
    if any(not r.finite for r in results):
        print("\nnote: a run diverged -- timings stand, but the physics does not")

    _export(results, args.json, args.csv)
    return 0


def cmd_sweep(args) -> int:
    names = resolve_names(args.backends)
    if not names:
        print("no backends selected", file=sys.stderr)
        return 1

    base = LBMConfig(tau=args.tau, dtype=args.dtype, seed=args.seed)
    biggest = LBMConfig(
        nx=args.sizes[-1][0], ny=args.sizes[-1][1], tau=args.tau, dtype=args.dtype
    )
    needed = working_set_mb(biggest)
    if needed > MEMORY_WARN_MB:
        print(f"warning: the largest grid needs roughly {needed:.0f} MiB per backend\n")

    print(f"sweeping {', '.join(names)} over {len(args.sizes)} grid sizes")
    print(f"{args.steps} timed steps, {args.warmup} warmup, best of {args.repeats}\n")

    collected: list[BenchmarkResult] = []

    def report(result: BenchmarkResult) -> None:
        collected.append(result)
        print(
            f"  {result.backend:<12} {result.nx}x{result.ny:<6} "
            f"{result.mlups:8.1f} MLUPS  {result.step_ms:7.3f} ms/step",
            flush=True,
        )

    results = sweep(
        names, args.sizes, base, args.steps, args.warmup, args.repeats, on_result=report
    )

    print()
    for nx, ny in args.sizes:
        group = [r for r in results if (r.nx, r.ny) == (nx, ny)]
        print(_table(RESULT_HEADERS, _result_rows(group)))
        print()

    _crossover(results, args.sizes)
    _export(results, args.json, args.csv)
    return 0


def _crossover(results: list[BenchmarkResult], sizes: list[tuple[int, int]]) -> None:
    """Report the smallest grid at which each GPU backend beats the CPU."""
    cpu_name = "numpy"
    gpu_names = sorted({r.backend for r in results if r.backend not in (cpu_name,)})
    lines = []
    for name in gpu_names:
        crossed = None
        best = 0.0
        for nx, ny in sizes:
            cpu = next(
                (r for r in results if r.backend == cpu_name and (r.nx, r.ny) == (nx, ny)),
                None,
            )
            other = next(
                (r for r in results if r.backend == name and (r.nx, r.ny) == (nx, ny)), None
            )
            if cpu is None or other is None:
                continue
            ratio = other.mlups / cpu.mlups
            best = max(best, ratio)
            if crossed is None and ratio > 1.0:
                crossed = (nx, ny)
        if best:
            where = f"from {crossed[0]}x{crossed[1]}" if crossed else "never in this range"
            lines.append([name, f"{best:.2f}x", where])
    if lines:
        print(_table(["backend", "best speedup", "faster than numpy"], lines))
        print()


def cmd_validate(args) -> int:
    config = _config_from_args(args)
    names = resolve_names(args.backends)
    others = [n for n in names if n != args.reference]
    if not others:
        print(f"nothing to compare against {args.reference}")
        return 0

    print(
        f"comparing {', '.join(others)} against {args.reference} after "
        f"{args.steps} steps at {config.label()} in {config.dtype}\n"
    )
    results = compare_backends(names, config, args.steps, args.reference, args.tolerance)

    rows = [
        [
            r.backend,
            f"{r.max_abs_rho:.3e}",
            f"{r.max_rel_rho:.3e}",
            f"{r.max_abs_u:.3e}",
            f"{r.tolerance:.0e}",
            "pass" if r.passed else "FAIL",
        ]
        for r in results
    ]
    print(_table(["backend", "max |drho|", "max rel drho", "max |du|", "tol", ""], rows))

    if config.dtype == "float32":
        print(
            "\nnote: float32 reductions differ by backend; run with --dtype float64 "
            "for a tight comparison"
        )
    failed = [r for r in results if not r.passed]
    return 1 if failed else 0


def cmd_sim(args) -> int:
    from .visualize import animate

    config = _config_from_args(args)
    backend = _build(resolve_names([args.backend])[0], args.threads)
    print(f"simulating {config.label()} on {backend.describe()}")
    if args.save:
        print(f"writing {args.save} -- this renders every {args.every} steps")

    solver = animate(
        config,
        backend,
        steps=args.steps,
        every=args.every,
        save=args.save,
        fps=args.fps,
        limit=args.limit,
    )
    print(f"done after {solver.steps_taken} steps")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluidbench",
        description="CPU vs GPU benchmark built on a D2Q9 lattice Boltzmann solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"fluidbench {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, *, with_size=True):
        sub.add_argument(
            "--tau",
            type=float,
            default=0.53,
            help="relaxation time; must exceed 0.5 (default: 0.53)",
        )
        sub.add_argument(
            "--dtype",
            choices=["float32", "float64"],
            default="float32",
            help="float32 is the fair default -- consumer GPUs are far slower in float64",
        )
        sub.add_argument("--seed", type=int, default=42, help="initial-condition seed")
        sub.add_argument(
            "--threads", type=int, default=None, help="CPU threads for the torch backend"
        )
        if with_size:
            sub.add_argument(
                "--size",
                type=_parse_size,
                default=(400, 100),
                metavar="NXxNY",
                help="grid size (default: 400x100)",
            )

    def add_timing(sub, steps: int, warmup: int):
        sub.add_argument("--steps", type=int, default=steps, help="timed steps")
        sub.add_argument("--warmup", type=int, default=warmup, help="untimed warmup steps")
        sub.add_argument("--repeats", type=int, default=3, help="timed repeats; best wins")
        sub.add_argument("--json", help="write results to a JSON file")
        sub.add_argument("--csv", help="write results to a CSV file")

    p_list = subs.add_parser("list", help="show which backends are available")
    p_list.add_argument("--threads", type=int, default=None, help=argparse.SUPPRESS)
    p_list.set_defaults(func=cmd_list)

    p_bench = subs.add_parser("bench", help="benchmark one grid size")
    p_bench.add_argument(
        "--backends", nargs="+", default=["all"], help="backend names, or 'all'"
    )
    add_common(p_bench)
    add_timing(p_bench, steps=200, warmup=20)
    p_bench.set_defaults(func=cmd_bench)

    p_sweep = subs.add_parser("sweep", help="benchmark across grid sizes")
    p_sweep.add_argument("--backends", nargs="+", default=["all"], help="backends, or 'all'")
    p_sweep.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SWEEP,
        metavar="NXxNY,...",
        help="comma separated grid sizes",
    )
    add_common(p_sweep, with_size=False)
    add_timing(p_sweep, steps=100, warmup=10)
    p_sweep.set_defaults(func=cmd_sweep)

    p_val = subs.add_parser("validate", help="check the backends agree")
    p_val.add_argument("--backends", nargs="+", default=["all"], help="backends, or 'all'")
    p_val.add_argument("--reference", default="numpy", help="backend to compare against")
    p_val.add_argument("--steps", type=int, default=100, help="steps before comparing")
    p_val.add_argument("--tolerance", type=float, default=1e-6, help="max relative error")
    add_common(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_sim = subs.add_parser("sim", help="run and visualise the flow")
    p_sim.add_argument("--backend", default="numpy", help="backend to simulate on")
    p_sim.add_argument("--steps", type=int, default=3000, help="total steps")
    p_sim.add_argument("--every", type=int, default=25, help="steps between frames")
    p_sim.add_argument("--save", help="write a .gif or .mp4 instead of opening a window")
    p_sim.add_argument("--fps", type=int, default=20, help="frames per second when saving")
    p_sim.add_argument("--limit", type=float, default=None, help="vorticity colour limit")
    add_common(p_sim)
    p_sim.set_defaults(func=cmd_sim)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
