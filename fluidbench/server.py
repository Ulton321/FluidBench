"""Local web front end.

    python main.py serve

Serves a dashboard on localhost that drives the *real* benchmark: `bench`,
`sweep` and `validate` call the same functions the CLI does, on the same
backends, so a number in the browser is the number in the terminal.

Three things this deliberately does not do:

* **No new dependencies.**  Stdlib `http.server` only.  FluidBench installs
  with numpy and matplotlib, and that matters on the kind of machine you
  actually want to benchmark -- a cluster node, a fresh CUDA container.
* **Nothing here is timed.**  The server thread marshals results; it never
  sits between the clock and the solver.  Requests are handled on other
  threads, and a global lock keeps two runs from ever overlapping, because
  two benchmarks sharing a GPU measure neither.
* **No simulation.**  The flow in the browser is a separate WebGL
  reimplementation (see `web/lbm-gl.js`), and the page says so.  The timed
  kernel is the Python one, here.

Results stream to the page over Server-Sent Events, which is a plain HTTP
response that never ends -- no websocket library, no framing to get wrong.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .backends import (
    TorchBackend,
    all_backend_names,
    available_backends,
    get_backend,
    resolve_names,
)
from .benchmark import compare_backends, run_benchmark, working_set_mb
from .lbm import LBMConfig

__all__ = ["serve", "JobManager"]

WEB_ROOT = Path(__file__).resolve().parent / "web"

#: Sent every 15 s of silence so proxies and browsers keep the stream open.
HEARTBEAT_SECONDS = 15.0

#: Upper bounds on what a request may ask for.  The server is bound to
#: localhost, but a typo in a text field should not fill up VRAM either.
MAX_CELLS = 64_000_000
MAX_STEPS = 100_000
MAX_REPEATS = 25
MAX_SIZES = 12


# --------------------------------------------------------------------------
# Backend cache
# --------------------------------------------------------------------------

_backends: dict[tuple[str, int | None], object] = {}
_backend_lock = threading.Lock()


def cached_backend(name: str, threads: int | None = None):
    """Build a backend once and reuse it.

    `sweep` already does this within a run, for the reason that matters:
    CUDA context creation costs hundreds of milliseconds and must not land
    on the first measurement.  Across runs from a browser the same argument
    applies, so the cache outlives the job.
    """
    key = (name, threads)
    with _backend_lock:
        backend = _backends.get(key)
        if backend is None:
            backend = get_backend(name)
            if threads and isinstance(backend, TorchBackend) and not backend.is_gpu:
                backend.set_num_threads(threads)
            _backends[key] = backend
        return backend


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class Cancelled(Exception):
    """Raised inside a runner when the browser asked it to stop."""


class Job:
    """One benchmark run, and the event log the page replays it from.

    Every event is kept, so a reload mid-run resumes the stream from the
    beginning rather than showing a half-empty table.
    """

    def __init__(self, job_id: str, command: str, params: dict):
        self.id = job_id
        self.command = command
        self.params = params
        self.created = time.time()
        self.events: list[dict] = []
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.cancelled = threading.Event()
        self.finished = threading.Event()

    def emit(self, type: str, **payload) -> None:
        event = {"seq": 0, "type": type, **payload}
        with self._lock:
            event["seq"] = len(self.events)
            self.events.append(event)
            subscribers = list(self._subscribers)
        for sub in subscribers:
            sub.put(event)

    def subscribe(self) -> tuple[list[dict], queue.Queue]:
        """History so far plus a queue of everything after it, atomically."""
        sub: queue.Queue = queue.Queue()
        with self._lock:
            history = list(self.events)
            self._subscribers.append(sub)
        return history, sub

    def unsubscribe(self, sub: queue.Queue) -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)

    def checkpoint(self) -> None:
        """Bail out if the browser pressed stop.

        Cancellation lands between measurements, never inside one: a run that
        stopped halfway through its repeats would report a time for work it
        did not finish.
        """
        if self.cancelled.is_set():
            raise Cancelled


class JobManager:
    """Runs one job at a time and remembers the last few."""

    HISTORY = 24

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self.current: str | None = None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self.jobs.get(job_id)

    def busy(self) -> bool:
        with self._lock:
            if self.current is None:
                return False
            job = self.jobs.get(self.current)
            return job is not None and not job.finished.is_set()

    def submit(self, command: str, params: dict) -> Job:
        job = Job(uuid.uuid4().hex[:12], command, params)
        with self._lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
            while len(self.order) > self.HISTORY:
                self.jobs.pop(self.order.pop(0), None)
            self.current = job.id
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        # Serialised deliberately: two benchmarks sharing a device measure
        # neither of them.  Queued jobs wait here rather than being refused,
        # but the UI declines to submit while one is in flight.
        with self._run_lock:
            start = time.perf_counter()
            try:
                job.emit("started", command=job.command, params=job.params)
                RUNNERS[job.command](job)
                job.emit("done", elapsed=time.perf_counter() - start)
            except Cancelled:
                job.emit("cancelled", elapsed=time.perf_counter() - start)
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                job.emit("error", message=str(exc) or type(exc).__name__)
            except Exception as exc:  # pragma: no cover - unexpected
                traceback.print_exc()
                job.emit("error", message=f"{type(exc).__name__}: {exc}")
            finally:
                job.finished.set()
                job.emit("closed")


# --------------------------------------------------------------------------
# Parameter parsing -- everything crossing the wire is checked
# --------------------------------------------------------------------------


def _int(params: dict, key: str, default: int, low: int, high: int) -> int:
    value = params.get(key, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a whole number") from None
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}, got {value}")
    return value


def _float(params: dict, key: str, default: float, low: float, high: float) -> float:
    value = params.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number") from None
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}, got {value}")
    return value


def _size(value, label: str = "size") -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().split("x")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"{label} must look like 400x100")
    if len(parts) != 2:
        raise ValueError(f"{label} must look like 400x100, got {value!r}")
    try:
        nx, ny = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        raise ValueError(f"{label} must look like 400x100, got {value!r}") from None
    if nx < 8 or ny < 8:
        raise ValueError("grid must be at least 8x8")
    if nx * ny > MAX_CELLS:
        raise ValueError(f"{nx}x{ny} is {nx * ny / 1e6:.1f} M cells; the cap is 64 M")
    return nx, ny


def _config(params: dict) -> LBMConfig:
    nx, ny = _size(params.get("size", "400x100"))
    dtype = params.get("dtype", "float32")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    return LBMConfig(
        nx=nx,
        ny=ny,
        tau=_float(params, "tau", 0.53, 0.5001, 5.0),
        dtype=dtype,
        seed=_int(params, "seed", 42, 0, 2**31 - 1),
    )


def _names(params: dict) -> list[str]:
    requested = params.get("backends") or ["all"]
    if isinstance(requested, str):
        requested = [requested]
    names = resolve_names([str(n) for n in requested])
    available = set(available_backends())
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"not available on this machine: {', '.join(missing)}")
    if not names:
        raise ValueError("select at least one backend")
    return names


def _threads(params: dict) -> int | None:
    value = params.get("threads")
    if value in (None, "", "auto"):
        return None
    return _int(params, "threads", 0, 1, 1024)


def _timing(params: dict, steps: int, warmup: int) -> tuple[int, int, int]:
    return (
        _int(params, "steps", steps, 1, MAX_STEPS),
        _int(params, "warmup", warmup, 0, MAX_STEPS),
        _int(params, "repeats", 3, 1, MAX_REPEATS),
    )


# --------------------------------------------------------------------------
# Runners -- thin wrappers over the same calls cli.py makes
# --------------------------------------------------------------------------


def _result_payload(result) -> dict:
    data = result.to_dict()
    data.pop("seconds_all", None)
    data["seconds_all"] = list(result.seconds_all)
    return data


def run_bench(job: Job) -> None:
    params = job.params
    config = _config(params)
    names = _names(params)
    threads = _threads(params)
    steps, warmup, repeats = _timing(params, 200, 20)

    job.emit(
        "plan",
        grid=config.label(),
        cells=config.cells,
        dtype=config.dtype,
        tau=config.tau,
        viscosity=config.viscosity,
        reynolds=config.reynolds,
        working_set_mb=working_set_mb(config),
        steps=steps,
        warmup=warmup,
        repeats=repeats,
        backends=names,
        total=len(names),
    )

    for index, name in enumerate(names):
        job.checkpoint()
        backend = cached_backend(name, threads)
        job.emit("progress", done=index, total=len(names), label=name,
                 device=backend.describe())
        result = run_benchmark(backend, config, steps, warmup, repeats)
        job.emit("result", result=_result_payload(result))
    job.emit("progress", done=len(names), total=len(names), label="")


def run_sweep(job: Job) -> None:
    params = job.params
    raw_sizes = params.get("sizes") or ["200x50", "400x100", "800x200", "1600x400"]
    if isinstance(raw_sizes, str):
        raw_sizes = [s for s in raw_sizes.split(",") if s.strip()]
    if len(raw_sizes) > MAX_SIZES:
        raise ValueError(f"at most {MAX_SIZES} grid sizes per sweep")
    sizes = [_size(s) for s in raw_sizes]
    if not sizes:
        raise ValueError("pick at least one grid size")

    names = _names(params)
    threads = _threads(params)
    steps, warmup, repeats = _timing(params, 100, 10)
    base = _config(params)

    biggest = LBMConfig(nx=sizes[-1][0], ny=sizes[-1][1], tau=base.tau, dtype=base.dtype)
    job.emit(
        "plan",
        sizes=[f"{nx}x{ny}" for nx, ny in sizes],
        dtype=base.dtype,
        tau=base.tau,
        steps=steps,
        warmup=warmup,
        repeats=repeats,
        backends=names,
        peak_working_set_mb=working_set_mb(biggest),
        total=len(sizes) * len(names),
    )

    # Backends are built once, outside the loop, so no measurement is charged
    # for context creation.
    built = {name: cached_backend(name, threads) for name in names}

    done = 0
    total = len(sizes) * len(names)
    for nx, ny in sizes:
        config = LBMConfig(
            nx=nx, ny=ny, tau=base.tau, radius=base.radius, inflow=base.inflow,
            rho0=base.rho0, noise=base.noise, seed=base.seed, dtype=base.dtype,
        )
        for name in names:
            job.checkpoint()
            job.emit("progress", done=done, total=total, label=f"{name} at {nx}x{ny}")
            result = run_benchmark(built[name], config, steps, warmup, repeats)
            done += 1
            job.emit("result", result=_result_payload(result))
    job.emit("progress", done=total, total=total, label="")


def run_validate(job: Job) -> None:
    params = job.params
    config = _config(params)
    names = _names(params)
    reference = params.get("reference") or "numpy"
    if reference not in available_backends():
        raise ValueError(f"reference backend {reference!r} is not available")
    if reference not in names:
        names = [reference, *names]
    others = [n for n in names if n != reference]
    if not others:
        raise ValueError(f"nothing to compare against {reference}")

    steps = _int(params, "steps", 100, 1, MAX_STEPS)
    tolerance = _float(params, "tolerance", 1e-6, 1e-15, 1.0)

    job.emit(
        "plan",
        grid=config.label(),
        dtype=config.dtype,
        steps=steps,
        tolerance=tolerance,
        reference=reference,
        backends=others,
        total=len(others),
        loose=config.dtype == "float32",
    )

    # Warm the backends before the reference run so the page shows the import
    # cost as progress rather than as an unexplained pause.
    for name in names:
        job.checkpoint()
        job.emit("progress", done=0, total=len(others), label=f"preparing {name}")
        cached_backend(name, _threads(params))

    job.emit("progress", done=0, total=len(others), label=f"reference run on {reference}")

    seen = 0

    def landed(result) -> None:
        nonlocal seen
        seen += 1
        row = asdict(result)
        row["passed"] = result.passed
        job.emit("comparison", row=row)
        job.emit("progress", done=seen, total=len(others), label="")

    compare_backends(names, config, steps, reference, tolerance, on_result=landed)


RUNNERS = {"bench": run_bench, "sweep": run_sweep, "validate": run_validate}


# --------------------------------------------------------------------------
# System description
# --------------------------------------------------------------------------


def describe_system() -> dict:
    import platform

    import numpy as np

    available = set(available_backends())
    backends = []
    for name in all_backend_names():
        entry = {"name": name, "ready": name in available, "gpu": False, "device": ""}
        if name in available:
            try:
                backend = cached_backend(name)
                entry["device"] = backend.describe()
                entry["gpu"] = bool(backend.is_gpu)
            except Exception as exc:  # pragma: no cover - driver problems
                entry["ready"] = False
                entry["device"] = f"error: {exc}"
        else:
            entry["device"] = "not installed / no device"
        backends.append(entry)

    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "backends": backends,
        "has_gpu": any(b["gpu"] and b["ready"] for b in backends),
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"FluidBench/{__version__}"
    jobs: JobManager  # set on the server class

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
        if self.server.verbose:  # type: ignore[attr-defined]
            sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    def _send(self, status: int, body: bytes, content_type: str, **headers) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8",
                   Cache_Control="no-store")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 1 << 20:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed JSON body: {exc}") from None
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        path = urlparse(self.path).path
        try:
            if path.startswith("/api/"):
                self._api_get(path)
            else:
                self._static(path)
        except BrokenPipeError:  # pragma: no cover - browser went away
            pass
        except ValueError as exc:
            self._error(400, str(exc))

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        path = urlparse(self.path).path
        try:
            params = self._body()
        except ValueError as exc:
            self._error(400, str(exc))
            return
        try:
            if path == "/api/jobs":
                self._start_job(params)
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                self._cancel_job(path.split("/")[3])
            else:
                self._error(404, f"no route for POST {path}")
        except BrokenPipeError:  # pragma: no cover
            pass
        except ValueError as exc:
            self._error(400, str(exc))

    def _api_get(self, path: str) -> None:
        if path == "/api/system":
            self._json(200, describe_system())
        elif path.startswith("/api/jobs/") and path.endswith("/events"):
            self._stream(path.split("/")[3])
        elif path.startswith("/api/jobs/"):
            job = self.jobs.get(path.split("/")[3])
            if job is None:
                self._error(404, "no such job")
            else:
                self._json(200, {"id": job.id, "command": job.command,
                                 "finished": job.finished.is_set(),
                                 "events": job.events})
        else:
            self._error(404, f"no route for GET {path}")

    def _start_job(self, params: dict) -> None:
        command = params.get("command")
        if command not in RUNNERS:
            self._error(400, f"command must be one of {', '.join(RUNNERS)}")
            return
        if self.jobs.busy():
            self._error(409, "a run is already in flight; stop it first")
            return
        job = self.jobs.submit(command, params.get("params") or {})
        self._json(202, {"id": job.id})

    def _cancel_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            self._error(404, "no such job")
            return
        job.cancelled.set()
        self._json(200, {"id": job.id, "cancelled": True})

    # -- server-sent events ------------------------------------------------

    def _stream(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            self._error(404, "no such job")
            return

        history, sub = job.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        # No Content-Length: the response ends when the run does, so the
        # connection has to close rather than be reused.
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        try:
            for event in history:
                self._event(event)
            closed = any(e["type"] == "closed" for e in history)
            while not closed:
                try:
                    event = sub.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self._event(event)
                closed = event["type"] == "closed"
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # the page navigated away mid-run; the job carries on
        finally:
            job.unsubscribe(sub)

    def _event(self, event: dict) -> None:
        payload = json.dumps(event, default=float)
        self.wfile.write(f"id: {event['seq']}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    # -- static files ------------------------------------------------------

    def _static(self, path: str) -> None:
        relative = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if not target.is_file() or WEB_ROOT not in target.parents:
            self._error(404, f"not found: {path}")
            return
        body = target.read_bytes()
        content_type = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        # The page is served from disk on every request: edit a file, reload,
        # see it.  Nothing here is big enough for caching to be worth the
        # confusion.
        self._send(200, body, content_type, Cache_Control="no-store")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, verbose: bool = False):
        self.verbose = verbose
        super().__init__(address, handler)


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    verbose: bool = False,
) -> int:
    """Run the dashboard until interrupted."""
    if not WEB_ROOT.is_dir():
        raise RuntimeError(f"web assets are missing from {WEB_ROOT}")

    handler = type("BoundHandler", (Handler,), {"jobs": JobManager()})
    try:
        httpd = Server((host, port), handler, verbose=verbose)
    except OSError as exc:
        raise RuntimeError(
            f"cannot bind {host}:{port} -- {exc}. Try --port {port + 1}."
        ) from None

    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "::1") else host
    url = f"http://{shown}:{httpd.server_address[1]}/"
    print(f"FluidBench dashboard on {url}")
    if host == "0.0.0.0":
        print("  bound to every interface -- anything on your network can start runs here")
    print("  press ctrl-c to stop")

    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0
