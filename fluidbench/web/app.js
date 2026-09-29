/* Dashboard wiring: read the machine, run a job, draw what comes back. */

import { LineChart, BarChart, backendColor, legend, invalidateTokens }
  from './charts.js';
import { FlowField, COLORMAPS } from './lbm-gl.js';

const $ = (id) => document.getElementById(id);

const DEFAULT_SWEEP = ['200x50', '400x100', '800x200', '1600x400', '3200x800'];

const state = {
  system: null,
  tab: 'bench',
  job: null,
  source: null,
  results: { bench: [], sweep: [], validate: [] },
  flow: null,
};

/* ---------------------------------------------------------------- utils */

const fmt = {
  mlups: (v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1)),
  ms: (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(3)),
  speedup: (v) => `${v >= 10 ? v.toFixed(1) : v.toFixed(2)}×`,
  sci: (v) => (v === 0 ? '0' : v.toExponential(3).replace('e', 'e')),
  cells: (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(2)} M` : `${(v / 1e3).toFixed(0)} k`),
  mib: (v) => (v >= 1024 ? `${(v / 1024).toFixed(2)} GiB` : `${v.toFixed(0)} MiB`),
};

function parseSize(text) {
  const m = /^\s*(\d+)\s*[x×]\s*(\d+)\s*$/i.exec(text || '');
  return m ? [Number(m[1]), Number(m[2])] : null;
}

function workingSetMb(nx, ny, dtype) {
  // 4 full (ny, nx, 9) arrays plus 6 planar temporaries -- the same model
  // working_set_mb() uses on the server.
  const itemsize = dtype === 'float64' ? 8 : 4;
  return (nx * ny * itemsize * 42) / 1048576;
}

function icon(name) {
  const paths = {
    info: '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 11v5M12 8v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    warn: '<path d="M12 4 2.5 20h19Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4M12 17v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    check: '<path d="M4 12.5 9.5 18 20 6.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>',
    cross: '<path d="M6 6l12 12M18 6 6 18" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>',
  };
  return `<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">${paths[name]}</svg>`;
}

/** Status never travels as colour alone: every badge carries an icon and a word. */
function badge(kind, label) {
  const glyph = kind === 'good' ? 'check' : kind === 'critical' ? 'cross' : 'warn';
  return `<span class="badge ${kind}">${icon(glyph)}${label}</span>`;
}

function note(kind, html) {
  const glyph = kind === 'good' ? 'check' : kind === 'info' ? 'info' : 'warn';
  const div = document.createElement('div');
  div.className = `note ${kind}`;
  div.innerHTML = `${icon(glyph)}<div>${html}</div>`;
  return div;
}

function table(host, headers, rows) {
  const head = `<thead><tr>${headers.map((h) => `<th>${h}</th>`).join('')}</tr></thead>`;
  const body = rows.length
    ? rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')
    : `<tr class="empty-row"><td colspan="${headers.length}">nothing measured yet</td></tr>`;
  host.innerHTML = `${head}<tbody>${body}</tbody>`;
}

function log(text) {
  const el = $('eventLog');
  const stamp = new Date().toLocaleTimeString([], { hour12: false });
  el.insertAdjacentHTML('beforeend', `<span class="t">${stamp}</span>  ${text}\n`);
  el.scrollTop = el.scrollHeight;
}

/* ------------------------------------------------------------- the API */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

/* ------------------------------------------------------------- machine */

async function loadSystem() {
  const system = await api('/api/system');
  state.system = system;

  $('deviceChips').innerHTML = system.backends
    .filter((b) => b.ready)
    .map((b) => `<span class="chip" title="${b.device}">
        <span class="dot" style="background:${backendColor(b.name)}"></span>${b.name}</span>`)
    .join('') || '<span class="chip is-off">no backends</span>';

  const ready = system.backends.filter((b) => b.ready);
  $('backendChecks').innerHTML = system.backends.map((b) => `
    <label class="check ${b.ready ? '' : 'is-disabled'}" title="${b.device}">
      <input type="checkbox" value="${b.name}" ${b.ready ? 'checked' : 'disabled'}>
      <span class="swatch" style="background:${backendColor(b.name)}"></span>
      <span>${b.name}</span>
      <span class="device">${b.ready ? (b.gpu ? 'GPU' : 'CPU') : 'unavailable'}</span>
    </label>`).join('');

  table($('machineTable'),
    ['backend', 'ready', 'device'],
    system.backends.map((b) => [
      `<span class="with-swatch"><span class="swatch" style="background:${backendColor(b.name)}"></span>${b.name}</span>`,
      b.ready ? badge('good', 'yes') : badge('warning', 'no'),
      `<span style="color:var(--text-secondary)">${b.device}</span>`,
    ]));

  $('machineHint').textContent = system.has_gpu
    ? `FluidBench ${system.version} · Python ${system.python} · numpy ${system.numpy} · ${system.platform}`
    : `No GPU backend found — install cupy, or a CUDA build of torch, to have anything to compare against. `
      + `FluidBench ${system.version} · Python ${system.python} · ${system.platform}`;

  if (!system.has_gpu) {
    $('benchNotes').appendChild(note('warning',
      '<strong>No GPU backend on this machine.</strong> <code>bench</code> will still time '
      + 'numpy against torch-cpu, which separates "the GPU is fast" from "numpy is '
      + 'single-threaded" — but the crossover this project is about needs a GPU.'));
  }
  log(`ready — ${ready.map((b) => b.name).join(', ') || 'no backends'}`);
}

/* -------------------------------------------------------------- config */

function selectedBackends() {
  return [...document.querySelectorAll('#backendChecks input:checked')].map((i) => i.value);
}

function selectedSizes() {
  return [...document.querySelectorAll('#sweepSizes input:checked')].map((i) => i.value);
}

function buildSweepSizes() {
  $('sweepSizes').innerHTML = DEFAULT_SWEEP.map((size, index) => {
    const [nx, ny] = parseSize(size);
    return `<label class="check">
      <input type="checkbox" value="${size}" ${index < 4 ? 'checked' : ''}>
      <span>${nx}×${ny}</span>
      <span class="device">${fmt.cells(nx * ny)} cells</span>
    </label>`;
  }).join('');
  $('sweepSizes').addEventListener('change', updateHints);
}

function updateHints() {
  const size = parseSize($('size').value);
  const dtype = $('dtype').value;
  if (size) {
    const [nx, ny] = size;
    const mb = workingSetMb(nx, ny, dtype);
    const tau = Number($('tau').value) || 0.53;
    const re = (0.1 * 2 * 0.13 * ny) / ((tau - 0.5) / 3);
    $('sizeHint').textContent =
      `${fmt.cells(nx * ny)} cells · about ${fmt.mib(mb)} per backend · Re ≈ ${re.toFixed(0)}`;
  } else {
    $('sizeHint').textContent = 'expected NXxNY, for example 400x100';
  }

  $('dtypeHint').textContent = dtype === 'float64'
    ? 'float64 measures the GPU’s double-precision penalty — a GeForce card runs it at 1/32 rate. Use it deliberately, and for validate.'
    : 'float32 is the fair default — a consumer GPU runs float64 at a fraction of the rate.';

  document.querySelectorAll('.presets button').forEach((b) => {
    b.classList.toggle('is-on', b.dataset.size === $('size').value.trim());
  });

  if (state.tab === 'sweep') {
    const sizes = selectedSizes().map(parseSize).filter(Boolean);
    const peak = sizes.length
      ? Math.max(...sizes.map(([nx, ny]) => workingSetMb(nx, ny, dtype))) : 0;
    const warn = peak > 3072;
    $('railError').hidden = !warn;
    if (warn) {
      $('railError').textContent =
        `The largest grid needs roughly ${fmt.mib(peak)} per backend. That will fit on a big card and not on a small one.`;
    }
  }
}

function collectParams(command) {
  const params = {
    dtype: $('dtype').value,
    tau: Number($('tau').value),
    seed: 42,
    backends: selectedBackends(),
    threads: $('threads').value.trim() || 'auto',
  };
  if (!params.backends.length) throw new Error('Select at least one backend.');

  if (command === 'sweep') {
    params.sizes = selectedSizes();
    if (!params.sizes.length) throw new Error('Select at least one grid size.');
    params.steps = Number($('steps').value);
    params.warmup = Number($('warmup').value);
    params.repeats = Number($('repeats').value);
  } else {
    const size = $('size').value.trim();
    if (!parseSize(size)) throw new Error('Grid must look like 400x100.');
    params.size = size;
    if (command === 'bench') {
      params.steps = Number($('steps').value);
      params.warmup = Number($('warmup').value);
      params.repeats = Number($('repeats').value);
    } else {
      params.steps = Number($('steps').value);
      params.tolerance = Number($('tolerance').value);
      params.reference = 'numpy';
      if (!Number.isFinite(params.tolerance) || params.tolerance <= 0) {
        throw new Error('Tolerance must be a positive number, for example 1e-6.');
      }
    }
  }
  return params;
}

/* ----------------------------------------------------------- running */

const RUN_LABEL = { bench: 'Run benchmark', sweep: 'Run sweep', validate: 'Run validation' };

function setRunning(running) {
  const button = $('runButton');
  button.classList.toggle('is-running', running);
  button.querySelector('.run-label').textContent =
    running ? 'Stop after this measurement' : (RUN_LABEL[state.tab] || 'Run');
  $('progressWrap').hidden = !running;
  if (!running) {
    $('progressFill').style.width = '0%';
    $('progressText').textContent = '';
  }
}

async function startRun() {
  if (state.job) {
    await api(`/api/jobs/${state.job}/cancel`, { method: 'POST' }).catch(() => {});
    $('progressText').textContent = 'stopping after the current measurement…';
    return;
  }

  let params;
  try {
    params = collectParams(state.tab);
  } catch (err) {
    $('railError').hidden = false;
    $('railError').textContent = err.message;
    return;
  }
  $('railError').hidden = true;

  state.results[state.tab] = [];
  renderCurrent();

  let job;
  try {
    job = await api('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ command: state.tab, params }),
    });
  } catch (err) {
    $('railError').hidden = false;
    $('railError').textContent = err.message;
    return;
  }

  state.job = job.id;
  setRunning(true);
  listen(job.id);
}

function listen(jobId) {
  state.source?.close();
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.source = source;
  source.onmessage = (message) => handleEvent(JSON.parse(message.data));
  source.onerror = () => {
    // The stream ends by design when the job closes; only an unfinished job
    // reaching here is a real failure.
    if (state.job === jobId) {
      source.close();
      state.job = null;
      setRunning(false);
      log('connection to the run was lost');
    }
  };
}

function handleEvent(event) {
  switch (event.type) {
    case 'started':
      log(`${event.command} started`);
      break;

    case 'plan':
      renderPlan(event);
      break;

    case 'progress': {
      const pct = event.total ? (event.done / event.total) * 100 : 0;
      $('progressFill').style.width = `${pct}%`;
      $('progressText').textContent = event.label
        ? `${event.done}/${event.total} — ${event.label}`
        : `${event.done}/${event.total}`;
      break;
    }

    case 'result':
      state.results[state.tab === 'sweep' ? 'sweep' : 'bench'].push(event.result);
      if (state.tab === 'sweep') renderSweep(); else renderBench();
      log(`${event.result.backend} at ${event.result.nx}x${event.result.ny}: `
        + `${fmt.mlups(event.result.mlups)} MLUPS, ${fmt.ms(event.result.step_ms)} ms/step`);
      break;

    case 'comparison':
      state.results.validate.push(event.row);
      renderValidate();
      log(`${event.row.backend}: max relative drho ${fmt.sci(event.row.max_rel_rho)} `
        + `— ${event.row.passed ? 'pass' : 'FAIL'}`);
      break;

    case 'done':
      log(`finished in ${event.elapsed.toFixed(1)} s`);
      finish();
      break;

    case 'cancelled':
      log('stopped');
      finish();
      break;

    case 'error':
      $('railError').hidden = false;
      $('railError').textContent = event.message;
      log(`error: ${event.message}`);
      finish();
      break;

    case 'closed':
      state.source?.close();
      break;

    default:
      break;
  }
}

function finish() {
  state.job = null;
  setRunning(false);
}

function renderPlan(plan) {
  if (state.tab === 'bench') {
    $('benchChartSub').innerHTML =
      `${plan.steps} timed steps, ${plan.warmup} warm-up, best of ${plan.repeats}, at `
      + `${plan.grid} in ${plan.dtype} — about ${fmt.mib(plan.working_set_mb)} per backend.`;
    log(`grid ${plan.grid}, tau ${plan.tau}, nu ${plan.viscosity.toFixed(4)}, Re ≈ ${plan.reynolds.toFixed(0)}`);
  } else if (state.tab === 'sweep') {
    log(`sweeping ${plan.backends.join(', ')} over ${plan.sizes.join(', ')}`);
  } else if (state.tab === 'validate') {
    log(`comparing against ${plan.reference} after ${plan.steps} steps in ${plan.dtype}`);
  }
}

/* --------------------------------------------------------- bench view */

function baselineOf(results) {
  return results.find((r) => r.backend === 'numpy') || results[0] || null;
}

let benchChart = null;

function renderBench() {
  const results = state.results.bench;
  const base = baselineOf(results);

  if (!benchChart) benchChart = new BarChart($('benchChart'), { suffix: '', measure: 'MLUPS' });

  benchChart.setData(results.map((r) => ({
    label: r.backend,
    value: r.mlups,
    color: backendColor(r.backend),
    note: r.dispatch_bound ? 'launch-bound' : (r.finite ? '' : 'diverged'),
    tip: `<div class="tip-row"><span class="tip-name">ms/step</span>
            <span class="tip-value">${fmt.ms(r.step_ms)}</span></div>
          <div class="tip-row"><span class="tip-name">GB/s (lower bound)</span>
            <span class="tip-value">${r.bandwidth_gbs.toFixed(1)}</span></div>`,
  })), { format: fmt.mlups });

  legend($('benchLegend'), results.map((r) => ({ name: r.backend, color: backendColor(r.backend) })));

  table($('benchTable'),
    ['backend', 'grid', 'dtype', 'steps', 'best (s)', 'ms/step', 'MLUPS', 'GB/s', 'speedup', 'note'],
    results.map((r) => [
      `<span class="with-swatch"><span class="swatch" style="background:${backendColor(r.backend)}"></span>${r.backend}</span>`,
      `${r.nx}×${r.ny}`, r.dtype, r.steps,
      r.seconds.toFixed(4), fmt.ms(r.step_ms), fmt.mlups(r.mlups), r.bandwidth_gbs.toFixed(1),
      base ? fmt.speedup(r.mlups / base.mlups) : '—',
      !r.finite ? badge('critical', 'diverged')
        : r.dispatch_bound ? badge('warning', 'launch-bound') : '',
    ]));

  // Hero: the one number this view is about.
  const hero = $('benchHero');
  if (!results.length || !base) { hero.hidden = true; } else {
    const fastest = results.reduce((a, b) => (b.mlups > a.mlups ? b : a));
    hero.hidden = false;
    const ratio = fastest.mlups / base.mlups;
    $('benchHeroLabel').textContent = `${fastest.backend} vs ${base.backend}`;
    $('benchHeroValue').textContent = fmt.speedup(ratio);
    $('benchHeroSub').textContent = ratio < 1
      ? `At ${fastest.nx}×${fastest.ny} the fastest backend is losing to ${base.backend}. That is the grid being too small, not the hardware being slow.`
      : `${fmt.mlups(fastest.mlups)} MLUPS against ${fmt.mlups(base.mlups)} at ${fastest.nx}×${fastest.ny}, ${fastest.dtype}.`;
  }

  const tiles = $('benchTiles');
  tiles.innerHTML = '';
  if (results.length) {
    const fastest = results.reduce((a, b) => (b.mlups > a.mlups ? b : a));
    const cells = fastest.nx * fastest.ny;
    addTile(tiles, 'Peak throughput', fmt.mlups(fastest.mlups), `MLUPS · ${fastest.backend}`);
    addTile(tiles, 'Time per step', fmt.ms(fastest.step_ms), `ms · ${fastest.backend}`);
    addTile(tiles, 'Effective bandwidth', fastest.bandwidth_gbs.toFixed(1), 'GB/s · minimum-traffic lower bound');
    addTile(tiles, 'Lattice', fmt.cells(cells), `cells · ${fastest.nx}×${fastest.ny}`);
  }

  renderBenchNotes(results);
}

function addTile(host, label, value, sub) {
  const el = document.createElement('div');
  el.className = 'tile';
  el.innerHTML = `<p class="tile-label">${label}</p>
                  <p class="tile-value">${value}</p>
                  <p class="tile-sub">${sub}</p>`;
  host.appendChild(el);
}

function renderBenchNotes(results) {
  const host = $('benchNotes');
  host.innerHTML = '';

  for (const r of results) {
    if (r.dispatch_bound) {
      host.appendChild(note('warning',
        `<p><strong>${r.backend} is launch-bound at ${r.nx}×${r.ny}.</strong> It spent
         ${Math.round(r.dispatch_fraction * 100)}% of each step merely queueing kernels.
         Each step issues roughly 60 of them, and at this size the GPU drains the queue as
         fast as Python can fill it — so this measures host dispatch overhead, not the
         device.</p>
         <p>Raise the grid size until that fraction falls. A speedup quoted at a
         launch-bound size is a number about your CPU.</p>`));
    }
    if (r.spread > 1.25) {
      host.appendChild(note('warning',
        `<strong>${r.backend} timings varied by ${r.spread.toFixed(2)}× across repeats.</strong>
         Something else is using the device, or the clocks are still moving. Raise the
         warm-up, or the repeat count, before trusting the number.`));
    }
    if (!r.finite) {
      host.appendChild(note('critical',
        `<strong>${r.backend} diverged.</strong> The timing still stands — the arithmetic
         happened — but the physics does not. Lower the free-stream velocity or raise τ.`));
    }
  }

  if (results.length > 1 && results.some((r) => r.backend === 'numpy')
      && results.some((r) => r.backend === 'torch-cpu')) {
    host.appendChild(note('info',
      `<strong>numpy is the baseline on purpose.</strong> It is single-threaded for
       elementwise work, which is why torch-cpu — same CPU, multithreaded kernels —
       runs about 2× faster. That row separates "the GPU is fast" from "numpy is
       single-threaded".`));
  }
}

/* --------------------------------------------------------- sweep view */

let sweepChart = null;
let speedupChart = null;

function renderSweep() {
  const results = state.results.sweep;
  if (!sweepChart) {
    sweepChart = new LineChart($('sweepChart'), {
      xScale: 'log', yScale: 'log',
      xLabel: 'lattice cells', yLabel: 'MLUPS',
      xFormat: (v) => fmt.cells(v),
      yFormat: (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1)),
      tipTitle: (v) => `${fmt.cells(v)} cells`,
    });
    speedupChart = new LineChart($('speedupChart'), {
      xScale: 'log', yScale: 'log',
      xLabel: 'lattice cells', yLabel: 'speedup vs numpy',
      xFormat: (v) => fmt.cells(v),
      yFormat: (v) => `${v >= 10 ? v.toFixed(0) : v.toFixed(1)}×`,
      tipTitle: (v) => `${fmt.cells(v)} cells`,
      baseline: 1, baselineLabel: 'parity with numpy',
    });
  }

  const names = [...new Set(results.map((r) => r.backend))];
  const byCells = (name) => results
    .filter((r) => r.backend === name)
    .map((r) => ({ x: r.nx * r.ny, y: r.mlups, meta: r }))
    .sort((a, b) => a.x - b.x);

  sweepChart.setData(names.map((name) => ({
    name, color: backendColor(name), points: byCells(name),
  })));
  legend($('sweepLegend'), names.map((n) => ({ name: n, color: backendColor(n) })));

  // Speedup is measured against numpy at the *same* grid size, never against
  // numpy's best.
  const cpuAt = new Map(results.filter((r) => r.backend === 'numpy')
    .map((r) => [r.nx * r.ny, r.mlups]));
  const others = names.filter((n) => n !== 'numpy');

  speedupChart.setData(others.map((name) => ({
    name, color: backendColor(name),
    points: results.filter((r) => r.backend === name && cpuAt.has(r.nx * r.ny))
      .map((r) => ({ x: r.nx * r.ny, y: r.mlups / cpuAt.get(r.nx * r.ny) }))
      .sort((a, b) => a.x - b.x),
  })));
  legend($('speedupLegend'), others.map((n) => ({ name: n, color: backendColor(n) })));

  const sizes = [...new Set(results.map((r) => `${r.nx}x${r.ny}`))];
  table($('sweepTable'),
    ['backend', 'grid', 'cells', 'ms/step', 'MLUPS', 'GB/s', 'speedup', 'note'],
    results.map((r) => {
      const cpu = cpuAt.get(r.nx * r.ny);
      return [
        `<span class="with-swatch"><span class="swatch" style="background:${backendColor(r.backend)}"></span>${r.backend}</span>`,
        `${r.nx}×${r.ny}`, fmt.cells(r.nx * r.ny),
        fmt.ms(r.step_ms), fmt.mlups(r.mlups), r.bandwidth_gbs.toFixed(1),
        cpu ? fmt.speedup(r.mlups / cpu) : '—',
        !r.finite ? badge('critical', 'diverged')
          : r.dispatch_bound ? badge('warning', 'launch-bound') : '',
      ];
    }));

  renderCrossover(results, sizes, cpuAt, others);
}

function renderCrossover(results, sizes, cpuAt, others) {
  const host = $('crossoverCards');
  host.innerHTML = '';
  const hero = $('sweepHero');
  hero.hidden = true;

  let bestOverall = null;
  for (const name of others) {
    const rows = results.filter((r) => r.backend === name && cpuAt.has(r.nx * r.ny))
      .map((r) => ({ cells: r.nx * r.ny, grid: `${r.nx}×${r.ny}`,
                     ratio: r.mlups / cpuAt.get(r.nx * r.ny) }))
      .sort((a, b) => a.cells - b.cells);
    if (!rows.length) continue;

    const best = rows.reduce((a, b) => (b.ratio > a.ratio ? b : a));
    const crossed = rows.find((r) => r.ratio > 1);
    addTile(host, `${name} — best speedup`, fmt.speedup(best.ratio),
      crossed ? `beats numpy from ${crossed.grid}` : 'never beats numpy in this range');
    if (!bestOverall || best.ratio > bestOverall.ratio) {
      bestOverall = { ...best, name, crossed };
    }
  }

  if (bestOverall && others.length) {
    hero.hidden = false;
    $('sweepHeroValue').textContent = fmt.speedup(bestOverall.ratio);
    $('sweepHeroSub').textContent = bestOverall.crossed
      ? `${bestOverall.name} at ${bestOverall.grid}. It first overtakes numpy at ${bestOverall.crossed.grid}, and the ratio stops growing once both sides are bandwidth bound.`
      : `${bestOverall.name} at ${bestOverall.grid} — still behind numpy across every size measured. Sweep further up.`;
  }

  const host2 = $('sweepNotes');
  host2.innerHTML = '';
  if (sizes.length > 1) {
    const bound = results.filter((r) => r.dispatch_bound);
    if (bound.length) {
      const smallest = bound.reduce((a, b) => (a.nx * a.ny < b.nx * b.ny ? a : b));
      const largest = bound.reduce((a, b) => (a.nx * a.ny > b.nx * b.ny ? a : b));
      host2.appendChild(note('warning',
        `<p><strong>Launch-bound up to ${largest.nx}×${largest.ny}.</strong> From
         ${smallest.nx}×${smallest.ny} upward the GPU spent most of each step waiting
         for Python to queue the next kernel. Those rows measure host overhead.</p>
         <p>This is the whole point of sweeping: below roughly a million cells the device
         is never the bottleneck, so a speedup quoted there says nothing about the
         hardware.</p>`));
    }
    host2.appendChild(note('info',
      `<strong>The sweep holds τ and the inflow velocity fixed while the grid grows</strong>,
       so the effective Reynolds number rises with size. Timings stay valid either way —
       the arithmetic is identical — and anything that does diverge is flagged rather
       than quietly reported.`));
  }
}

/* ------------------------------------------------------ validate view */

function renderValidate() {
  const rows = state.results.validate;
  table($('validateTable'),
    ['backend', 'max |Δρ|', 'max relative Δρ', 'max |Δu|', 'tolerance', ''],
    rows.map((r) => [
      `<span class="with-swatch"><span class="swatch" style="background:${backendColor(r.backend)}"></span>${r.backend}</span>`,
      fmt.sci(r.max_abs_rho), fmt.sci(r.max_rel_rho), fmt.sci(r.max_abs_u),
      r.tolerance.toExponential(0),
      r.passed ? badge('good', 'pass') : badge('critical', 'fail'),
    ]));

  const host = $('validateNotes');
  host.innerHTML = '';
  if (!rows.length) return;

  if ($('dtype').value === 'float32') {
    host.appendChild(note('warning',
      `<strong>This ran in float32, where a tight tolerance is meaningless.</strong>
       The rounding floor sits around 10⁻⁷, so agreement here says only that nothing
       is badly wrong. Switch to float64 and a tolerance of 1e-9 for a real comparison.`));
  }
  const failed = rows.filter((r) => !r.passed);
  host.appendChild(failed.length
    ? note('critical', `<strong>${failed.map((r) => r.backend).join(', ')} drifted past the tolerance.</strong>
        Either the tolerance is tighter than the precision allows, or the backends are no
        longer running the same workload — which would invalidate every timing on the
        other tabs.`)
    : note('good', `<strong>Every backend tracked ${rows[0].reference} to within the tolerance.</strong>
        The timings on the other tabs are comparing the same workload, which is the claim
        this whole project rests on.`));
}

/* ----------------------------------------------------------- the flow */

const MAP_GRADIENTS = [
  'linear-gradient(90deg,#9ec5f4,#3987e5,#383835,#e34948,#f5a3a2)',
  'linear-gradient(90deg,#5eead4,#199e70,#2c2c2a,#c98500,#fbd07a)',
  'linear-gradient(90deg,#0d0f17,#9ec5f4)',
];

function setupFlow() {
  const canvas = $('flow');
  let field;
  try {
    field = new FlowField(canvas, { dim: 0.82 });
  } catch (err) {
    canvas.hidden = true;
    const fallback = $('glFallback');
    fallback.hidden = false;
    fallback.innerHTML = `<strong>The background flow is off.</strong> ${err.message}. `
      + 'Everything else on this page works — the benchmark runs in Python, not here.';
    document.querySelector('#panel-flow .controls')?.setAttribute('hidden', '');
    log(`flow preview unavailable: ${err.message}`);
    return;
  }

  state.flow = field;

  $('flowMap').innerHTML = COLORMAPS
    .map((m) => `<option value="${m.id}">${m.name}</option>`).join('');

  const sync = (status) => {
    $('flowStatus').innerHTML = status.diverged
      ? `${badge('critical', 'diverged')}`
      : `<span><b>${status.grid}</b> grid</span>
         <span><b>${status.steps.toLocaleString()}</b> steps</span>
         <span>Re ≈ <b>${status.reynolds.toFixed(0)}</b></span>
         ${status.spinning ? `<span>spinning up — ${Math.round(status.spinProgress * 100)}%</span>` : ''}`;
    if (status.diverged) {
      $('flowStatus').innerHTML += ' <span>the preview blew up — reseed, or raise τ</span>';
    }
    drawScaleLegend(status.limit);
  };
  field.onStatus = sync;

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.6);
    if (field.resize(window.innerWidth, window.innerHeight, dpr)) {
      field.reset({ width: window.innerWidth, height: window.innerHeight });
    }
  };
  resize();
  window.addEventListener('resize', debounce(resize, 220));

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (reduced.matches) {
    field.renderStill();
    $('flowPause').textContent = 'Play';
    log('reduced motion is on — the flow is a still frame');
  } else {
    field.start();
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) field.stop();
    else if (!reduced.matches && $('flowPause').textContent === 'Pause') field.start();
  });

  // -- controls
  const bind = (id, valueId, format, apply) => {
    const input = $(id);
    const show = () => { $(valueId).textContent = format(Number(input.value)); };
    input.addEventListener('input', () => { show(); apply(Number(input.value)); });
    show();
  };

  bind('flowTau', 'flowTauValue', (v) => v.toFixed(3),
    (v) => field.setParams({ tau: v }));
  bind('flowInflow', 'flowInflowValue', (v) => v.toFixed(3),
    (v) => field.setParams({ inflow: v }));
  bind('flowRows', 'flowRowsValue', (v) => `${v} rows`,
    (v) => field.setParams({ rows: v }));
  bind('flowSpeed', 'flowSpeedValue', (v) => `${v}`,
    (v) => field.setParams({ stepsPerFrame: v }));
  bind('flowContrast', 'flowContrastValue', (v) => `${v.toFixed(2)}×`,
    (v) => field.setParams({ contrast: v }));
  bind('flowDim', 'flowDimValue', (v) => `${v}%`,
    (v) => field.setParams({ dim: v / 100 }));

  $('flowMap').addEventListener('change', (e) => {
    const id = Number(e.target.value);
    field.setParams({ colormap: id });
    $('flowMapNote').textContent = COLORMAPS[id].note;
    drawScaleLegend(field.limit);
  });
  $('flowMapNote').textContent = COLORMAPS[0].note;

  $('flowSustained').addEventListener('change', (e) => {
    field.setParams({ sustained: e.target.checked });
    log(e.target.checked
      ? 'inlet on — the wake is driven'
      : 'inlet off — periodic and unforced, like the Python solver: the flow will decay');
  });

  $('flowPause').addEventListener('click', () => {
    const button = $('flowPause');
    if (field.running) { field.stop(); button.textContent = 'Play'; }
    else { field.start(); button.textContent = 'Pause'; }
  });

  $('flowReseed').addEventListener('click', () => {
    field.reseed();
    if (!field.running && !reduced.matches) field.start();
    log('flow reseeded');
  });

  $('flowFocus').addEventListener('click', () => setFocus(true));
  drawScaleLegend(field.limit);
}

function drawScaleLegend(limit) {
  const id = Number($('flowMap').value || 0);
  const signed = id !== 2;
  $('scaleLegend').innerHTML = `
    <div class="scale-bar" style="background:${MAP_GRADIENTS[id]}"></div>
    <div class="scale-marks">
      <span>${signed ? `−${limit.toExponential(1)}` : '0'}</span>
      <span>${signed ? '0' : 'vorticity magnitude'}</span>
      <span>+${limit.toExponential(1)}</span>
    </div>`;
}

function setFocus(on) {
  document.body.classList.toggle('is-focused', on);
  $('focusHint').hidden = !on;
  if (on) state.flow?.setParams({ dim: 1 });
  else state.flow?.setParams({ dim: Number($('flowDim').value) / 100 });
}

function debounce(fn, ms) {
  let handle;
  return (...args) => { clearTimeout(handle); handle = setTimeout(() => fn(...args), ms); };
}

/* ------------------------------------------------------------- chrome */

function renderCurrent() {
  if (state.tab === 'bench') renderBench();
  else if (state.tab === 'sweep') renderSweep();
  else if (state.tab === 'validate') renderValidate();
}

function selectTab(name) {
  state.tab = name;
  document.querySelectorAll('.tab').forEach((tab) => {
    const on = tab.id === `tab-${name}`;
    tab.classList.toggle('is-active', on);
    tab.setAttribute('aria-selected', String(on));
  });
  document.querySelectorAll('.panel').forEach((panel) => {
    panel.hidden = panel.id !== `panel-${name}`;
  });

  const runnable = ['bench', 'sweep', 'validate'].includes(name);
  $('runButton').disabled = !runnable;
  $('sizesField').hidden = name !== 'sweep';
  $('toleranceField').hidden = name !== 'validate';
  document.querySelector('.field:has(#size)').hidden = name === 'sweep';
  if (!state.job) setRunning(false);

  // validate has no repeats or warm-up; hiding them beats showing controls
  // that do nothing.
  $('warmup').closest('.field-row').hidden = name === 'validate';
  $('repeats').closest('.field-row').hidden = name === 'validate';
  $('steps').value = name === 'sweep' ? 100 : name === 'validate' ? 100 : 200;

  updateHints();
  if (runnable) renderCurrent();
}

function setupTheme() {
  const stored = localStorage.getItem('fluidbench-theme');
  const initial = stored
    || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  applyTheme(initial);
  $('themeToggle').addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('fluidbench-theme', theme); } catch { /* private mode */ }
  invalidateTokens();
  // The flow is bright at the extremes, which a light page cannot carry at
  // full strength; the canvas opacity in style.css does the rest.
  state.flow?.setParams({ colormap: Number($('flowMap')?.value || 0) });
  renderCurrent();
  if (state.system) {
    $('deviceChips').querySelectorAll('.dot').forEach((dot, index) => {
      const ready = state.system.backends.filter((b) => b.ready);
      if (ready[index]) dot.style.background = backendColor(ready[index].name);
    });
  }
}

function setupCopy() {
  document.querySelectorAll('[data-copy]').forEach((button) => {
    button.addEventListener('click', async () => {
      const el = $(button.dataset.copy);
      const rows = [...el.querySelectorAll('tr')]
        .map((tr) => [...tr.children].map((cell) => cell.innerText.trim()).join('\t'))
        .join('\n');
      try {
        await navigator.clipboard.writeText(rows);
        const original = button.textContent;
        button.textContent = 'Copied';
        setTimeout(() => { button.textContent = original; }, 1400);
      } catch {
        button.textContent = 'Copy failed';
      }
    });
  });
}

/* ---------------------------------------------------------------- boot */

function boot() {
  buildSweepSizes();
  setupTheme();
  setupCopy();

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => selectTab(tab.id.replace('tab-', '')));
  });
  document.querySelectorAll('.presets button').forEach((button) => {
    button.addEventListener('click', () => {
      $('size').value = button.dataset.size;
      updateHints();
    });
  });
  ['size', 'dtype', 'tau'].forEach((id) => $(id).addEventListener('input', updateHints));
  $('dtype').addEventListener('change', updateHints);
  $('runButton').addEventListener('click', startRun);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.body.classList.contains('is-focused')) setFocus(false);
  });
  $('focusHint').addEventListener('click', () => setFocus(false));

  selectTab('bench');
  setupFlow();
  loadSystem().catch((err) => {
    $('railError').hidden = false;
    $('railError').textContent = `Could not reach the server: ${err.message}`;
  });
}

boot();
