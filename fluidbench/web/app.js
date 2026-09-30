/* FluidBench viewer: a full-screen lattice Boltzmann simulation and the
 * handful of controls that drive it.  All the physics lives in lbm-gl.js. */

import { FlowField, COLORMAPS } from './lbm-gl.js';

const $ = (id) => document.getElementById(id);
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

const canvas = $('flow');
let field;

try {
  field = new FlowField(canvas);
} catch (err) {
  canvas.hidden = true;
  $('ui').hidden = true;
  $('fallback').hidden = false;
  $('fallbackReason').textContent = `${err.message}.`;
  throw err;
}

const ui = {
  paused: false,
  hidden: false,
};

/* -- sizing ------------------------------------------------------------ */

function fit() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const changed = field.resize(window.innerWidth, window.innerHeight, dpr);
  return changed;
}

function rebuild() {
  field.reset({ width: window.innerWidth, height: window.innerHeight });
  if (ui.paused) setPaused(false);
}

let resizeTimer = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (fit()) rebuild(); }, 200);
});

/* -- playback ---------------------------------------------------------- */

function setPaused(paused) {
  ui.paused = paused;
  if (paused) field.stop(); else field.start();
  const btn = $('btnPlay');
  btn.classList.toggle('is-paused', paused);
  btn.setAttribute('aria-label', paused ? 'Play' : 'Pause');
  btn.dataset.tip = paused ? 'Play · Space' : 'Pause · Space';
  $('btnStep').disabled = !paused;
}

function stepOnce() {
  if (!ui.paused || field.diverged) return;
  field.advance(field.opts.stepsPerFrame);
  field.render();
  updateStats();
}

function restart() {
  field.reseed();
  $('alert').hidden = true;
  if (ui.paused) setPaused(false);
}

function setHidden(hidden) {
  ui.hidden = hidden;
  $('ui').classList.toggle('is-hidden', hidden);
  const restore = $('restore');
  restore.hidden = !hidden;
  if (hidden) {                       // restart the fade each time
    restore.style.animation = 'none';
    void restore.offsetWidth;
    restore.style.animation = '';
  }
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

$('btnPlay').addEventListener('click', () => setPaused(!ui.paused));
$('btnStep').addEventListener('click', stepOnce);
$('btnReset').addEventListener('click', restart);
$('alertReset').addEventListener('click', restart);
$('btnHide').addEventListener('click', () => setHidden(true));
$('btnFull').addEventListener('click', toggleFullscreen);

$('btnCollapse').addEventListener('click', () => {
  const panel = $('panel');
  const collapsed = panel.classList.toggle('is-collapsed');
  $('btnCollapse').setAttribute('aria-expanded', String(!collapsed));
  $('btnCollapse').setAttribute('aria-label', collapsed ? 'Expand panel' : 'Collapse panel');
});

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = e.target.tagName;
  if (tag === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox') return;

  switch (e.key) {
    case ' ':
      if (tag === 'BUTTON') return;          // let the focused button handle it
      e.preventDefault();
      setPaused(!ui.paused);
      break;
    case '.': stepOnce(); break;
    case 'r': case 'R': restart(); break;
    case 'h': case 'H': setHidden(!ui.hidden); break;
    case 'f': case 'F': toggleFullscreen(); break;
    case 'Escape': if (ui.hidden) setHidden(false); break;
    case '1': case '2': case '3': case '4':
      if (tag !== 'INPUT') selectView(Number(e.key) - 1);
      break;
    default: return;
  }
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) field.stop();
  else if (!ui.paused && !reducedMotion.matches) field.start();
});

/* -- parameters -------------------------------------------------------- */

function paintRange(input) {
  const min = Number(input.min), max = Number(input.max);
  const pct = ((Number(input.value) - min) / (max - min)) * 100;
  input.style.setProperty('--pct', `${pct}%`);
}

function bind(id, format, apply) {
  const input = $(id);
  const out = $(`${id}Out`);
  const show = () => {
    out.textContent = format(Number(input.value));
    paintRange(input);
  };
  input.addEventListener('input', () => {
    show();
    apply(Number(input.value));
    updateStats();
  });
  show();
}

bind('tau', (v) => v.toFixed(3), (v) => field.setParams({ tau: v }));
bind('inflow', (v) => v.toFixed(3), (v) => field.setParams({ inflow: v }));
bind('radius', (v) => `${Math.round(v * 200)}%`,
  (v) => field.setParams({ radiusFraction: v }));
bind('contrast', (v) => `${v.toFixed(2)}×`, (v) => field.setParams({ contrast: v }));
bind('speed', (v) => `${v}`, (v) => field.setParams({ stepsPerFrame: v }));

// Resolution rebuilds the lattice, so apply it on release rather than on
// every tick of the drag.
{
  const input = $('rows');
  const show = () => { $('rowsOut').textContent = `${input.value} rows`; paintRange(input); };
  input.addEventListener('input', show);
  input.addEventListener('change', () => {
    field.opts.rows = Number(input.value);
    rebuild();
  });
  show();
}

$('sustained').addEventListener('change', (e) => {
  field.setParams({ sustained: e.target.checked });
});

/* -- visualization ----------------------------------------------------- */

const views = $('views');
views.innerHTML = COLORMAPS.map((m) =>
  `<button type="button" role="radio" data-id="${m.id}" aria-checked="false"
     title="${m.name} · ${m.id + 1}">${m.name.replace('Vorticity magnitude', '|Vorticity|')}</button>`
).join('');

function selectView(id) {
  const map = COLORMAPS[id];
  if (!map) return;
  field.setParams({ colormap: id });
  for (const btn of views.children) {
    btn.setAttribute('aria-checked', String(Number(btn.dataset.id) === id));
  }
  $('legendBar').style.background = map.gradient;
  $('legendLabels').innerHTML = map.labels.map((l) => `<span>${l}</span>`).join('');
  $('viewNote').textContent = map.note;
}

views.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-id]');
  if (btn) selectView(Number(btn.dataset.id));
});
views.addEventListener('keydown', (e) => {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
  e.preventDefault();
  const dir = e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 1;
  const next = (field.opts.colormap + dir + COLORMAPS.length) % COLORMAPS.length;
  selectView(next);
  views.children[next].focus();
});

/* -- dragging the obstacle --------------------------------------------- */

function toDomain(e) {
  return { x: e.clientX / window.innerWidth, y: 1 - e.clientY / window.innerHeight };
}

function overObstacle(e) {
  const { nx, ny } = field.state;
  const p = toDomain(e);
  const dx = p.x * nx - field.centre[0];
  const dy = p.y * ny - field.centre[1];
  const slop = Math.max(3, field.radius * 0.25);
  return dx * dx + dy * dy < (field.radius + slop) ** 2;
}

let drag = null;

canvas.addEventListener('pointerdown', (e) => {
  if (!overObstacle(e)) return;
  const p = toDomain(e);
  const o = field.obstacle();
  drag = { id: e.pointerId, dx: o.x - p.x, dy: o.y - p.y };
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add('grabbing');
});

canvas.addEventListener('pointermove', (e) => {
  if (drag && e.pointerId === drag.id) {
    const p = toDomain(e);
    field.setObstacle(p.x + drag.dx, p.y + drag.dy);
    return;
  }
  canvas.classList.toggle('can-grab', overObstacle(e));
});

const endDrag = (e) => {
  if (!drag || e.pointerId !== drag.id) return;
  drag = null;
  canvas.classList.remove('grabbing');
};
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

/* -- readout ----------------------------------------------------------- */

const fmt = new Intl.NumberFormat('en-US');
let frames = 0;
let fpsClock = performance.now();
let fps = 0;

function updateStats() {
  const { tau, inflow } = field.opts;
  const nu = (tau - 0.5) / 3;
  const re = (inflow * 2 * field.radius) / nu;
  $('statRe').textContent = fmt.format(Math.round(re));
  $('statMa').textContent = (inflow * Math.sqrt(3)).toFixed(3);
  $('statGrid').textContent = `${field.state.nx}×${field.state.ny}`;
  $('statSteps').textContent = fmt.format(field.steps);
  $('statFps').textContent = ui.paused ? '—' : String(Math.round(fps));
}

field.onStatus = (s) => {
  const spin = $('spinup');
  spin.hidden = !s.spinning;
  if (s.spinning) $('spinupFill').style.width = `${Math.round(s.spinProgress * 100)}%`;
  $('alert').hidden = !s.diverged;
};

function tick(now) {
  requestAnimationFrame(tick);
  if (field.running) frames++;
  const elapsed = now - fpsClock;
  if (elapsed >= 500) {
    fps = (frames * 1000) / elapsed;
    frames = 0;
    fpsClock = now;
    updateStats();
  }
}

/* -- start ------------------------------------------------------------- */

$('gpuName').textContent = field.renderer();
$('gpuName').title = field.renderer();

if (window.matchMedia('(max-width: 640px)').matches) $('btnCollapse').click();

fit();
field.reset({ width: window.innerWidth, height: window.innerHeight });
selectView(0);
setPaused(false);

if (reducedMotion.matches) {
  field.stop();
  field.renderStill();
  setPaused(true);
}

updateStats();
requestAnimationFrame(tick);
