/* Small SVG chart layer.
 *
 * No chart library: the dashboard is served off a localhost socket on a
 * machine that may well have no internet at all (a cluster node, a fresh
 * CUDA container), so a CDN tag would be a page that renders blank exactly
 * where it is most wanted.
 *
 * Colours are read from CSS custom properties rather than hard-coded here,
 * so the palette lives in one place and a theme swap moves the charts with
 * everything else.
 */

const NS = 'http://www.w3.org/2000/svg';

export function svg(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    node.setAttribute(key, String(value));
  }
  if (parent) parent.appendChild(node);
  return node;
}

let tokenCache = null;
export function invalidateTokens() { tokenCache = null; }

function tokens() {
  if (tokenCache) return tokenCache;
  const style = getComputedStyle(document.documentElement);
  const get = (name, fallback) =>
    (style.getPropertyValue(name) || fallback).trim();
  tokenCache = {
    surface: get('--chart-surface', '#1b1e26'),
    grid: get('--chart-grid', '#2c2c2a'),
    axis: get('--chart-axis', '#383835'),
    muted: get('--text-muted', '#898781'),
    secondary: get('--text-secondary', '#c3c2b7'),
    primary: get('--text-primary', '#ffffff'),
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => get(`--series-${i}`, '#3987e5')),
  };
  return tokenCache;
}

/* Backends keep one colour across the whole dashboard: identity follows the
 * entity, never its rank, so filtering the list never repaints the survivors.
 * numpy leads because every speedup on the page is quoted against it. */
const SLOT = { numpy: 0, 'torch-cpu': 1, 'torch-cuda': 2, cupy: 3 };

export function backendColor(name) {
  const palette = tokens().series;
  return palette[SLOT[name] ?? 4] || palette[0];
}

export function compact(value, digits = 1) {
  const n = Math.abs(value);
  if (!Number.isFinite(value)) return '—';
  if (n >= 1e9) return `${(value / 1e9).toFixed(digits)}B`;
  if (n >= 1e6) return `${(value / 1e6).toFixed(digits)}M`;
  if (n >= 1e3) return `${(value / 1e3).toFixed(digits)}k`;
  if (n >= 100) return value.toFixed(0);
  if (n >= 10) return value.toFixed(digits);
  return value.toFixed(Math.max(digits, 2));
}

function niceTicks(min, max, count = 5) {
  if (!(max > min)) return [min];
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) {
    ticks.push(Number(t.toFixed(10)));
  }
  return ticks;
}

function logTicks(min, max) {
  const ticks = [];
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  for (let e = lo; e <= hi; e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= min * 0.999 && v <= max * 1.001) ticks.push(v);
    }
  }
  return ticks.length >= 2 ? ticks : [min, max];
}

function scale(kind, domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (kind === 'log') {
    const l0 = Math.log10(Math.max(d0, 1e-12));
    const l1 = Math.log10(Math.max(d1, 1e-12));
    const span = l1 - l0 || 1;
    return (v) => r0 + ((Math.log10(Math.max(v, 1e-12)) - l0) / span) * (r1 - r0);
  }
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

/* ---------------------------------------------------------------------- */

class Chart {
  constructor(host) {
    this.host = host;
    this.host.classList.add('chart');
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    this.host.appendChild(this.tip);
    this.observer = new ResizeObserver(() => this.render());
    this.observer.observe(this.host);
  }

  clear() {
    this.host.querySelector('svg')?.remove();
    const width = Math.max(240, this.host.clientWidth);
    const height = Math.max(160, this.host.clientHeight || 280);
    const node = svg('svg', {
      width, height, viewBox: `0 0 ${width} ${height}`, role: 'img',
    }, this.host);
    this.host.insertBefore(node, this.tip);
    return { node, width, height };
  }

  showTip(html, x, y) {
    this.tip.innerHTML = html;
    this.tip.hidden = false;
    const box = this.host.getBoundingClientRect();
    const w = this.tip.offsetWidth;
    const left = Math.min(Math.max(8, x + 14), box.width - w - 8);
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${Math.max(8, y - 12)}px`;
  }

  hideTip() { this.tip.hidden = true; }

  destroy() { this.observer.disconnect(); }
}

/* ---------------------------------------------------------------------- */

/**
 * Multi-series line chart with a crosshair that reads every series at once.
 *
 * `series`: [{ name, color, points: [{x, y, meta}] }]
 */
export class LineChart extends Chart {
  constructor(host, options = {}) {
    super(host);
    this.options = {
      xScale: 'log', yScale: 'log', xLabel: '', yLabel: '',
      xFormat: compact, yFormat: compact, valueSuffix: '',
      ...options,
    };
    this.series = [];
    host.addEventListener('pointermove', (e) => this.hover(e));
    host.addEventListener('pointerleave', () => { this.hideTip(); this.moveCrosshair(null); });
  }

  setData(series, options = {}) {
    this.series = series.filter((s) => s.points.length);
    Object.assign(this.options, options);
    this.render();
  }

  render() {
    const t = tokens();
    const { node, width, height } = this.clear();
    this.geometry = null;
    if (!this.series.length) {
      svg('text', {
        x: width / 2, y: height / 2, 'text-anchor': 'middle',
        fill: t.muted, 'font-size': 13,
      }, node).textContent = 'no results yet';
      return;
    }

    // Room on the right for the end labels, and a band at the bottom deep
    // enough for the tick text plus the axis title -- a fixed height that
    // excludes the axis band is how these cards grow a nested scrollbar.
    const pad = { top: 18, right: 86, bottom: 46, left: 56 };
    const plotW = Math.max(40, width - pad.left - pad.right);
    const plotH = Math.max(40, height - pad.top - pad.bottom);

    const xs = this.series.flatMap((s) => s.points.map((p) => p.x));
    const ys = this.series.flatMap((s) => s.points.map((p) => p.y)).filter((v) => v > 0);
    if (!ys.length) return;

    const xDomain = [Math.min(...xs), Math.max(...xs)];
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);
    if (this.options.yScale === 'log') {
      yMin = 10 ** Math.floor(Math.log10(yMin));
      yMax = 10 ** Math.ceil(Math.log10(yMax));
    } else {
      yMin = Math.min(0, yMin);
      yMax = yMax * 1.08;
    }
    if (this.options.yFloor !== undefined) yMin = this.options.yFloor;

    const x = scale(this.options.xScale, xDomain, [pad.left, pad.left + plotW]);
    const y = scale(this.options.yScale, [yMin, yMax], [pad.top + plotH, pad.top]);

    const yTicks = this.options.yScale === 'log'
      ? logTicks(yMin, yMax) : niceTicks(yMin, yMax, 5);
    const xTicks = this.options.xScale === 'log'
      ? logTicks(xDomain[0], xDomain[1]) : niceTicks(xDomain[0], xDomain[1], 5);

    // Gridlines: solid hairlines one step off the surface, and nothing louder.
    for (const tick of yTicks) {
      const yy = y(tick);
      if (yy < pad.top - 1 || yy > pad.top + plotH + 1) continue;
      svg('line', {
        x1: pad.left, x2: pad.left + plotW, y1: yy, y2: yy,
        stroke: t.grid, 'stroke-width': 1,
      }, node);
      svg('text', {
        x: pad.left - 10, y: yy + 4, 'text-anchor': 'end',
        fill: t.muted, 'font-size': 11, class: 'tick',
      }, node).textContent = this.options.yFormat(tick);
    }

    for (const tick of xTicks) {
      const xx = x(tick);
      if (xx < pad.left - 1 || xx > pad.left + plotW + 1) continue;
      svg('text', {
        x: xx, y: pad.top + plotH + 20, 'text-anchor': 'middle',
        fill: t.muted, 'font-size': 11, class: 'tick',
      }, node).textContent = this.options.xFormat(tick);
    }

    svg('line', {
      x1: pad.left, x2: pad.left + plotW,
      y1: pad.top + plotH, y2: pad.top + plotH,
      stroke: t.axis, 'stroke-width': 1,
    }, node);

    if (this.options.xLabel) {
      svg('text', {
        x: pad.left + plotW / 2, y: height - 8, 'text-anchor': 'middle',
        fill: t.muted, 'font-size': 11,
      }, node).textContent = this.options.xLabel;
    }
    if (this.options.yLabel) {
      svg('text', {
        x: 14, y: pad.top + plotH / 2, 'text-anchor': 'middle',
        fill: t.muted, 'font-size': 11,
        transform: `rotate(-90 14 ${pad.top + plotH / 2})`,
      }, node).textContent = this.options.yLabel;
    }

    if (this.options.baseline !== undefined) {
      const yy = y(this.options.baseline);
      if (yy > pad.top && yy < pad.top + plotH) {
        svg('line', {
          x1: pad.left, x2: pad.left + plotW, y1: yy, y2: yy,
          stroke: t.secondary, 'stroke-width': 1, opacity: 0.55,
        }, node);
        svg('text', {
          x: pad.left + 6, y: yy - 6, fill: t.secondary, 'font-size': 10,
        }, node).textContent = this.options.baselineLabel || '';
      }
    }

    this.crosshair = svg('line', {
      y1: pad.top, y2: pad.top + plotH, stroke: t.secondary,
      'stroke-width': 1, opacity: 0, 'pointer-events': 'none',
    }, node);

    const ends = [];
    for (const s of this.series) {
      const pts = [...s.points].sort((a, b) => a.x - b.x);
      const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.x).toFixed(2)} ${y(p.y).toFixed(2)}`);
      svg('path', {
        d: path.join(' '), fill: 'none', stroke: s.color,
        'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      }, node);

      for (const p of pts) {
        // A 2px ring in the surface colour keeps a dot legible where two
        // series cross -- a stroke around the mark would add ink that is
        // not data.
        svg('circle', {
          cx: x(p.x), cy: y(p.y), r: 4.5, fill: s.color,
          stroke: t.surface, 'stroke-width': 2,
        }, node);
      }
      const last = pts[pts.length - 1];
      ends.push({ series: s, x: x(last.x), y: y(last.y) });
    }

    // Direct end labels, nudged apart only enough to stop an overlap; past
    // four series they would converge into noise, and the legend carries it.
    if (ends.length <= 4) {
      ends.sort((a, b) => a.y - b.y);
      let previous = -Infinity;
      for (const end of ends) {
        const yy = Math.max(end.y, previous + 14);
        previous = yy;
        const label = svg('text', {
          x: end.x + 10, y: yy + 4, fill: t.secondary, 'font-size': 11,
          class: 'end-label',
        }, node);
        label.textContent = end.series.name;
        if (Math.abs(yy - end.y) > 2) {
          svg('line', {
            x1: end.x + 5, y1: end.y, x2: end.x + 8, y2: yy,
            stroke: t.axis, 'stroke-width': 1,
          }, node);
        }
      }
    }

    this.geometry = { x, y, pad, plotW, plotH, xDomain };
  }

  moveCrosshair(px) {
    if (!this.crosshair) return;
    if (px === null) { this.crosshair.setAttribute('opacity', 0); return; }
    this.crosshair.setAttribute('x1', px);
    this.crosshair.setAttribute('x2', px);
    this.crosshair.setAttribute('opacity', 0.35);
  }

  hover(event) {
    if (!this.geometry) return;
    const box = this.host.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    const { x, pad, plotW } = this.geometry;
    if (px < pad.left - 12 || px > pad.left + plotW + 12) {
      this.hideTip(); this.moveCrosshair(null); return;
    }

    // Snap to the nearest x present in the data, so the tooltip always
    // quotes a measurement rather than an interpolation.
    const allX = [...new Set(this.series.flatMap((s) => s.points.map((p) => p.x)))];
    let best = allX[0];
    for (const candidate of allX) {
      if (Math.abs(x(candidate) - px) < Math.abs(x(best) - px)) best = candidate;
    }
    this.moveCrosshair(x(best));

    const rows = this.series.map((s) => {
      const point = s.points.find((p) => p.x === best);
      if (!point) return '';
      return `<div class="tip-row">
        <span class="swatch" style="background:${s.color}"></span>
        <span class="tip-name">${s.name}</span>
        <span class="tip-value">${this.options.yFormat(point.y)}${this.options.valueSuffix}</span>
      </div>`;
    }).join('');
    const title = this.options.tipTitle
      ? this.options.tipTitle(best)
      : this.options.xFormat(best);
    this.showTip(`<div class="tip-title">${title}</div>${rows}`, px, py);
  }
}

/* ---------------------------------------------------------------------- */

/** Horizontal bars, value at the tip, one fixed colour per backend. */
export class BarChart extends Chart {
  constructor(host, options = {}) {
    super(host);
    this.options = { format: compact, suffix: '', ...options };
    this.items = [];
  }

  setData(items, options = {}) {
    this.items = items;
    Object.assign(this.options, options);
    this.render();
  }

  render() {
    const t = tokens();
    const { node, width, height } = this.clear();
    if (!this.items.length) {
      svg('text', {
        x: width / 2, y: height / 2, 'text-anchor': 'middle',
        fill: t.muted, 'font-size': 13,
      }, node).textContent = 'no results yet';
      return;
    }

    const labelW = 92;
    const valueW = 76;
    const pad = { top: 10, right: valueW, bottom: 10, left: labelW };
    const plotW = Math.max(30, width - pad.left - pad.right);
    const rows = this.items.length;
    const band = (height - pad.top - pad.bottom) / rows;
    // Capped rather than filling the band: the leftover is the air that keeps
    // a bar chart from reading as a solid block.
    const thickness = Math.min(24, Math.max(8, band - 12));

    const max = Math.max(...this.items.map((i) => i.value), 0);
    const x = scale('linear', [0, max || 1], [pad.left, pad.left + plotW]);

    this.items.forEach((item, index) => {
      const cy = pad.top + band * index + band / 2;
      const end = x(item.value);
      const radius = Math.min(4, thickness / 2);

      svg('text', {
        x: pad.left - 12, y: cy + 4, 'text-anchor': 'end',
        fill: t.secondary, 'font-size': 12,
      }, node).textContent = item.label;

      // Square at the baseline, 4px rounded at the data end.
      const top = cy - thickness / 2;
      const w = Math.max(radius, end - pad.left);
      svg('path', {
        d: `M${pad.left} ${top} H${pad.left + w - radius} a${radius} ${radius} 0 0 1 ${radius} ${radius}
            V${top + thickness - radius} a${radius} ${radius} 0 0 1 ${-radius} ${radius}
            H${pad.left} Z`,
        fill: item.color,
      }, node);

      svg('text', {
        x: end + 10, y: cy + 4, fill: t.primary, 'font-size': 12,
        'font-weight': 600,
      }, node).textContent = this.options.format(item.value) + this.options.suffix;

      if (item.note) {
        svg('text', {
          x: end + 10, y: cy + 17, fill: t.muted, 'font-size': 10,
        }, node).textContent = item.note;
      }

      const hit = svg('rect', {
        x: pad.left, y: top - 6, width: plotW + valueW, height: thickness + 12,
        fill: 'transparent',
      }, node);
      hit.addEventListener('pointermove', (e) => {
        const box = this.host.getBoundingClientRect();
        this.showTip(
          `<div class="tip-title">${item.label}</div>
           <div class="tip-row"><span class="swatch" style="background:${item.color}"></span>
           <span class="tip-name">${this.options.measure || 'value'}</span>
           <span class="tip-value">${this.options.format(item.value)}${this.options.suffix}</span></div>
           ${item.tip || ''}`,
          e.clientX - box.left, e.clientY - box.top);
      });
      hit.addEventListener('pointerleave', () => this.hideTip());
    });

    svg('line', {
      x1: pad.left, x2: pad.left, y1: pad.top, y2: height - pad.bottom,
      stroke: t.axis, 'stroke-width': 1,
    }, node);
  }
}

/** A legend is always present for two or more series. */
export function legend(host, entries) {
  host.innerHTML = '';
  if (entries.length < 2) { host.hidden = true; return; }
  host.hidden = false;
  for (const entry of entries) {
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.innerHTML = `<span class="swatch" style="background:${entry.color}"></span>${entry.name}`;
    host.appendChild(item);
  }
}
