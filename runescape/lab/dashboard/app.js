'use strict';
// GE Strategy Lab dashboard: reads a run's files from lab/serve.py and draws them as hand-rolled SVG.
// Colors are CSS custom properties (index.html), applied through style so the theme toggle needs no redraw.

const $ = id => document.getElementById(id);
const SVGNS = 'http://www.w3.org/2000/svg';
const POLL_MS = 5000;
const MAX_EVAL_CURVES = 40;

function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function div(cls, text, parent) {
  const d = document.createElement('div');
  if (cls) d.className = cls;
  if (text != null) d.textContent = text;
  if (parent) parent.appendChild(d);
  return d;
}

// --- formatting -------------------------------------------------------------------------------------------------

const weekly = g => Math.expm1(g) * 100;  // log growth per week -> % per week
const fmtPct = (v, digits = 1) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(digits) + '%';
function fmtGp(n) {
  const a = Math.abs(n), s = n < 0 ? '−' : '';
  if (a >= 1e9) return s + +(a / 1e9).toFixed(2) + 'b';
  if (a >= 1e6) return s + +(a / 1e6).toFixed(1) + 'm';
  if (a >= 1e3) return s + +(a / 1e3).toFixed(0) + 'k';
  return s + Math.round(a);
}
const fmtDate = ts => new Date(ts * 1000).toLocaleDateString([], { month: 'short', day: 'numeric' });
const fmtDayTime = ts => new Date(ts * 1000).toLocaleString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' });
function fmtDuration(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m ${Math.floor(s % 60)}s`;
}

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw)), r = raw / mag;
  const step = (r >= 7.5 ? 10 : r >= 3.5 ? 5 : r >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function nearest(points, x) {  // index of the point (sorted by x) closest to x
  let lo = 0, hi = points.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (points[mid][0] < x) lo = mid; else hi = mid; }
  return Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? lo : hi;
}

// --- line chart ---------------------------------------------------------------------------------------------------
// o: {height, series: [{name, color, points: [[x, y]], width, opacity, dots, hover}], bands: [{lo, hi, color,
//     opacity}], refs: [{y, label}], markers: [{x, y, shape: 'up'|'down', color, tip}], xFmt, yFmt, xTickFmt,
//     xTicks, yTickFmt, compact, yIncludeZero, connectGaps (draw lines across missing points), tipExtra(x)}

function linePlot(box, o) {
  const W = Math.max(240, box.clientWidth || 600), H = o.height || 240;
  const m = { l: o.compact ? 46 : 54, r: 14, t: 10, b: o.compact ? 20 : 26 };
  const xs = [], ys = [];
  for (const s of o.series) for (const [x, y] of s.points) { xs.push(x); if (y != null) ys.push(y); }
  for (const b of o.bands || []) for (const pts of [b.lo, b.hi]) for (const [x, y] of pts) { xs.push(x); ys.push(y); }
  for (const r of o.refs || []) ys.push(r.y);
  for (const mk of o.markers || []) { xs.push(mk.x); ys.push(mk.y); }
  if (o.yIncludeZero) ys.push(0);
  box.replaceChildren();
  if (!xs.length || !ys.length) { div('empty', 'No data yet', box); return; }
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (o.xMin != null) x0 = o.xMin;
  if (o.xMax != null) x1 = o.xMax;
  if (x1 === x0) x1 = x0 + 1;
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0 || Math.abs(y1) || 1) * 0.06;
  y0 -= pad; y1 += pad;
  const yt = niceTicks(y0, y1, o.compact ? 3 : 5);
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  const X = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = y => m.t + (1 - (y - y0) / (y1 - y0)) * (H - m.t - m.b);

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, tabindex: 0, role: 'img',
                          'aria-label': o.label || 'chart' });
  for (const v of yt) {
    el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), style: 'stroke:var(--grid);stroke-width:1' }, svg);
    el('text', { x: m.l - 6, y: Y(v) + 3.5, 'text-anchor': 'end' }, svg).textContent = (o.yTickFmt || o.yFmt || String)(v);
  }
  const xt = o.xTicks || niceTicks(x0, x1, o.compact ? 3 : 6);
  for (const v of xt) {
    if (v < x0 || v > x1) continue;
    el('text', { x: X(v), y: H - 6, 'text-anchor': 'middle' }, svg).textContent = (o.xTickFmt || o.xFmt || String)(v);
  }
  el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, style: 'stroke:var(--axis);stroke-width:1' }, svg);

  for (const b of o.bands || []) {
    if (!b.lo.length) continue;
    const d = b.hi.map(([x, y], i) => `${i ? 'L' : 'M'}${X(x)},${Y(y)}`).join('') +
      b.lo.slice().reverse().map(([x, y]) => `L${X(x)},${Y(y)}`).join('') + 'Z';
    el('path', { d, style: `fill:${b.color};fill-opacity:${b.opacity};stroke:none` }, svg);
  }
  for (const r of o.refs || []) {
    el('line', { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), style: 'stroke:var(--text-muted);stroke-width:1' }, svg);
    if (r.label) el('text', { x: W - m.r, y: Y(r.y) - 5, 'text-anchor': 'end', class: 'label' }, svg).textContent = r.label;
  }
  for (const s of o.series) {
    let d = '', pen = false;
    for (const [x, y] of s.points) {
      if (y == null) { if (!o.connectGaps) pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
      pen = true;
    }
    el('path', { d, style: `fill:none;stroke:${s.color};stroke-width:${s.width || 2};stroke-opacity:${s.opacity ?? 1};` +
                 'stroke-linejoin:round;stroke-linecap:round' }, svg);
    for (const [x, y, big] of s.dots ? s.points : []) {
      if (y == null) continue;
      el('circle', { cx: X(x), cy: Y(y), r: big ? 5 : 4,
                     style: `fill:${s.color};stroke:var(--surface-1);stroke-width:2` }, svg);
    }
  }
  for (const mk of o.markers || []) {
    const cx = X(mk.x), cy = Y(mk.y), r = 5;
    const d = mk.shape === 'down' ? `M${cx - r},${cy - r * 0.8}L${cx + r},${cy - r * 0.8}L${cx},${cy + r}Z`
                                  : `M${cx - r},${cy + r * 0.8}L${cx + r},${cy + r * 0.8}L${cx},${cy - r}Z`;
    el('path', { d, style: `fill:${mk.color};stroke:var(--surface-1);stroke-width:2;stroke-linejoin:round` }, svg);
  }

  // hover: a crosshair that snaps to the nearest x, and one tooltip listing every series there
  const hoverSeries = o.series.filter(s => s.hover !== false && s.points.length);
  const tip = div('tooltip', null, null);
  box.append(svg, tip);
  if (o.tooltip === false || !hoverSeries.length) return;
  const allX = [...new Set(hoverSeries.flatMap(s => s.points.map(p => p[0])))].sort((a, b) => a - b);
  const cross = el('line', { y1: m.t, y2: H - m.b, style: 'stroke:var(--axis);stroke-width:1;display:none' }, svg);
  const ring = [];
  let idx = -1;
  const show = i => {
    idx = Math.max(0, Math.min(allX.length - 1, i));
    const x = allX[idx];
    cross.setAttribute('x1', X(x)); cross.setAttribute('x2', X(x)); cross.style.display = '';
    tip.replaceChildren();
    div('head', (o.xFmt || String)(x), tip);
    ring.forEach(c => c.remove()); ring.length = 0;
    for (const s of hoverSeries) {
      const k = nearest(s.points, x);
      if (k < 0 || s.points[k][1] == null) continue;
      if (o.snapTolerance != null && Math.abs(s.points[k][0] - x) > o.snapTolerance) continue;
      const row = div('row', null, tip);
      const key = el('svg', { width: 14, height: 4 });
      el('line', { x1: 0, x2: 14, y1: 2, y2: 2, style: `stroke:${s.color};stroke-width:2;stroke-opacity:${s.opacity ?? 1}` }, key);
      const strong = document.createElement('strong'); strong.textContent = (o.yFmt || String)(s.points[k][1]);
      const name = document.createElement('span'); name.textContent = s.name;
      row.append(key, strong, name);
      ring.push(el('circle', { cx: X(s.points[k][0]), cy: Y(s.points[k][1]), r: 4,
                               style: `fill:${s.color};stroke:var(--surface-1);stroke-width:2` }, svg));
    }
    for (const line of o.tipExtra ? o.tipExtra(x) : []) div('row', line, tip).style.color = 'var(--text-secondary)';
    tip.style.display = 'block';
    const left = X(x) + 14, w = tip.offsetWidth;
    tip.style.left = `${left + w > W ? X(x) - w - 14 : left}px`;
    tip.style.top = `${m.t}px`;
  };
  const hide = () => { cross.style.display = 'none'; tip.style.display = 'none'; ring.forEach(c => c.remove()); };
  svg.addEventListener('pointermove', ev => {
    const rect = svg.getBoundingClientRect();
    const x = x0 + ((ev.clientX - rect.left) * W / rect.width - m.l) / (W - m.l - m.r) * (x1 - x0);
    let best = 0;
    for (let i = 1; i < allX.length; i++) if (Math.abs(allX[i] - x) < Math.abs(allX[best] - x)) best = i;
    show(best);
  });
  svg.addEventListener('pointerleave', hide);
  svg.addEventListener('blur', hide);
  svg.addEventListener('keydown', ev => {
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') {
      show(idx < 0 ? allX.length - 1 : idx + (ev.key === 'ArrowRight' ? 1 : -1));
      ev.preventDefault();
    }
  });
}

function legend(box, items) {
  box.replaceChildren();
  for (const it of items) {
    const k = document.createElement('span'); k.className = 'key';
    const sw = el('svg', { width: it.shape ? 12 : 16, height: 12 });
    if (it.shape === 'up') el('path', { d: 'M1,11L11,11L6,1Z', style: `fill:${it.color}` }, sw);
    else if (it.shape === 'down') el('path', { d: 'M1,1L11,1L6,11Z', style: `fill:${it.color}` }, sw);
    else if (it.shape === 'band') el('rect', { x: 0, y: 2, width: 16, height: 8, rx: 2, style: `fill:${it.color};fill-opacity:0.25` }, sw);
    else el('line', { x1: 0, x2: 16, y1: 6, y2: 6, style: `stroke:${it.color};stroke-width:2;stroke-opacity:${it.opacity ?? 1}` }, sw);
    const t = document.createElement('span'); t.textContent = it.name;
    k.append(sw, t);
    box.appendChild(k);
  }
}

function table(box, headers, rows) {
  const t = document.createElement('table');
  const tr = t.createTHead().insertRow();
  headers.forEach(([h, num]) => { const th = document.createElement('th'); th.textContent = h; if (num) th.className = 'num'; tr.appendChild(th); });
  const body = t.createTBody();
  for (const r of rows) {
    const row = body.insertRow();
    r.forEach((v, i) => { const td = row.insertCell(); td.textContent = v; if (headers[i][1]) td.className = 'num'; });
  }
  box.replaceChildren(t);
  return t;
}

// --- heatmap --------------------------------------------------------------------------------------------------------

function heatmap(box, rows, cols, values, tipText) {
  box.replaceChildren();
  if (!rows.length) { div('empty', 'No champion yet', box); return; }
  const W = Math.max(240, box.clientWidth || 600), cell = 18, gap = 2, m = { l: 64, t: 4, b: 34 };
  const cw = (W - m.l - 8) / cols.length;
  const H = m.t + rows.length * (cell + gap) + m.b;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': 'buy volume by price tier' });
  const tip = div('tooltip');
  const steps = 5;  // validated ordinal ramp (index.html --seq-1..5)
  rows.forEach((label, r) => {
    const y = m.t + r * (cell + gap);
    el('text', { x: m.l - 6, y: y + cell / 2 + 4, 'text-anchor': 'end' }, svg).textContent = label;
    cols.forEach((_, c) => {
      const v = values[r][c];
      const fill = v <= 0 ? 'var(--grid)' : `var(--seq-${Math.min(steps, Math.max(1, Math.ceil(v * steps)))})`;
      const rect = el('rect', { x: m.l + c * cw + gap / 2, y, width: Math.max(1, cw - gap), height: cell, rx: 3,
                                style: `fill:${fill}` }, svg);
      rect.addEventListener('pointermove', ev => {
        tip.replaceChildren();
        tipText(r, c).forEach((line, i) => {
          if (i === 0) { const s = document.createElement('strong'); s.textContent = line; div('row', null, tip).appendChild(s); }
          else div('row', line, tip).style.color = 'var(--text-secondary)';
        });
        tip.style.display = 'block';
        const rect2 = box.getBoundingClientRect();
        tip.style.left = `${Math.min(ev.clientX - rect2.left + 12, rect2.width - tip.offsetWidth)}px`;
        tip.style.top = `${ev.clientY - rect2.top + 12}px`;
        rect.style.opacity = '0.8';
      });
      rect.addEventListener('pointerleave', () => { tip.style.display = 'none'; rect.style.opacity = ''; });
    });
  });
  const yb = m.t + rows.length * (cell + gap) + 14;
  cols.forEach((label, c) => {
    if (label) el('text', { x: m.l + c * cw + cw / 2, y: yb, 'text-anchor': 'middle' }, svg).textContent = label;
  });
  // scale legend
  const lx = m.l, ly = yb + 8;
  el('text', { x: lx - 6, y: ly + 9, 'text-anchor': 'end' }, svg).textContent = 'share';
  for (let s = 0; s <= steps; s++) {
    el('rect', { x: lx + s * 22, y: ly, width: 20, height: 10, rx: 2,
                 style: `fill:${s ? `var(--seq-${s})` : 'var(--grid)'}` }, svg);
  }
  el('text', { x: lx + (steps + 1) * 22 + 4, y: ly + 9 }, svg).textContent = '0 → 100% of the row';
  box.append(svg, tip);
}

// --- data -----------------------------------------------------------------------------------------------------------

const state = { api: null, run: null, gens: [], champion: null, champions: [], evals: new Map(), timer: null };

async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  return r.ok ? r.json() : null;
}
async function getText(url) {
  const r = await fetch(url, { cache: 'no-store' });
  return r.ok ? r.text() : '';
}
const jsonl = text => text.split('\n').filter(Boolean).map(line => JSON.parse(line));

async function refresh() {
  $('content').classList.add('stale');
  try {
    state.api = await getJSON('/api/runs');
    const runs = state.api?.runs || [];
    const sel = $('run-select');
    const wanted = decodeURIComponent(location.hash.slice(1)) || sel.value || runs[0]?.run;
    sel.replaceChildren(...runs.map(r => {
      const opt = document.createElement('option');
      opt.value = r.run;
      opt.textContent = `${r.run}${r.summary ? `  (${fmtPct(weekly(r.summary.best_val))}/wk)` : ''}`;
      return opt;
    }));
    state.run = runs.find(r => r.run === wanted) || runs[0] || null;
    if (state.run) sel.value = state.run.run;
    if (!state.run) { $('run-sub').textContent = 'No runs yet: start one with uv run python -m lab.evolve'; return; }
    const base = `/runs/${encodeURIComponent(state.run.run)}`;
    const [gensText, champion, champsText] = await Promise.all([
      getText(`${base}/gens.jsonl`), getJSON(`${base}/champion.json`), getText(`${base}/champions.jsonl`)]);
    state.gens = jsonl(gensText);
    state.champion = champion;
    state.champions = jsonl(champsText);
    const wantedEvals = state.run.evals.slice(-MAX_EVAL_CURVES);
    await Promise.all(wantedEvals.filter(f => !state.evals.has(`${state.run.run}/${f}`)).map(async f => {
      const e = await getJSON(`${base}/evals/${f}`);
      if (e) state.evals.set(`${state.run.run}/${f}`, e);
    }));
    render();
    $('updated').textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    $('updated').textContent = `Couldn't load: ${err.message}`;
  } finally {
    $('content').classList.remove('stale');
  }
}

// --- panels ---------------------------------------------------------------------------------------------------------

function render() {
  const run = state.run, gens = state.gens, cfg = run.config, sum = run.summary || {};
  const sp = cfg.splits;
  $('run-sub').textContent = `${cfg.policy}${cfg.args.name ? ` "${cfg.args.name}"` : ''} · ${cfg.dim} parameters · ` +
    `${2 * cfg.args.pairs + 1} variants × ${cfg.args.episodes} weeks per generation · trains ${fmtDate(sp.train.start)}–` +
    `${fmtDate(sp.train.end)}, validates ${fmtDate(sp.validation.start)}–${fmtDate(sp.validation.end)} · ` +
    `fills ${cfg.args.fill_model}`;
  renderKpis(run, gens, sum);
  renderFitness(gens);
  renderValidation(gens, run.baseline);
  renderEquity(run);
  renderTiers();
  renderReplay();
  renderParams(gens, cfg);
  renderBoard();
}

function renderKpis(run, gens, sum) {
  const rules = run.baseline ? weekly(run.baseline.score) : null;
  if (sum.best_val != null && isFinite(sum.best_val)) {
    const champ = weekly(sum.best_val);
    $('kpi-champion').textContent = `${fmtPct(champ)}/wk`;
    const d = $('kpi-champion-delta');
    if (rules != null) {
      const diff = champ - rules;
      d.textContent = `${diff >= 0 ? '▲' : '▼'} ${Math.abs(diff).toFixed(1)} pts vs rules · gen ${sum.best_gen}`;
      d.className = `delta ${diff >= 0 ? 'up' : 'down'}`;
    }
  } else { $('kpi-champion').textContent = '–'; }
  $('kpi-rules').textContent = rules != null ? `${fmtPct(rules)}/wk` : '–';
  $('kpi-rules-sub').textContent = run.baseline ?
    `worst week ${fmtPct(weekly(run.baseline.worst))}, max drawdown ${(run.baseline.max_drawdown * 100).toFixed(1)}%` : '';
  const last = gens[gens.length - 1];
  $('kpi-gens').textContent = gens.length ? `${gens.length.toLocaleString()} / ${run.config.args.gens.toLocaleString()}` : '0';
  $('kpi-gens-sub').textContent = last ? `${last.sec.toFixed(1)} s each · running ${fmtDuration(last.time)}` : '';
  $('kpi-holdout').textContent = String(state.api.holdout_looks);
}

function renderFitness(gens) {
  const q = k => gens.map(g => [g.gen, weekly(g.fit_q[k])]);
  legend($('fit-legend'), [{ name: 'Population median', color: 'var(--series-1)' },
                           { name: 'Middle 50% / 80%', color: 'var(--series-1)', shape: 'band' }]);
  linePlot($('fit-chart'), {
    label: 'training fitness by generation',
    series: [{ name: 'median', color: 'var(--series-1)', points: q(2) }],
    bands: [{ lo: q(0), hi: q(4), color: 'var(--series-1)', opacity: 0.10 },
            { lo: q(1), hi: q(3), color: 'var(--series-1)', opacity: 0.16 }],
    xFmt: x => `gen ${x}`, xTickFmt: x => String(x), yFmt: v => fmtPct(v),
    tipExtra: x => {
      const g = gens.find(r => r.gen === x);
      return g ? [`middle 50%: ${fmtPct(weekly(g.fit_q[1]))} to ${fmtPct(weekly(g.fit_q[3]))}`,
                  `middle 80%: ${fmtPct(weekly(g.fit_q[0]))} to ${fmtPct(weekly(g.fit_q[4]))}`,
                  `best variant: ${fmtPct(weekly(g.fit_max))}`] : [];
    },
  });
  const rows = gens.filter((g, i) => i % 10 === 0 || i === gens.length - 1).map(g =>
    [g.gen, fmtPct(weekly(g.fit_q[0])), fmtPct(weekly(g.fit_q[2])), fmtPct(weekly(g.fit_q[4])), fmtPct(weekly(g.fit_max))]);
  table($('fit-table'), [['Gen', true], ['10th pct', true], ['Median', true], ['90th pct', true], ['Best', true]], rows);
}

function renderValidation(gens, baseline) {
  const smooth = [];
  for (let i = 0; i < gens.length; i++) {
    const w = gens.slice(Math.max(0, i - 9), i + 1);
    smooth.push([gens[i].gen, weekly(w.reduce((a, g) => a + g.center_mean, 0) / w.length)]);
  }
  const val = gens.filter(g => g.val).map(g => [g.gen, weekly(g.val.score), !!g.champion]);
  const unseen = gens.filter(g => g.val_unseen).map(g => [g.gen, weekly(g.val_unseen.score)]);
  const series = [{ name: 'training', color: 'var(--series-1)', points: smooth },
                  { name: 'validation', color: 'var(--series-2)', points: val, dots: true }];
  const keys = [{ name: 'Training weeks', color: 'var(--series-1)' },
                { name: unseen.length ? 'Validation weeks, training items' : 'Validation weeks', color: 'var(--series-2)' }];
  if (unseen.length) {
    series.push({ name: 'validation, unseen items', color: 'var(--series-3)', points: unseen });
    keys.push({ name: 'Validation weeks, unseen items', color: 'var(--series-3)' });
  }
  legend($('val-legend'), [...keys, { name: 'Rules on validation', color: 'var(--text-muted)' }]);
  linePlot($('val-chart'), {
    label: 'training vs validation by generation', series,
    refs: baseline ? [{ y: weekly(baseline.score), label: `rules ${fmtPct(weekly(baseline.score))}` }] : [],
    xFmt: x => `gen ${x}`, xTickFmt: x => String(x), yFmt: v => fmtPct(v),
  });
  const head = [['Gen', true], ['Validation', true], ['Worst week', true], ['Worst quarter', true]];
  if (unseen.length) head.push(['Unseen items', true]);
  table($('val-table'), [...head, ['Champion', false]], gens.filter(g => g.val).map(g => [
    g.gen, fmtPct(weekly(g.val.score)), fmtPct(weekly(g.val.worst)), fmtPct(weekly(g.val.cvar25)),
    ...(unseen.length ? [g.val_unseen ? fmtPct(weekly(g.val_unseen.score)) : ''] : []), g.champion ? 'new champion' : '']));
}

function meanCurve(curves) {
  const n = Math.min(...curves.map(c => c.length));
  const out = [];
  for (let k = 0; k < n; k++) out.push([k / (n - 1) * 7, (curves.reduce((a, c) => a + c[k], 0) / curves.length - 1) * 100]);
  return out;
}

function renderEquity(run) {
  const evals = run.evals.slice(-MAX_EVAL_CURVES).map(f => state.evals.get(`${run.run}/${f}`)).filter(Boolean);
  const champGen = run.summary?.best_gen;
  const series = evals.map((e, i) => ({
    name: `gen ${e.gen}`, color: 'var(--history)', points: meanCurve(e.curves), width: 1.5,
    opacity: 0.12 + 0.5 * (i + 1) / evals.length, hover: false,
  }));
  const champ = evals.find(e => e.gen === champGen);
  if (champ) series.push({ name: `champion (gen ${champ.gen})`, color: 'var(--series-1)', points: meanCurve(champ.curves) });
  const latest = evals[evals.length - 1];
  if (latest && latest !== champ) series.push({ name: `latest (gen ${latest.gen})`, color: 'var(--series-2)', points: meanCurve(latest.curves) });
  legend($('eq-legend'), [{ name: 'Champion', color: 'var(--series-1)' }, { name: 'Latest evaluation', color: 'var(--series-2)' },
                          { name: 'Earlier evaluations', color: 'var(--history)', opacity: 0.5 }]);
  linePlot($('eq-chart'), {
    label: 'champion equity through the validation weeks', series,
    xFmt: x => `day ${x.toFixed(1)}`, xTickFmt: x => `day ${x}`, xTicks: [0, 1, 2, 3, 4, 5, 6, 7],
    yFmt: v => fmtPct(v), yIncludeZero: true,
  });
}

const TIER_LABELS = Array.from({ length: 20 }, (_, i) => ({ 0: '1m', 4: '10m', 8: '100m', 12: '1b', 16: '10b' }[i] || ''));
function tierRange(i) { return `${fmtGp(10 ** (6 + i / 4))}–${fmtGp(10 ** (6 + (i + 1) / 4))}`; }

function renderTiers() {
  const champs = state.champions.slice(-16);
  const values = champs.map(c => { const t = c.price_tiers.reduce((a, b) => a + b, 0) || 1; return c.price_tiers.map(v => v / t); });
  heatmap($('tier-chart'), champs.map(c => `gen ${c.gen}`), TIER_LABELS, values, (r, c) => [
    `${(values[r][c] * 100).toFixed(1)}% of gp spent on buys`, `items priced ${tierRange(c)}`,
    `champion from gen ${champs[r].gen} (${fmtPct(weekly(champs[r].score))}/wk)`]);
}

function renderReplay() {
  const box = $('replay');
  box.replaceChildren();
  const rep = state.champion?.replay;
  legend($('replay-legend'), [{ name: 'Instasell price', color: 'var(--series-1)' }, { name: 'Instabuy price', color: 'var(--series-2)' },
                              { name: 'Buy fill', color: 'var(--series-1)', shape: 'up' },
                              { name: 'Sell fill', color: 'var(--series-2)', shape: 'down' }]);
  if (!rep || !rep.items.length) { div('empty', 'No champion trades yet', box); return; }
  $('replay-desc').textContent = `Champion from gen ${state.champion.gen}, its median validation week (from ` +
    `${fmtDate(rep.episode_start)}) at 1.8b: ${fmtPct(weekly(rep.growth))} that week. Its most-traded items, with ` +
    'the instasell and instabuy prices (30-minute averages) and every fill.';
  for (const item of rep.items) {
    const tile = div(null, null, box);
    let pnl = 0, units = 0;
    for (const [, side, qty, price] of item.fills) { pnl += (side ? 1 : -1) * qty * price; if (!side) units += qty; }
    div('tile-title', item.name, tile);
    div('tile-sub', `${units} bought · ${item.fills.length} fills · net cash ${fmtGp(pnl)} (before tax)`, tile);
    const chart = div('chart', null, tile);
    const pts = key => item[key].map((v, k) => [item.t0 + k * item.step, v]);
    linePlot(chart, {
      compact: true, height: 150, label: `${item.name} prices and fills`, connectGaps: true,
      series: [{ name: 'instasell', color: 'var(--series-1)', points: pts('instasell'), width: 1.5 },
               { name: 'instabuy', color: 'var(--series-2)', points: pts('instabuy'), width: 1.5 }],
      markers: item.fills.map(([ts, side, qty, price]) => ({ x: ts, y: price, shape: side ? 'down' : 'up',
                                                             color: side ? 'var(--series-2)' : 'var(--series-1)' })),
      xFmt: fmtDayTime, xTickFmt: x => new Date(x * 1000).toLocaleDateString([], { weekday: 'short' }),
      xTicks: Array.from({ length: 8 }, (_, d) => item.t0 + d * 86400), yFmt: fmtGp,
      tipExtra: x => item.fills.filter(f => Math.abs(f[0] - x) <= item.step / 2)
        .map(([, side, qty, price]) => `${side ? 'sold' : 'bought'} ${qty} @ ${fmtGp(price)}`),
    });
  }
}

const PARAM_TICK_FMT = {
  buy_outbid_spread: v => `${(v * 100).toFixed(1)}%`, sell_outbid_spread: v => `${(v * 100).toFixed(1)}%`,
};
const PARAM_FMT = {
  max_quote_age_h: v => `${v.toFixed(2)} h`, min_trade_buckets: v => v.toFixed(1),
  max_drift: v => `${(v * 100).toFixed(1)}%`, max_volume_spike: v => `${v.toFixed(2)}×`,
  buy_outbid_spread: v => `${(v * 100).toFixed(1)}% of spread`, sell_outbid_spread: v => `${(v * 100).toFixed(1)}% of spread`,
  max_item_share: v => `${(v * 100).toFixed(0)}%`, min_margin_pct: v => `${(v * 100).toFixed(2)}%`,
  min_price: fmtGp, max_price_ratio: v => `${v.toFixed(1)}×`, min_4h_volume_gp: fmtGp,
};
const PARAM_NAMES = {
  max_quote_age_h: 'Max quote age', min_trade_buckets: 'Min trading buckets / 24h', max_drift: 'Max drift from 3-day median',
  max_volume_spike: 'Max volume spike', buy_outbid_spread: 'Buy outbid', sell_outbid_spread: 'Sell undercut',
  max_item_share: 'Max share of equity per item', min_margin_pct: 'Min margin', min_price: 'Min buy price',
  max_price_ratio: 'Max / min price', min_4h_volume_gp: 'Min 4h volume (gp)',
};

function renderParams(gens, cfg) {
  const card = $('params-card');
  const withValues = gens.filter(g => g.values);
  card.hidden = !withValues.length;
  if (!withValues.length) return;
  const box = $('params');
  box.replaceChildren();
  for (const key of Object.keys(withValues[0].values)) {
    const tile = div(null, null, box);
    const fmt = PARAM_FMT[key] || (v => v.toPrecision(3));
    const now = withValues[withValues.length - 1].values[key];
    div('tile-title', PARAM_NAMES[key] || key, tile);
    div('tile-sub', `now ${fmt(now)}${cfg.start_values ? ` · started ${fmt(cfg.start_values[key])}` : ''}`, tile);
    const chart = div('chart', null, tile);
    linePlot(chart, {
      compact: true, height: 110, label: PARAM_NAMES[key] || key,
      series: [{ name: PARAM_NAMES[key] || key, color: 'var(--series-1)', points: withValues.map(g => [g.gen, g.values[key]]) }],
      refs: cfg.start_values ? [{ y: cfg.start_values[key] }] : [],
      xFmt: x => `gen ${x}`, xTickFmt: x => String(x), yFmt: fmt, yTickFmt: PARAM_TICK_FMT[key] || fmt,
    });
  }
}

function renderBoard() {
  const runs = (state.api?.runs || []).filter(r => r.summary).slice()
    .sort((a, b) => (b.summary.best_val ?? -1e9) - (a.summary.best_val ?? -1e9));
  const box = $('board');
  if (!runs.length) { box.replaceChildren(); div('empty', 'No runs yet', box); return; }
  const t = table(box, [['Run', false], ['Policy', false], ['Gens', true], ['Validation', true], ['Rules', true],
                        ['vs rules', true], ['Trained through', false], ['Forward', true], ['Forward rules', true]],
    runs.map(r => {
      const s = r.summary, f = r.scores?.forward;
      return [r.run, r.config.policy, s.gens, fmtPct(weekly(s.best_val)), fmtPct(weekly(s.baseline_val)),
              `${(weekly(s.best_val) - weekly(s.baseline_val)).toFixed(1)} pts`, fmtDate(s.trained_through),
              f ? `${fmtPct(weekly(f.score))} (${f.days.toFixed(0)}d)` : '–', f?.rules != null ? fmtPct(weekly(f.rules)) : '–'];
    }));
  [...t.tBodies[0].rows].forEach((row, i) => {
    row.classList.add('clickable');
    if (runs[i].run === state.run?.run) row.classList.add('selected');
    row.addEventListener('click', () => { location.hash = encodeURIComponent(runs[i].run); });
  });
}

// --- controls -------------------------------------------------------------------------------------------------------

const THEMES = ['auto', 'light', 'dark'];
function applyTheme(theme) {
  if (theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $('theme').textContent = `Theme: ${theme}`;
  localStorage.setItem('lab-theme', theme);
}
$('theme').addEventListener('click', () => {
  const cur = localStorage.getItem('lab-theme') || 'auto';
  applyTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
});
applyTheme(localStorage.getItem('lab-theme') || 'auto');

$('run-select').addEventListener('change', e => { location.hash = encodeURIComponent(e.target.value); });
window.addEventListener('hashchange', refresh);
$('live').addEventListener('change', schedule);
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => state.run && render(), 150); });

function schedule() {
  clearInterval(state.timer);
  if ($('live').checked) state.timer = setInterval(refresh, POLL_MS);
}
refresh();
schedule();
