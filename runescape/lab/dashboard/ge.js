'use strict';
// GE replay viewer: plays a lab.replay log back as the in-game Grand Exchange, backpack and money pouch.
//
// Every check-in has two frames: arriving (what happened since the last one: fills sit in their slots, sold coins
// and bought items not yet collected) and leaving (after Collect all, the aborts and the new offers). Coins from
// sales reach the pouch when they're collected, as in game. Each leaving frame's pouch and backpack are checked
// against the log's own cash and stock.

const $ = id => document.getElementById(id);
const GE_SLOTS = 8;
const BACKPACK = 28;
const sprite = id => `https://secure.runescape.com/m=itemdb_rs/obj_sprite.gif?id=${id}`;
const TIME = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/Chicago', weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit',
  minute: '2-digit', hour12: false, timeZoneName: 'short',
});

let data = null;
let frames = [];
let at = 0;
let timer = null;

// --- formatting -------------------------------------------------------------------------------------------------

function gp(n) {  // compact: 1.47B, 206.7M, 5.64M, 850K
  const a = Math.abs(n), s = n < 0 ? '−' : '';
  if (a >= 1e9) return s + (a / 1e9).toFixed(2) + 'B';
  if (a >= 1e8) return s + (a / 1e6).toFixed(1) + 'M';
  if (a >= 1e6) return s + (a / 1e6).toFixed(2) + 'M';
  if (a >= 1e3) return s + Math.round(a / 1e3) + 'K';
  return s + a;
}
const signed = n => (n >= 0 ? '+' : '') + gp(n);
const full = n => n.toLocaleString('en-US') + ' gp';
const pouchText = n => Math.floor(n / 1e6) + 'M';  // the in-game pouch: whole millions
const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const item = n => data.items[n];

function icon(n, count) {
  const it = item(n);
  const initials = it.name.split(/\s+/).slice(0, 2).map(w => w[0]).join('');
  const badge = count > 1 ? `<span class="count">${count}</span>` : '';
  return `<img src="${sprite(it.id)}" alt="" loading="lazy" data-initials="${esc(initials)}">${badge}`;
}
// an item without a sprite shows its initials instead
document.addEventListener('error', e => {
  const img = e.target;
  if (img.tagName !== 'IMG' || !img.dataset.initials) return;
  img.replaceWith(Object.assign(document.createElement('span'), { className: 'initials', textContent: img.dataset.initials }));
}, true);

// --- building the frames ----------------------------------------------------------------------------------------

function build() {
  const S = data.meta.slots, ev = data.events, checks = data.checks;
  let pouch = data.meta.stack;
  const bag = new Map();  // item -> units in the backpack
  const slots = new Array(S).fill(null);  // offer objects
  const addBag = (n, q) => { const v = (bag.get(n) || 0) + q; v ? bag.set(n, v) : bag.delete(n); };
  const snap = (kind, k, t, activity) => frames.push({
    kind, k, t, pouch, activity,
    bag: [...bag].filter(([, q]) => q > 0),
    slots: slots.map(o => o && { ...o }),
    uncollected: slots.some(o => o && (o.coins > 0 || (o.side === 'buy' && o.filled > o.taken))),
  });
  const problems = [];
  frames = [];
  let i = 0;
  for (let k = 0; k <= checks.length; k++) {
    const t = k < checks.length ? checks[k].t : data.meta.end;

    // the market since the last check-in: fills
    const sold = new Map();
    while (i < ev.length && ev[i].t < t) {
      const e = ev[i++];
      const o = slots[e.slot];
      o.filled = e.filled;
      if (e.side === 'sell') { o.coins += e.net; o.profit += e.profit; }
      const s = sold.get(o.id) || { o, qty: 0, profit: 0 };
      s.qty += e.fill_qty;
      s.profit += e.profit || 0;
      sold.set(o.id, s);
    }
    const arriving = [...sold.values()].map(({ o, qty, profit }) => o.side === 'sell'
      ? { what: 'Sold', text: `${qty} × ${item(o.item).name} at ${gp(o.price)}`, pl: profit }
      : { what: 'Bought', text: `${qty} × ${item(o.item).name} at ${gp(o.price)}` });
    snap('arrive', k, t, arriving);

    // the check-in: collect all, then aborts and new offers
    const acts = [];
    let collected = 0;
    for (const o of slots) {
      if (!o) continue;
      pouch += o.coins;
      collected += o.coins;
      o.coins = 0;
      if (o.side === 'buy' && o.filled > o.taken) { addBag(o.item, o.filled - o.taken); o.taken = o.filled; }
    }
    if (collected) acts.push({ what: 'Collected', text: `${gp(collected)} coins from sales` });
    while (i < ev.length && ev[i].t === t && ev[i].type !== 'fill') {
      const e = ev[i++];
      const name = e.item != null ? item(e.item).name : '';
      if (e.type === 'collect') {
        acts.push({ what: 'Done', text: `${e.side} ${e.filled}/${e.qty} ${name}` });
        slots[e.slot] = null;
      } else if (e.type === 'cancel') {
        const o = slots[e.slot];
        if (e.side === 'buy') pouch += e.refund;
        else addBag(o.item, o.qty - o.filled);
        acts.push({ what: 'Aborted', text: `${e.side} ${e.filled}/${e.qty} ${name} at ${gp(e.price)}` +
          (e.side === 'buy' ? ` (${gp(e.refund)} back)` : ' (back to the backpack)') });
        slots[e.slot] = null;
      } else if (e.type === 'place') {
        const o = { id: e.offer, slot: e.slot, item: e.item, side: e.side, price: e.price, qty: e.qty, filled: 0,
                    taken: 0, coins: 0, profit: 0, ctx: e };
        slots[e.slot] = o;
        if (e.side === 'buy') pouch -= e.price * e.qty;
        else addBag(e.item, -e.qty);
        const why = e.side === 'buy' && e.margin != null ? ` (aims to sell at ${gp(e.planned_sell)}, ` +
          `${signed(e.margin * e.qty)} after tax)` : e.avg_cost != null ? ` (cost ${gp(e.avg_cost)} each)` : '';
        acts.push({ what: e.side === 'buy' ? 'Buy offer' : 'Sell offer', text: `${e.qty} × ${name} at ${gp(e.price)}${why}` });
      } else if (e.type === 'dump') {
        addBag(e.item, -e.qty);
        pouch += e.net;
        acts.push({ what: 'Sold off', text: `${e.qty} × ${name} at ${gp(e.price)} (end of data)`, pl: e.profit });
      }
    }
    snap('leave', k, t, acts);

    // the log's own cash and stock at this check-in must match the pouch and backpack
    if (k < checks.length) {
      const c = checks[k];
      if (pouch !== c.cash) problems.push(`check ${k}: pouch ${pouch} != log cash ${c.cash}`);
      const listed = new Map();
      for (const o of slots) if (o && o.side === 'sell') listed.set(o.item, (listed.get(o.item) || 0) + o.qty - o.filled);
      for (const [n, q] of Object.entries(c.held)) {
        const want = q - (listed.get(+n) || 0);
        if ((bag.get(+n) || 0) !== want) problems.push(`check ${k}: backpack ${item(n).name} ${bag.get(+n) || 0} != ${want}`);
      }
    } else if (pouch !== data.meta.final_cash) {
      problems.push(`end: pouch ${pouch} != final cash ${data.meta.final_cash}`);
    }
  }
  if (problems.length) console.warn(`${problems.length} mismatches`, problems.slice(0, 20));
  else console.info(`replay checks out: ${frames.length} frames, pouch and backpack match the log at every check-in`);
}

// --- drawing ----------------------------------------------------------------------------------------------------

function offerTip(o) {
  const c = o.ctx, it = item(o.item);
  const lines = [`${it.name} (buy limit ${it.limit})`, `${o.side} ${o.filled}/${o.qty} at ${full(o.price)}`,
    `When placed: instasell ${full(c.instasell)}, instabuy ${full(c.instabuy)}`];
  if (c.margin != null) lines.push(`Plans to sell at ${full(c.planned_sell)}: ${full(c.margin)} each after tax`);
  if (c.avg_cost != null) lines.push(`Cost ${full(c.avg_cost)} each`);
  if (o.side === 'sell' && o.filled) lines.push(`Profit so far ${full(o.profit)}`);
  return esc(lines.join('\n'));
}

function slotHtml(g, o) {
  if (g >= data.meta.slots) {
    return `<div class="slot locked"><div class="slot-head">Slot ${g + 1}</div><div class="note">In use elsewhere</div></div>`;
  }
  if (!o) {
    const btn = (label, color, d) => `<div class="btn"><div class="box"><svg width="22" height="22" viewBox="0 0 22 22">
        <path d="${d}" fill="${color}" stroke="#1a140c" stroke-width="1.2"/></svg></div>${label}</div>`;
    return `<div class="slot empty"><div class="slot-head">Slot ${g + 1}</div><div class="buttons">
      ${btn('Buy', '#d8c22a', 'M7 2h8v9h5l-9 9-9-9h5z')}${btn('Sell', '#e08a2a', 'M7 20h8v-9h5l-9-9-9 9h5z')}</div></div>`;
  }
  const done = o.filled === o.qty;
  const pl = o.side === 'sell' && o.filled
    ? `<span class="pl ${o.profit >= 0 ? 'gain' : 'loss'}">${signed(o.profit)}</span>` : '';
  return `<div class="slot offer" title="${offerTip(o)}">
    <div class="slot-head">${g + 1} - ${o.side === 'buy' ? 'Buy' : 'Sell'}</div><span class="abort">✕</span>
    <div class="slot-main"><div class="icon">${icon(o.item, o.qty)}</div><div class="name">${esc(item(o.item).name)}</div></div>
    <div class="price-row"><span class="price">${gp(o.price)}</span>${pl}</div>
    <div class="bar${done ? ' done' : ''}"><div class="fill" style="width:${100 * o.filled / o.qty}%"></div>
      <span>${o.filled} / ${o.qty}</span></div>
  </div>`;
}

function render() {
  const f = frames[at];
  const html = [];
  for (let g = 0; g < GE_SLOTS; g++) html.push(slotHtml(g, f.slots[g]));
  html.push('<div class="slot lent"><div class="slot-head">Lent item</div><div class="well"></div></div>');
  $('slots').innerHTML = html.join('');

  const cells = f.bag.slice(0, BACKPACK).map(([n, q]) =>
    `<div class="bp-cell" title="${esc(`${item(n).name} × ${q}`)}">${icon(n, q)}</div>`);
  while (cells.length < BACKPACK) cells.push('<div class="bp-cell"></div>');
  $('backpack').innerHTML = cells.join('');
  $('pouch').textContent = pouchText(f.pouch);
  $('pouch-box').title = full(f.pouch);
  $('collect-btn').classList.toggle('idle', !f.uncollected);

  const end = f.k >= data.checks.length;
  $('when').textContent = end ? `End of data · ${TIME.format(f.t * 1000)}`
    : `Check-in ${f.k + 1} / ${data.checks.length} · ${TIME.format(f.t * 1000)}`;
  $('phase').textContent = f.kind === 'arrive' ? (end ? '— before selling off' : '— arriving')
    : (end ? '— everything sold off' : '— after trading');
  const worth = end ? data.meta.final_cash : data.checks[f.k].equity;
  const change = worth / data.meta.stack - 1;
  $('worth').textContent = `Net worth ${pouchText(worth)} (${change >= 0 ? '+' : '−'}${Math.abs(change * 100).toFixed(1)}% ` +
    `from ${pouchText(data.meta.stack)})${segment(f.t)}`;
  $('scrub').value = at;

  $('activity-head').textContent = f.kind === 'arrive' ? 'Since the last check-in' : 'At this check-in';
  $('activity').innerHTML = f.activity.length ? f.activity.map(a => `<li><span class="what">${a.what}</span>
      <span class="detail">${esc(a.text)}</span>${a.pl != null
        ? `<span class="${a.pl >= 0 ? 'gain' : 'loss'}">${signed(a.pl)}</span>` : ''}</li>`).join('')
    : '<li><span class="detail">Nothing</span></li>';
}

function segment(t) {  // which of the champion's data splits this moment is in
  const s = data.meta.segments.find(s => s.start <= t && t < s.end) || data.meta.segments.at(-1);
  return s ? ` · ${s.name === 'forward' ? 'unseen data' : s.name + ' data'}` : '';
}

// --- controls ---------------------------------------------------------------------------------------------------

function go(to) {
  at = Math.max(0, Math.min(frames.length - 1, to));
  render();
  if (at === frames.length - 1) stop();
}
// frame 2k arrives at check-in k, 2k + 1 leaves it
const stepCheck = dir => go(dir > 0 ? 2 * Math.floor(at / 2) + 2 : at % 2 ? at - 1 : at - 2);
function stop() { clearInterval(timer); timer = null; $('play').textContent = '▶ Play'; }
function play() {
  if (timer) return stop();
  if (at === frames.length - 1) at = 0;
  $('play').textContent = '⏸ Pause';
  timer = setInterval(() => go(at + 1), +$('delay').value / 2);  // two frames per check-in
}

$('first').onclick = () => go(0);
$('last').onclick = () => go(frames.length - 1);
$('prev').onclick = () => go(at - 1);
$('next').onclick = () => go(at + 1);
$('prev-check').onclick = () => stepCheck(-1);
$('next-check').onclick = () => stepCheck(1);
$('play').onclick = play;
$('delay').onchange = () => { if (timer) { stop(); play(); } };
$('scrub').oninput = e => go(+e.target.value);
$('replay').onchange = e => load(e.target.value);
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === ' ') { e.preventDefault(); play(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); e.shiftKey ? stepCheck(1) : go(at + 1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); e.shiftKey ? stepCheck(-1) : go(at - 1); }
  else if (e.key === 'Home') go(0);
  else if (e.key === 'End') go(frames.length - 1);
});

async function load(name) {
  stop();
  $('when').textContent = 'Loading…';
  data = await (await fetch(`/replays/${encodeURIComponent(name)}`)).json();
  build();
  $('scrub').max = frames.length - 1;
  const m = data.meta;
  $('hint').textContent = `${m.run} gen ${m.gen} as ${m.policy}: ${pouchText(m.stack)} start, ${m.slots} slots, ` +
    `${m.fill_model} fills. Each check-in shows twice: arriving (before collecting) and after trading. ` +
    `Keys: ←/→ step, Shift+←/→ check-in, Space play. Hover an offer for what the strategy saw.`;
  go(0);
}

(async () => {
  try {
    const list = await (await fetch('/api/replays')).json();
    if (!list.length) throw new Error('No replays yet: run  uv run python -m lab.replay');
    // <made>-<run stamp>-<policy>-<run name>.json -> "made Oct 28 23:04 · soft-no-bonds"
    const label = n => {
      const m = n.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})\d{2}-\d{8}-\d{6}-(.+)\.json$/);
      return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]} · ${m[6]}` : n;
    };
    $('replay').innerHTML = list.map(r => `<option value="${esc(r.name)}">${esc(label(r.name))}</option>`).join('');
    const wanted = new URLSearchParams(location.search).get('replay');
    const name = list.some(r => r.name === wanted) ? wanted : list[0].name;
    $('replay').value = name;
    await load(name);
  } catch (err) {
    $('window').insertAdjacentHTML('afterbegin', `<div class="error">${esc(String(err.message || err))}</div>`);
  }
})();
