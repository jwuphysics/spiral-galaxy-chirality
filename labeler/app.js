/* Spiral Chirality · Labeler.  Vanilla JS, no dependencies.
 *
 * Convention (matches Galaxy Zoo 1): key 1 = "ccw" = S-wise / anti-clockwise,
 * key 2 = "cw" = Z-wise / clockwise, key 3 = "discard", key 0 = clear.
 * Every keypress persists the full labels object immediately via POST.
 */
'use strict';

const params   = new URLSearchParams(location.search);
const MANIFEST = params.get('manifest') || '../data/manifest.json';
const LABELS   = params.get('labels')   || 'labels.json';
const IMGBASE  = MANIFEST.slice(0, MANIFEST.lastIndexOf('/') + 1);

const KEY_TO_LABEL = { 1: 'ccw', 2: 'cw', 3: 'discard' };
const LABEL_WORD   = { ccw: 'counterclockwise · S', cw: 'clockwise · Z', discard: 'discard' };

const el = {};
for (const id of ['msg', 'viewer', 'galaxy', 'gid', 'pos', 'word', 'fill', 'ptext', 'warn'])
  el[id] = document.getElementById(id);

let items  = [];   // manifest items, in manifest order
let labels = {};   // id -> {label, ts}
let idx    = 0;    // current position

const imgURL = (item) => IMGBASE + item.file;

function firstUnlabeled(from) {
  for (let i = from; i < items.length; i++)
    if (!labels[items[i].id]) return i;
  return -1;
}

function render() {
  const it = items[idx];
  el.galaxy.src = imgURL(it);
  el.gid.textContent = it.id;
  el.pos.textContent = `${idx + 1} / ${items.length}`;

  const cur = labels[it.id];
  el.word.textContent = cur ? LABEL_WORD[cur.label] : '—';
  el.word.className = cur ? cur.label : 'none';

  let done = 0, discarded = 0;
  for (const item of items) {
    const l = labels[item.id];
    if (l) { done++; if (l.label === 'discard') discarded++; }
  }
  el.fill.style.width = items.length ? (100 * done / items.length).toFixed(2) + '%' : '0';
  el.ptext.textContent = `${done} of ${items.length} labeled · ${discarded} discarded`;

  for (const j of [idx + 1, idx - 1])           // preload neighbours
    if (items[j]) new Image().src = imgURL(items[j]);
}

function warn(text) {
  el.warn.textContent = text;
  el.warn.hidden = !text;
}

/* Saves are serialized so an early snapshot can never overwrite a later one. */
let saveChain = Promise.resolve();
function persist() {
  const body = JSON.stringify(
    { version: 1, updated: new Date().toISOString(), labels }, null, 1);
  saveChain = saveChain.then(async () => {
    try {
      const r = await fetch(LABELS, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if (!r.ok && r.status !== 204) throw new Error('HTTP ' + r.status);
      warn('');
    } catch (err) {
      warn(`Saving failed (${err.message}). Is serve.py still running? ` +
           'Labels are kept in this page — fix the server, then label anything to retry.');
    }
  });
}

function go(i) {
  idx = Math.max(0, Math.min(items.length - 1, i));
  render();
}

function setLabel(name) {
  labels[items[idx].id] = { label: name, ts: new Date().toISOString() };
  persist();
  const next = firstUnlabeled(idx + 1);   // next unlabeled, else simply next
  go(next >= 0 ? next : idx + 1);
}

function clearLabel() {
  if (labels[items[idx].id]) {
    delete labels[items[idx].id];
    persist();
  }
  render();
}

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || items.length === 0) return;
  if (KEY_TO_LABEL[e.key])           setLabel(KEY_TO_LABEL[e.key]);
  else if (e.key === '0')            clearLabel();
  else if (e.key === 'ArrowLeft')    go(idx - 1);
  else if (e.key === 'ArrowRight')   go(idx + 1);
  else return;
  e.preventDefault();
});

async function init() {
  let manifest;
  try {
    const r = await fetch(MANIFEST, { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    manifest = await r.json();
  } catch (err) {
    el.msg.textContent =
      `Could not load the manifest at ${MANIFEST} (${err.message}). ` +
      'From the project root run:  python3 labeler/serve.py  — then open ' +
      'http://localhost:8801/labeler/ . If the dataset is still downloading, try again in a bit.';
    el.msg.hidden = false;
    return;
  }
  items = manifest.items || [];
  if (items.length === 0) {
    el.msg.textContent = 'The manifest loaded but contains no items.';
    el.msg.hidden = false;
    return;
  }
  try {
    const r = await fetch(LABELS, { cache: 'no-store' });
    if (r.ok) labels = (await r.json()).labels || {};   // 404 → start empty
  } catch (err) { /* no labels yet */ }

  el.viewer.hidden = false;
  const first = firstUnlabeled(0);
  go(first >= 0 ? first : 0);
}

init();
