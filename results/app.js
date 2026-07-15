/* Spiral Chirality · Results — renderer.
 * Vanilla JS, no libraries. Reads window.RESULTS (set by results.js) so the
 * page works over file:// as well as http. All charts are inline SVG with
 * viewBox (resolution independent). Degrades gracefully on missing data.
 */
(function () {
  'use strict';

  var R = window.RESULTS || {};
  var META = R.meta || {};
  var HIST = Array.isArray(R.history)
    ? R.history.filter(function (h) { return h && isFinite(+h.epoch); })
    : [];
  var TEST = R.test || {};
  var EXAMPLES = R.examples || {};

  var INK = '#586e75', MUTED = '#93a1a1', RULE = '#d9cfb8';
  var CLASS_COLOR = { ccw: '#268bd2', cw: '#cb4b16', discard: '#93a1a1' };
  function classColor(c) { return CLASS_COLOR[c] || CLASS_COLOR.discard; }

  /* ---- tiny helpers -------------------------------------------------- */

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  var SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) { for (var k in attrs) { n.setAttribute(k, attrs[k]); } }
    return n;
  }
  function svText(x, y, str, attrs) {
    var t = sv('text', attrs);
    t.setAttribute('x', x);
    t.setAttribute('y', y);
    t.textContent = str;
    return t;
  }
  function fmtInt(n) {
    n = Math.round(+n);
    if (!isFinite(n)) { return '—'; }
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }
  function fmtStamp(iso) {
    if (!iso) { return ''; }
    return String(iso).replace('T', ' ').replace(/\.\d+/, '')
      .replace(/(?:Z|\+00:00)$/, ' UTC');
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function niceTicks(lo, hi, target) {
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) { return [lo]; }
    var raw = (hi - lo) / Math.max(1, target);
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag, step;
    if (norm >= 7.5) { step = 10 * mag; }
    else if (norm >= 3.5) { step = 5 * mag; }
    else if (norm >= 1.5) { step = 2 * mag; }
    else { step = mag; }
    var ticks = [];
    var v = Math.ceil(lo / step - 1e-9) * step;
    for (; v <= hi + step * 1e-6; v += step) { ticks.push(+v.toPrecision(12)); }
    return { ticks: ticks, step: step };
  }
  function tickFormatter(step) {
    var d = Math.max(0, Math.min(6, -Math.floor(Math.log(step) / Math.LN10 + 1e-9)));
    return function (v) { return (+v).toFixed(d); };
  }

  /* ---- header --------------------------------------------------------*/

  function renderHeader() {
    var slot = document.getElementById('run-meta');
    if (!slot) { return; }
    var bits = [];
    if (META.dataset) { bits.push(META.dataset); }
    if (isFinite(+META.n_train)) {
      bits.push('train ' + fmtInt(META.n_train) + ' / val ' + fmtInt(META.n_val) +
                ' / test ' + fmtInt(META.n_test));
    }
    if (isFinite(+META.n_params)) { bits.push(fmtInt(META.n_params) + ' parameters'); }
    if (isFinite(+META.epochs)) { bits.push(META.epochs + ' epochs'); }
    if (META.seed != null) { bits.push('seed ' + META.seed); }
    if (META.timestamp) { bits.push(fmtStamp(META.timestamp)); }
    slot.textContent = bits.join(' · ');
    if (META.notes) {
      slot.appendChild(document.createTextNode(bits.length ? ' · ' : ''));
      slot.appendChild(el('span', 'notes', META.notes));
    }
  }

  /* ---- training curves ---------------------------------------------- */

  function lineChart(cfg) {
    var W = 470, H = 286, ML = 47, MR = 62, MT = 20, MB = 34;
    var series = [
      { key: cfg.trainKey, label: 'train', dash: null },
      { key: cfg.valKey, label: 'val', dash: '5 4' }
    ];
    var xs = HIST.map(function (h) { return +h.epoch; });
    var xlo = Math.min.apply(null, xs), xhi = Math.max.apply(null, xs);
    if (xlo === xhi) { xlo -= 1; xhi += 1; }
    var vals = [];
    series.forEach(function (s) {
      HIST.forEach(function (h) {
        var v = +h[s.key];
        if (isFinite(v)) { vals.push(v); }
      });
    });
    if (!vals.length) { return null; }
    var ylo = Math.min.apply(null, vals), yhi = Math.max.apply(null, vals);
    var pad = (yhi - ylo) * 0.07 || Math.abs(yhi) * 0.1 || 0.05;
    ylo -= pad; yhi += pad;

    function X(v) { return ML + (v - xlo) / (xhi - xlo) * (W - ML - MR); }
    function Y(v) { return H - MB - (v - ylo) / (yhi - ylo) * (H - MT - MB); }

    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' });

    /* range-frame axes (span the data, Tufte-style) */
    svg.appendChild(sv('line', { x1: X(xlo), x2: X(xhi), y1: H - MB, y2: H - MB, 'class': 'axis' }));
    svg.appendChild(sv('line', { x1: ML, x2: ML, y1: Y(ylo), y2: Y(yhi), 'class': 'axis' }));

    /* x ticks: integers only */
    var xt = niceTicks(xlo, xhi, 4);
    var xticks = xt.ticks.filter(function (v) { return v === Math.round(v); });
    if (xticks.length < 2) { xticks = [Math.round(xlo), Math.round(xhi)]; }
    xticks.forEach(function (v) {
      var x = X(v);
      svg.appendChild(sv('line', { x1: x, x2: x, y1: H - MB, y2: H - MB + 4, 'class': 'axis' }));
      svg.appendChild(svText(x, H - MB + 16, String(v), { 'class': 'tick-label', 'text-anchor': 'middle' }));
    });

    /* y ticks */
    var yt = niceTicks(ylo, yhi, 3);
    var yfmt = tickFormatter(yt.step);
    yt.ticks.forEach(function (v) {
      var y = Y(v);
      svg.appendChild(sv('line', { x1: ML - 4, x2: ML, y1: y, y2: y, 'class': 'axis' }));
      svg.appendChild(svText(ML - 8, y + 3.5, yfmt(v), { 'class': 'tick-label', 'text-anchor': 'end' }));
    });

    /* series */
    series.forEach(function (s) {
      var pts = [];
      HIST.forEach(function (h) {
        var v = +h[s.key];
        if (isFinite(v)) { pts.push([X(+h.epoch), Y(v)]); }
      });
      if (!pts.length) { return; }
      if (pts.length === 1) {
        svg.appendChild(sv('circle', { cx: pts[0][0], cy: pts[0][1], r: 2.2, fill: INK }));
      } else {
        var d = 'M' + pts.map(function (p) {
          return p[0].toFixed(1) + ' ' + p[1].toFixed(1);
        }).join('L');
        var path = sv('path', { d: d, 'class': 'series' });
        if (s.dash) { path.setAttribute('stroke-dasharray', s.dash); }
        svg.appendChild(path);
      }
      s.end = pts[pts.length - 1];
      s.pts = pts;
    });

    /* direct labels at line ends, nudged apart if they collide */
    var labeled = series.filter(function (s) { return s.end; });
    labeled.forEach(function (s) { s.ly = s.end[1]; });
    if (labeled.length === 2 && Math.abs(labeled[0].ly - labeled[1].ly) < 13) {
      var mid = (labeled[0].ly + labeled[1].ly) / 2;
      var order = labeled[0].ly <= labeled[1].ly ? [labeled[0], labeled[1]] : [labeled[1], labeled[0]];
      order[0].ly = mid - 6.5;
      order[1].ly = mid + 6.5;
    }
    labeled.forEach(function (s) {
      svg.appendChild(svText(s.end[0] + 7, clamp(s.ly, MT + 4, H - MB - 2) + 3.5,
        s.label, { 'class': 'end-label' }));
    });

    /* best-validation dot + annotation */
    var bi = -1, bv = NaN;
    HIST.forEach(function (h, i) {
      var v = +h[cfg.valKey];
      if (!isFinite(v)) { return; }
      if (bi < 0 || (cfg.best === 'max' ? v > bv : v < bv)) { bi = i; bv = v; }
    });
    if (bi >= 0) {
      var bx = X(+HIST[bi].epoch), by = Y(bv);
      svg.appendChild(sv('circle', { cx: bx, cy: by, r: 2.6, fill: INK }));
      var lab = (cfg.best === 'max' ? 'best val ' : 'min val ') +
        (+bv).toFixed(cfg.bestDecimals != null ? cfg.bestDecimals : 3) +
        ' · epoch ' + HIST[bi].epoch;
      /* Place the annotation near the dot in empty space. Sample every curve
       * segment densely under the text (interpolated, not just data points),
       * then try increasing vertical offsets (below, then above) at the dot's
       * x and at two shifted positions; take the first clear baseline. */
      var HALF = 66;               /* half of the text's horizontal extent */
      function curveYsNear(cx0) {
        var ys = [];
        series.forEach(function (s) {
          var pts = s.pts || [];
          if (pts.length === 1) {
            if (Math.abs(pts[0][0] - cx0) <= HALF + 6) { ys.push(pts[0][1]); }
            return;
          }
          for (var pi = 1; pi < pts.length; pi++) {
            var a = pts[pi - 1], b = pts[pi];
            var lo = Math.max(Math.min(a[0], b[0]), cx0 - HALF - 6);
            var hi = Math.min(Math.max(a[0], b[0]), cx0 + HALF + 6);
            if (lo > hi) { continue; }
            var n = Math.max(1, Math.ceil((hi - lo) / 3));
            for (var si = 0; si <= n; si++) {
              var px = lo + (hi - lo) * si / n;
              var t = (b[0] - a[0]) ? (px - a[0]) / (b[0] - a[0]) : 0;
              ys.push(a[1] + (b[1] - a[1]) * t);
            }
          }
        });
        return ys;
      }
      function bandClear(ys, ty0) {
        if (ty0 < MT + 8 || ty0 > H - MB - 4) { return false; }
        for (var ni = 0; ni < ys.length; ni++) {
          var dy = ys[ni] - ty0;   /* text occupies ~[ty-9, ty+3] + margin */
          if (dy > -13 && dy < 7) { return false; }
        }
        return true;
      }
      var tx0 = clamp(bx, ML + HALF, W - MR - HALF);
      var txCands = [tx0,
        clamp(tx0 - 90, ML + HALF, W - MR - HALF),
        clamp(tx0 + 90, ML + HALF, W - MR - HALF)];
      var tx = null, ty = null;
      txCands.some(function (cx0) {
        var ys = curveYsNear(cx0);
        return [16, 22, 28, 34, 40, 48, 56, 66, 78, 92].some(function (off) {
          if (bandClear(ys, by + off)) { tx = cx0; ty = by + off; return true; }
          if (bandClear(ys, by - off)) { tx = cx0; ty = by - off; return true; }
          return false;
        });
      });
      if (ty == null) { tx = tx0; ty = clamp(by + 18, MT + 8, H - MB - 4); }
      svg.appendChild(svText(tx, ty, lab, { 'class': 'annot', 'text-anchor': 'middle' }));
    }

    var fig = el('figure', 'chart-fig');
    fig.appendChild(svg);
    fig.appendChild(el('figcaption', 'chart-cap', cfg.caption));
    return fig;
  }

  function renderTraining() {
    var slot = document.getElementById('charts');
    if (!slot) { return; }
    if (!HIST.length) {
      slot.appendChild(el('p', 'empty-note', 'No training history recorded.'));
      return;
    }
    var loss = lineChart({
      trainKey: 'train_loss', valKey: 'val_loss', best: 'min',
      caption: 'Cross-entropy loss by epoch; train solid, validation dashed.'
    });
    var acc = lineChart({
      trainKey: 'train_acc', valKey: 'val_acc', best: 'max',
      caption: 'Classification accuracy by epoch; train solid, validation dashed.'
    });
    if (loss) { slot.appendChild(loss); }
    if (acc) { slot.appendChild(acc); }
    if (!loss && !acc) {
      slot.appendChild(el('p', 'empty-note', 'No training history recorded.'));
    }
  }

  /* ---- test performance ----------------------------------------------*/

  function accuracyBlock() {
    var box = el('div', 'acc-block');
    var acc = +TEST.accuracy;
    box.appendChild(el('div', 'acc-number', isFinite(acc) ? (acc * 100).toFixed(1) + '%' : '—'));
    var n = isFinite(+TEST.n) ? +TEST.n : (TEST.p_cw || []).length;
    box.appendChild(el('div', 'acc-note',
      'test accuracy on ' + fmtInt(n) + ' held-out galaxies'));
    return box;
  }

  function confusionTable() {
    var conf = TEST.confusion;
    if (!conf || conf.length !== 2) { return null; }
    var names = (META.class_names && META.class_names.length === 2)
      ? META.class_names : ['ccw', 'cw'];
    var table = el('table', 'confusion');

    var thead = el('thead');
    var r0 = el('tr');
    r0.appendChild(el('td'));
    var thP = el('th', null, 'predicted');
    thP.colSpan = 2;
    r0.appendChild(thP);
    thead.appendChild(r0);
    var r1 = el('tr', 'class-row');
    r1.appendChild(el('td'));
    names.forEach(function (c) {
      var th = el('th', null, c);
      th.style.color = classColor(c);
      r1.appendChild(th);
    });
    thead.appendChild(r1);
    table.appendChild(thead);

    var tbody = el('tbody');
    conf.forEach(function (row, i) {
      var tr = el('tr');
      var th = el('th');
      th.appendChild(el('span', 'true-label', 'true'));
      var cls = el('span', null, names[i]);
      cls.style.color = classColor(names[i]);
      th.appendChild(cls);
      tr.appendChild(th);
      var total = row.reduce(function (a, b) { return a + (+b || 0); }, 0);
      row.forEach(function (count) {
        var frac = total > 0 ? (+count || 0) / total : 0;
        var td = el('td');
        td.appendChild(document.createTextNode(fmtInt(count)));
        td.appendChild(el('span', 'pct', (100 * frac).toFixed(0) + '%'));
        td.style.backgroundColor = 'rgba(238,232,213,' + frac.toFixed(3) + ')';
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  function histogram() {
    var pv = (TEST.p_cw || []).map(Number).filter(function (p) { return isFinite(p); });
    if (!pv.length) { return null; }
    var W = 470, H = 216, ML = 40, MR = 14, MT = 26, MB = 32;
    var NB = 30, counts = [];
    var i;
    for (i = 0; i < NB; i++) { counts.push(0); }
    pv.forEach(function (p) {
      counts[clamp(Math.floor(p * NB), 0, NB - 1)]++;
    });
    var cmax = Math.max.apply(null, counts) || 1;

    function X(p) { return ML + p * (W - ML - MR); }
    function Y(c) { return H - MB - c / cmax * (H - MT - MB); }

    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' });

    /* faint decision boundary at 0.5 */
    svg.appendChild(sv('line', {
      x1: X(0.5), x2: X(0.5), y1: MT, y2: H - MB,
      stroke: RULE, 'stroke-width': 1, 'stroke-dasharray': '2 4'
    }));

    /* bars */
    for (i = 0; i < NB; i++) {
      if (!counts[i]) { continue; }
      var x0 = X(i / NB), x1 = X((i + 1) / NB);
      var y = Y(counts[i]);
      var center = (i + 0.5) / NB;
      svg.appendChild(sv('rect', {
        x: (x0 + 0.6).toFixed(2), y: y.toFixed(2),
        width: (x1 - x0 - 1.2).toFixed(2), height: (H - MB - y).toFixed(2),
        fill: center < 0.5 ? CLASS_COLOR.ccw : CLASS_COLOR.cw,
        'fill-opacity': 0.5
      }));
    }

    /* axes */
    svg.appendChild(sv('line', { x1: ML, x2: W - MR, y1: H - MB, y2: H - MB, 'class': 'axis' }));
    [0, 0.5, 1].forEach(function (p) {
      var x = X(p);
      svg.appendChild(sv('line', { x1: x, x2: x, y1: H - MB, y2: H - MB + 4, 'class': 'axis' }));
      svg.appendChild(svText(x, H - MB + 16, p === 0.5 ? '0.5' : String(p),
        { 'class': 'tick-label', 'text-anchor': 'middle' }));
    });
    svg.appendChild(sv('line', { x1: ML, x2: ML, y1: Y(0), y2: Y(cmax), 'class': 'axis' }));
    [0, cmax].forEach(function (c) {
      var y = Y(c);
      svg.appendChild(sv('line', { x1: ML - 4, x2: ML, y1: y, y2: y, 'class': 'axis' }));
      svg.appendChild(svText(ML - 8, y + 3.5, String(c), { 'class': 'tick-label', 'text-anchor': 'end' }));
    });

    /* direct class labels over each half */
    svg.appendChild(svText(X(0.25), MT - 9, 'predicted ccw', {
      'class': 'annot', 'text-anchor': 'middle', 'font-style': 'italic', fill: CLASS_COLOR.ccw
    }));
    svg.appendChild(svText(X(0.75), MT - 9, 'predicted cw', {
      'class': 'annot', 'text-anchor': 'middle', 'font-style': 'italic', fill: CLASS_COLOR.cw
    }));

    var fig = el('figure', 'test-right');
    fig.appendChild(svg);
    fig.appendChild(el('figcaption', 'chart-cap',
      'Predicted P(cw) over the test set, 30 bins; the dotted line is the decision boundary.'));
    return fig;
  }

  function renderTest() {
    var slot = document.getElementById('test-body');
    if (!slot) { return; }
    if (!TEST || (!isFinite(+TEST.accuracy) && !TEST.confusion && !(TEST.p_cw || []).length)) {
      slot.appendChild(el('p', 'empty-note', 'No test results recorded.'));
      return;
    }
    var left = el('div', 'test-left');
    left.appendChild(accuracyBlock());
    var conf = confusionTable();
    if (conf) { left.appendChild(conf); }
    slot.appendChild(left);
    var hist = histogram();
    if (hist) { slot.appendChild(hist); }
  }

  /* ---- galleries ------------------------------------------------------*/

  function thumb(src, alt) {
    var wrap = el('span', 'thumb');
    function toMissing() {
      wrap.className = 'thumb missing';
      wrap.textContent = '';
      wrap.appendChild(el('span', 'missing-label', 'unavailable'));
    }
    if (!src) { toMissing(); return wrap; }
    var img = document.createElement('img');
    img.alt = alt || '';
    img.addEventListener('error', toMissing);
    img.src = src;
    wrap.appendChild(img);
    return wrap;
  }

  function exampleFigure(ex) {
    var fig = el('figure', 'example');
    var pair = el('div', 'pair');
    pair.appendChild(thumb(ex.file, 'galaxy ' + ex.id));
    pair.appendChild(thumb(ex.input_file || ex.logpolar_file, 'what the network sees for ' + ex.id));
    fig.appendChild(pair);

    var cap = el('figcaption');
    if (ex.id != null) { cap.appendChild(el('div', 'ex-id', String(ex.id))); }
    var verdict = el('div', 'ex-verdict');
    var t = el('span', null, String(ex['true']));
    t.style.color = classColor(ex['true']);
    verdict.appendChild(t);
    verdict.appendChild(el('span', 'arrow', ' → '));
    var p = el('span', null, String(ex.pred));
    p.style.color = classColor(ex.pred);
    verdict.appendChild(p);
    cap.appendChild(verdict);
    if (isFinite(+ex.p_cw)) {
      cap.appendChild(el('div', 'ex-p', 'p(cw) ' + (+ex.p_cw).toFixed(2)));
    }
    fig.appendChild(cap);
    return fig;
  }

  function renderGalleries() {
    var slot = document.getElementById('galleries-body');
    if (!slot) { return; }
    var groups = [
      { key: 'correct_confident', title: 'Correct & confident', note: null },
      { key: 'incorrect', title: 'Misclassified',
        note: 'Every miss merits a look — faint arms, bars, and flocculent structure are the usual suspects.' },
      { key: 'uncertain', title: 'Most uncertain', note: null }
    ];
    groups.forEach(function (g) {
      var items = Array.isArray(EXAMPLES[g.key]) ? EXAMPLES[g.key] : [];
      slot.appendChild(el('h3', null, g.title));
      if (g.note) { slot.appendChild(el('p', 'gallery-note', g.note)); }
      if (!items.length) {
        slot.appendChild(el('p', 'empty-note', 'None.'));
        return;
      }
      var gal = el('div', 'gallery');
      items.forEach(function (ex) {
        if (ex) { gal.appendChild(exampleFigure(ex)); }
      });
      slot.appendChild(gal);
    });
  }

  /* ---- footer ----------------------------------------------------------*/

  function renderFooter() {
    var slot = document.getElementById('colophon');
    if (!slot) { return; }
    slot.textContent = 'ChiralityNet · ' +
      (isFinite(+META.n_params) ? fmtInt(META.n_params) : '—') +
      ' parameters · a small CNN scored on the image minus its mirror';
  }

  renderHeader();
  renderTraining();
  renderTest();
  renderGalleries();
  renderFooter();
})();
