/* ================================================================ charts.js
   Hand-rolled SVG charts for the Experiments page. No external library on
   purpose: the site is served from 127.0.0.1 with no network, and everything
   else here (the architecture diagram) is already hand-drawn SVG, so a CDN
   dependency would be both unavailable offline and out of character.

   All four builders take (hostEl, opts), clear the host and append one <svg>
   that scales with its container (viewBox + width:100%).

     lineChart(host, {series, xLabel, yLabel, yRightLabel, mark, xTicks})
     rocChart(host, {curves})
     confusionMatrix(host, {tp, fp, tn, fn, threshold})
     groupedBars(host, {panels, legend})
   ================================================================ */
(function (global) {
  const NS = "http://www.w3.org/2000/svg";
  const CYAN = "#00BFFF", GREEN = "#00E676", AMBER = "#FFA726", RED = "#FF6B6B",
        MUTED = "#A0B4CC", WHITE = "#FFFFFF", GRID = "#243060", PANEL = "#161d36";
  const MONO = "Consolas,'Cascadia Mono',ui-monospace,monospace";

  // Series colours, in assignment order. Cyan first so a single-series chart
  // matches the site's primary accent.
  const SERIES_COLORS = [CYAN, AMBER, GREEN, RED, "#B388FF", "#9BE7B5"];
  global.CHART_COLORS = SERIES_COLORS;

  function el(name, attrs, text) {
    const n = document.createElementNS(NS, name);
    // setAttribute stringifies, so a null would be written as the literal "null" -
    // which SVG rejects for typed attributes (transform, stroke-dasharray) and logs
    // a console error per element. Skipping them is what "no attribute" means here.
    for (const k in attrs) {
      if (attrs[k] === null || attrs[k] === undefined) continue;
      n.setAttribute(k, attrs[k]);
    }
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function txt(x, y, s, o) {
    o = o || {};
    return el("text", {
      x: x, y: y, "text-anchor": o.anchor || "middle",
      "font-size": o.size || 11, "font-weight": o.weight || 400,
      "font-family": MONO, fill: o.color || MUTED,
      transform: o.rotate ? `rotate(${o.rotate} ${x} ${y})` : null,
    }, s);
  }
  function svgRoot(w, h) {
    const s = el("svg", {viewBox: `0 0 ${w} ${h}`, width: "100%",
                         preserveAspectRatio: "xMidYMid meet"});
    s.style.display = "block";
    return s;
  }
  function mount(host, svg) { host.innerHTML = ""; host.appendChild(svg); return svg; }

  /* "Nice" axis bounds: round the data range out to a readable step so the
     gridlines land on numbers a reader can actually read off the axis. */
  function niceScale(lo, hi, ticks) {
    if (!(hi > lo)) { hi = lo + 1; }
    const raw = (hi - lo) / (ticks || 5);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
    return {lo: Math.floor(lo / step) * step, hi: Math.ceil(hi / step) * step, step: step};
  }
  function fmtTick(v, step) {
    const dp = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step)));
    return v.toFixed(dp);
  }

  /* ------------------------------------------------------------- lineChart
     series: [{name, x:[], y:[], color?, axis:"left"|"right", dashed?, marker?}]
     mark:   {x, label}  - a ringed point + vertical rule (the selected epoch)
     Nulls inside y are treated as gaps, not zeros: an arm whose semantic loss
     is undefined for some epochs must not draw a line down to the floor. */
  global.lineChart = function lineChart(host, opts) {
    const W = opts.width || 620, H = opts.height || 260;
    const hasRight = opts.series.some(s => s.axis === "right");
    const m = {t: 18, r: hasRight ? 54 : 16, b: 42, l: 54};
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const svg = svgRoot(W, H);

    const all = opts.series.reduce((a, s) => a.concat(s.x), []);
    const xlo = Math.min.apply(null, all), xhi = Math.max.apply(null, all);
    const pick = side => opts.series.filter(s => (s.axis === "right") === (side === "right"));
    function bounds(list) {
      const v = list.reduce((a, s) => a.concat(s.y.filter(n => n !== null && n !== undefined)), []);
      return v.length ? niceScale(Math.min.apply(null, v), Math.max.apply(null, v), 4)
                      : {lo: 0, hi: 1, step: 0.5};
    }
    const L = bounds(pick("left")), R = hasRight ? bounds(pick("right")) : null;

    const sx = v => m.l + (xhi === xlo ? iw / 2 : (v - xlo) / (xhi - xlo) * iw);
    const syFor = b => v => m.t + ih - (v - b.lo) / (b.hi - b.lo) * ih;

    // horizontal gridlines + left axis ticks
    for (let v = L.lo; v <= L.hi + 1e-9; v += L.step) {
      const y = syFor(L)(v);
      svg.appendChild(el("line", {x1: m.l, y1: y, x2: m.l + iw, y2: y,
                                  stroke: GRID, "stroke-width": 1}));
      svg.appendChild(txt(m.l - 7, y + 3.5, fmtTick(v, L.step), {anchor: "end", size: 10}));
    }
    if (R) {
      for (let v = R.lo; v <= R.hi + 1e-9; v += R.step) {
        svg.appendChild(txt(m.l + iw + 7, syFor(R)(v) + 3.5, fmtTick(v, R.step),
                            {anchor: "start", size: 10, color: AMBER}));
      }
    }
    // x ticks: one per integer epoch, thinned when the run is long
    const xs = opts.xTicks || Array.from(new Set(all)).sort((a, b) => a - b);
    const stride = Math.ceil(xs.length / 12);
    xs.forEach((v, i) => {
      if (i % stride) return;
      svg.appendChild(txt(sx(v), m.t + ih + 16, String(v), {size: 10}));
    });
    if (opts.xLabel) svg.appendChild(txt(m.l + iw / 2, H - 6, opts.xLabel, {size: 10.5}));
    if (opts.yLabel) svg.appendChild(txt(13, m.t + ih / 2, opts.yLabel,
                                         {size: 10.5, rotate: -90}));
    if (opts.yRightLabel) svg.appendChild(txt(W - 8, m.t + ih / 2, opts.yRightLabel,
                                              {size: 10.5, rotate: -90, color: AMBER}));

    if (opts.mark !== undefined && opts.mark !== null && opts.mark.x !== undefined) {
      const x = sx(opts.mark.x);
      svg.appendChild(el("line", {x1: x, y1: m.t, x2: x, y2: m.t + ih,
                                  stroke: GREEN, "stroke-width": 1.2,
                                  "stroke-dasharray": "4 4", opacity: 0.75}));
      svg.appendChild(txt(x, m.t - 5, opts.mark.label || "selected",
                          {size: 9.9, color: GREEN, weight: 600}));
    }

    opts.series.forEach((s, si) => {
      const color = s.color || SERIES_COLORS[si % SERIES_COLORS.length];
      const sy = syFor(s.axis === "right" ? R : L);
      // Split into runs of consecutive non-null points so gaps stay gaps.
      let run = [];
      const flush = () => {
        if (run.length > 1) {
          svg.appendChild(el("polyline", {
            points: run.map(p => `${p[0]},${p[1]}`).join(" "),
            fill: "none", stroke: color, "stroke-width": 1.9,
            "stroke-linejoin": "round", "stroke-linecap": "round",
            "stroke-dasharray": s.dashed ? "5 4" : null,
          }));
        } else if (run.length === 1) {
          svg.appendChild(el("circle", {cx: run[0][0], cy: run[0][1], r: 2.6, fill: color}));
        }
        run = [];
      };
      s.x.forEach((xv, i) => {
        const yv = s.y[i];
        if (yv === null || yv === undefined) { flush(); return; }
        run.push([sx(xv), sy(yv)]);
      });
      flush();
      // Ring the selected checkpoint on every series that has a point there.
      if (opts.mark && opts.mark.x !== undefined) {
        const i = s.x.indexOf(opts.mark.x);
        if (i >= 0 && s.y[i] !== null && s.y[i] !== undefined) {
          svg.appendChild(el("circle", {cx: sx(s.x[i]), cy: sy(s.y[i]), r: 4.6,
                                        fill: "none", stroke: color, "stroke-width": 2}));
        }
      }
    });

    // axes drawn last so they sit above the gridlines
    svg.appendChild(el("line", {x1: m.l, y1: m.t + ih, x2: m.l + iw, y2: m.t + ih,
                                stroke: MUTED, "stroke-width": 1.2}));
    svg.appendChild(el("line", {x1: m.l, y1: m.t, x2: m.l, y2: m.t + ih,
                                stroke: MUTED, "stroke-width": 1.2}));
    return mount(host, svg);
  };

  /* --------------------------------------------------------------- rocChart
     curves: [{name, fpr:[], tpr:[], auc, color?}] */
  global.rocChart = function rocChart(host, opts) {
    const W = opts.width || 300, H = opts.height || 300;
    const m = {t: 14, r: 14, b: 40, l: 44};
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const svg = svgRoot(W, H);
    const sx = v => m.l + v * iw, sy = v => m.t + ih - v * ih;

    for (let i = 0; i <= 4; i++) {
      const g = i / 4;
      svg.appendChild(el("line", {x1: m.l, y1: sy(g), x2: m.l + iw, y2: sy(g),
                                  stroke: GRID, "stroke-width": 1}));
      svg.appendChild(el("line", {x1: sx(g), y1: m.t, x2: sx(g), y2: m.t + ih,
                                  stroke: GRID, "stroke-width": 1}));
      svg.appendChild(txt(m.l - 6, sy(g) + 3.5, g.toFixed(2), {anchor: "end", size: 9.5}));
      svg.appendChild(txt(sx(g), m.t + ih + 14, g.toFixed(2), {size: 9.5}));
    }
    // chance line - the thing every ROC must be read against
    svg.appendChild(el("line", {x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1),
                                stroke: MUTED, "stroke-width": 1,
                                "stroke-dasharray": "4 4", opacity: 0.6}));

    (opts.curves || []).forEach((c, i) => {
      const color = c.color || SERIES_COLORS[i % SERIES_COLORS.length];
      svg.appendChild(el("polyline", {
        points: c.fpr.map((f, j) => `${sx(f)},${sy(c.tpr[j])}`).join(" "),
        fill: "none", stroke: color, "stroke-width": 2,
        "stroke-linejoin": "round",
      }));
      svg.appendChild(txt(m.l + 8, m.t + 14 + i * 14,
        `${c.name}  AUC ${c.auc === null || c.auc === undefined ? "–" : c.auc.toFixed(4)}`,
        {anchor: "start", size: 10.5, color: color, weight: 600}));
    });

    svg.appendChild(txt(m.l + iw / 2, H - 6, "false positive rate", {size: 10}));
    svg.appendChild(txt(12, m.t + ih / 2, "true positive rate", {size: 10, rotate: -90}));
    svg.appendChild(el("line", {x1: m.l, y1: m.t + ih, x2: m.l + iw, y2: m.t + ih,
                                stroke: MUTED, "stroke-width": 1.2}));
    svg.appendChild(el("line", {x1: m.l, y1: m.t, x2: m.l, y2: m.t + ih,
                                stroke: MUTED, "stroke-width": 1.2}));
    return mount(host, svg);
  };

  /* ------------------------------------------------------- confusionMatrix
     Correct cells green, errors red, opacity scaled by share of the row so a
     dominant cell reads at a glance. */
  global.confusionMatrix = function confusionMatrix(host, o) {
    const W = 300, H = 250, cell = 84, x0 = 96, y0 = 58;
    const svg = svgRoot(W, H);
    const cells = [
      {r: 0, c: 0, v: o.tn, ok: true,  t: "TN"},
      {r: 0, c: 1, v: o.fp, ok: false, t: "FP"},
      {r: 1, c: 0, v: o.fn, ok: false, t: "FN"},
      {r: 1, c: 1, v: o.tp, ok: true,  t: "TP"},
    ];
    const rowTot = [o.tn + o.fp, o.fn + o.tp];
    cells.forEach(c => {
      const x = x0 + c.c * cell, y = y0 + c.r * cell;
      const share = rowTot[c.r] ? c.v / rowTot[c.r] : 0;
      svg.appendChild(el("rect", {
        x: x, y: y, width: cell - 4, height: cell - 4, rx: 6,
        fill: c.ok ? GREEN : RED, "fill-opacity": (0.10 + 0.55 * share).toFixed(3),
        stroke: c.ok ? GREEN : RED, "stroke-width": 1.2, "stroke-opacity": 0.55,
      }));
      svg.appendChild(txt(x + (cell - 4) / 2, y + (cell - 4) / 2 + 2, String(c.v),
                          {size: 21, weight: 700, color: WHITE}));
      svg.appendChild(txt(x + (cell - 4) / 2, y + (cell - 4) / 2 + 20, c.t,
                          {size: 10.5, color: c.ok ? GREEN : RED, weight: 600}));
    });
    svg.appendChild(txt(x0 + cell - 2, 26, "predicted", {size: 10.5}));
    svg.appendChild(txt(x0 + cell / 2 - 2, 44, "no event", {size: 10}));
    svg.appendChild(txt(x0 + cell + cell / 2 - 2, 44, "event", {size: 10}));
    svg.appendChild(txt(20, y0 + cell - 6, "actual", {size: 10.5, rotate: -90}));
    svg.appendChild(txt(88, y0 + cell / 2, "no event", {anchor: "end", size: 10}));
    svg.appendChild(txt(88, y0 + cell + cell / 2, "event", {anchor: "end", size: 10}));
    if (o.threshold !== undefined) {
      svg.appendChild(txt(W / 2, H - 8, `threshold ${o.threshold}`,
                          {size: 10, color: AMBER}));
    }
    return mount(host, svg);
  };

  /* -------------------------------------------------------------- histChart
     Score distribution for one bucket, one outlined series per arm.

     Every bucket here holds a SINGLE class (the three TTE buckets are positives,
     the TN bucket is negatives), so a reliability curve is undefined per bucket -
     it needs both classes. What this shows instead is where each arm places that
     one class, which is the calibration question that actually matters here: two
     arms can rank identically and still sit on opposite sides of 0.5.

     The half of the axis where a correct call lands is shaded, and each arm's mean
     score is ticked on the baseline, so a distribution-shift shows up directly.

     opts: {series:[{name,color,values}], threshold, correctSide:"high"|"low", bins} */
  global.histChart = function histChart(host, opts) {
    const W = opts.width || 330, H = opts.height || 210;
    const m = {t: 14, r: 12, b: 42, l: 40};
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const bins = opts.bins || 20;
    const thr = opts.threshold === undefined ? 0.5 : opts.threshold;
    const svg = svgRoot(W, H);

    const hists = opts.series.map(s => {
      const counts = new Array(bins).fill(0);
      s.values.forEach(v => {
        const i = Math.min(bins - 1, Math.max(0, Math.floor(v * bins)));
        counts[i]++;
      });
      const mean = s.values.length
        ? s.values.reduce((a, b) => a + b, 0) / s.values.length : null;
      return {series: s, counts, mean};
    });
    const peak = Math.max(1, ...hists.map(h => Math.max(...h.counts)));
    const sx = v => m.l + v * iw;
    const sy = c => m.t + ih - (c / peak) * ih;

    // shade the side of the threshold where this bucket's class is called correctly
    if (opts.correctSide) {
      const from = opts.correctSide === "high" ? sx(thr) : sx(0);
      const to = opts.correctSide === "high" ? sx(1) : sx(thr);
      svg.appendChild(el("rect", {x: from, y: m.t, width: Math.max(0, to - from),
                                  height: ih, fill: GREEN, "fill-opacity": 0.06}));
    }
    for (let i = 0; i <= 2; i++) {
      const y = m.t + ih - (i / 2) * ih;
      svg.appendChild(el("line", {x1: m.l, y1: y, x2: m.l + iw, y2: y,
                                  stroke: GRID, "stroke-width": 1}));
      svg.appendChild(txt(m.l - 6, y + 3.5, String(Math.round(peak * i / 2)),
                          {anchor: "end", size: 9.5}));
    }
    [0, 0.25, 0.5, 0.75, 1].forEach(v =>
      svg.appendChild(txt(sx(v), m.t + ih + 15, v.toFixed(2), {size: 9.5})));

    hists.forEach((h, i) => {
      const color = h.series.color || SERIES_COLORS[i % SERIES_COLORS.length];
      // step outline rather than filled bars: three overlaid arms stay readable
      const pts = [];
      h.counts.forEach((c, b) => {
        pts.push(`${sx(b / bins)},${sy(c)}`, `${sx((b + 1) / bins)},${sy(c)}`);
      });
      svg.appendChild(el("polyline", {points: pts.join(" "), fill: "none",
                                      stroke: color, "stroke-width": 1.7,
                                      "stroke-linejoin": "round"}));
      if (h.mean !== null) {
        svg.appendChild(el("line", {x1: sx(h.mean), y1: m.t + ih - 5,
                                    x2: sx(h.mean), y2: m.t + ih + 5,
                                    stroke: color, "stroke-width": 2.4}));
        svg.appendChild(txt(sx(h.mean), m.t + ih + 30, h.mean.toFixed(2),
                            {size: 9, color: color, weight: 700}));
      }
    });

    svg.appendChild(el("line", {x1: sx(thr), y1: m.t, x2: sx(thr), y2: m.t + ih,
                                stroke: AMBER, "stroke-width": 1.3,
                                "stroke-dasharray": "4 3"}));
    svg.appendChild(txt(sx(thr), m.t - 3, String(thr), {size: 9, color: AMBER}));
    svg.appendChild(txt(m.l + iw / 2, H - 4, "score", {size: 10}));
    svg.appendChild(el("line", {x1: m.l, y1: m.t + ih, x2: m.l + iw, y2: m.t + ih,
                                stroke: MUTED, "stroke-width": 1.2}));
    return mount(host, svg);
  };

  /* -------------------------------------------------------------- groupedBars
     panels: [{title, sub, bars:[{name, value, max, color?}]}]
     Each bar is drawn over a dimmed "max" bar, so the reader sees the count AND
     the ground-truth ceiling it is being measured against without a second axis. */
  global.groupedBars = function groupedBars(host, opts) {
    const panels = opts.panels || [];
    const pw = 250, ph = 220, gap = 12;
    const W = panels.length * pw + (panels.length - 1) * gap, H = ph;
    const svg = svgRoot(W, H);

    panels.forEach((p, pi) => {
      const ox = pi * (pw + gap);
      const m = {t: 44, b: 46, l: 34, r: 12};
      const iw = pw - m.l - m.r, ih = ph - m.t - m.b;
      const ceiling = Math.max(1, ...p.bars.map(b => b.max || 0));
      const sy = v => m.t + ih - (v / ceiling) * ih;

      svg.appendChild(txt(ox + pw / 2, 18, p.title, {size: 12.1, color: WHITE, weight: 700}));
      if (p.sub) svg.appendChild(txt(ox + pw / 2, 33, p.sub, {size: 9.9}));

      for (let i = 0; i <= 2; i++) {
        const y = m.t + ih - (i / 2) * ih;
        svg.appendChild(el("line", {x1: ox + m.l, y1: y, x2: ox + m.l + iw, y2: y,
                                    stroke: GRID, "stroke-width": 1}));
        svg.appendChild(txt(ox + m.l - 6, y + 3.5, String(Math.round(ceiling * i / 2)),
                            {anchor: "end", size: 9.5}));
      }

      const n = p.bars.length || 1;
      const slot = iw / n, bw = Math.min(46, slot * 0.62);
      p.bars.forEach((b, bi) => {
        const cx = ox + m.l + slot * (bi + 0.5);
        const color = b.color || SERIES_COLORS[bi % SERIES_COLORS.length];
        // shadow bar = the ground-truth total for this bucket
        svg.appendChild(el("rect", {
          x: cx - bw / 2, y: sy(b.max), width: bw, height: Math.max(0, m.t + ih - sy(b.max)),
          rx: 3, fill: MUTED, "fill-opacity": 0.16, stroke: MUTED,
          "stroke-opacity": 0.3, "stroke-width": 1, "stroke-dasharray": "3 3",
        }));
        svg.appendChild(el("rect", {
          x: cx - bw / 2, y: sy(b.value), width: bw,
          height: Math.max(0, m.t + ih - sy(b.value)), rx: 3,
          fill: color, "fill-opacity": 0.85,
        }));
        svg.appendChild(txt(cx, sy(b.value) - 5, String(b.value),
                            {size: 10.5, color: WHITE, weight: 700}));
        svg.appendChild(txt(cx, m.t + ih + 15, b.name, {size: 9.9, color: color, weight: 600}));
        const pct = b.max ? Math.round(100 * b.value / b.max) : 0;
        svg.appendChild(txt(cx, m.t + ih + 28, `${pct}% of ${b.max}`, {size: 9}));
      });

      svg.appendChild(el("line", {x1: ox + m.l, y1: m.t + ih, x2: ox + m.l + iw,
                                  y2: m.t + ih, stroke: MUTED, "stroke-width": 1.2}));
    });
    return mount(host, svg);
  };
})(window);
