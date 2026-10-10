/* pilot.js - the "Boxed-teacher pilot" view of experiments.html.

   One row per pilot window (same 51 windows for R1 / R2 / R3, so the runs can be read side by side).
   Sorting and filtering follow the Cross-Experiment Comparison table: click a header to cycle
   asc -> desc -> off, nulls always last, a filter row under the header. Comments live in the SAME
   localStorage store as the comparison view (ccp:review-notes:v1), under their own dataset key, so
   Export / Import notes keep working for both. The store, not the DOM, is the source of truth:
   the tbody is rebuilt on every sort/filter, and typing a note deliberately does not re-render.

   createPilotView({data, esc, notes, saveNotes, exportNotes, importNotes}) -> {show()} */
function createPilotView(deps){
  const D = deps.data, esc = deps.esc, KEY = D.dataset_key;
  const ALL_RUNS = Object.keys(D.runs);
  let RUNS = (D.default_on || ALL_RUNS).filter(r => ALL_RUNS.includes(r));       // runs shown as columns (chip row)
  const $ = id => document.getElementById(id);
  const state = {sort: null, dir: 1, filters: {}, disagree: false, failsOnly: true};
  const e3Wrong = r => { const x = r.runs.E3; return !!(x && x.verdict && ((x.verdict === "yes") !== ((r.label_vis !== undefined ? r.label_vis : r.label) === 1))); };
  let built = false;

  const HV = D.hv_key || (KEY + "_hv");
  const hvOf = r => ((deps.notes()[HV] || {})[r.key]) || "";
  function setHV(key, v){
    const N = deps.notes();
    if (!N[HV]) N[HV] = {};
    if (v) N[HV][key] = v; else delete N[HV][key];
    deps.saveNotes();
  }
  const noteOf = r => ((deps.notes()[KEY] || {})[r.key]) || "";
  function setNote(key, text){
    const N = deps.notes();
    if (!N[KEY]) N[KEY] = {};
    if (text) N[KEY][key] = text; else delete N[KEY][key];
    deps.saveNotes();
  }
  function updateNoteCount(){
    const n = Object.keys(deps.notes()[KEY] || {}).length;
    $("pilotNoteCount").textContent = n ? `${n} note${n === 1 ? "" : "s"} on this table` : "";
  }

  const isRight = (r, run) => {
    const x = r.runs[run];
    return x && x.verdict ? ((x.verdict === "yes") === ((r.label_vis !== undefined ? r.label_vis : r.label) === 1)) : null;
  };
  const verdictGet = run => r => { const x = r.runs[run]; return x && x.verdict ? x.verdict : null; };

  function columns(){
    const cols = [
      {key: "grid", label: "Frames 4×4 (red box)", type: "none"},
      {key: "video_id", label: "Video ID", type: "num", get: r => parseInt(r.video_id, 10), cell: r => esc(r.video_id)},
      {key: "tte", label: "TTE", type: "select", get: r => r.tte.toFixed(1),
       opts: [["", "All"], ["0.5", "0.5 s"], ["1.0", "1.0 s"], ["1.5", "1.5 s"]],
       cell: r => r.label === 1 ? `${r.tte.toFixed(1)} s` : `<span class="dim" title="normal clip: window ends ${r.tte.toFixed(1)} s before a fake event at the video midpoint">mid ${r.tte.toFixed(1)} s</span>`},
      {key: "set", label: "Set", type: "select",
       title: "pilot = the 51-window prompt study; rescue = E6 rescue test on crash windows the visibility rule removes (window ends before time_of_alert + 0.27 s)",
       get: r => r.set === "rescue" ? (r.valid ? "rescue_kept" : "rescue_removed") : "pilot",
       opts: [["", "All"], ["pilot", "pilot"], ["rescue_kept", "rescue · kept by rule"], ["rescue_removed", "rescue · removed by rule"]],
       cell: r => r.set !== "rescue" ? '<span class="dim">pilot</span>'
         : (r.valid ? '<span class="chip neg" title="the visibility rule keeps this window">rescue · kept</span>'
                    : '<span class="chip pos" title="the visibility rule removes this window: it ends before time_of_alert + 0.27 s">rescue · removed</span>')},
      {key: "label", label: "GT label", type: "select",
       title: "Nexar event label → visible-hazard label (label_vis). A crash window that ends before the annotated alert shows 1 → 0 · pre-alert: nothing announces the collision yet, so the target verdict is no. 1 → 1 · seen early = E7-pre found the hazard already visible.",
       get: r => r.pre_alert ? (r.label_vis === 1 ? "1>1" : "1>0") : String(r.label),
       opts: [["", "All"], ["1", "1 · crash"], ["0", "0 · normal"], ["1>0", "1 → 0 · pre-alert"], ["1>1", "1 → 1 · seen early"]],
       cell: r => r.pre_alert
         ? `<span class="chip pos" title="Nexar label 1, visible-hazard label ${r.label_vis}">1 → ${r.label_vis}</span><div class="dim hvhint">pre-alert${r.label_vis === 1 ? " · seen early" : ""}${r.status && r.status !== "ok" ? " · " + esc(r.status) : ""}</div>`
         : `<span class="chip ${r.label === 1 ? "pos" : "neg"}">${r.label}</span>${r.status && r.status !== "ok" ? `<div class="dim hvhint">${esc(r.status)}</div>` : ""}`},
    ];
    RUNS.forEach(run => cols.push({
      key: run + "_v", label: run + "_verdict", type: "verdict", run,
      title: D.runs[run], get: verdictGet(run),
      cell: r => {
        const x = r.runs[run];
        if (!x || !x.verdict) return '<span class="dim">–</span>';
        const ok = isRight(r, run);
        const rk = x.risk !== null && x.risk !== undefined ? ` <span class="prisk" title="risk_score 0-100 (yes if ≥ 50)">${esc(x.risk)}</span>` : "";
        const fl = x.from_e4 ? ` <span class="pflip" title="this answer replaces E3's: it comes from the ${esc(x.from_e4)} re-run (the window was an E3 mistake)">↻ ${esc(x.from_e4)}</span>` : "";
        return `<span class="pv ${ok ? "ok" : "no"}" title="${ok ? "matches" : "differs from"} the GT label">${esc(x.verdict)} ${ok ? "✓" : "✗"}</span>${rk}${fl}`;
      }}));
    RUNS.forEach(run => cols.push({
      key: run + "_r", label: run + "_reasoning", type: "text", title: D.runs[run], reason: true,
      get: r => { const x = r.runs[run]; return x && x.explanation ? x.explanation : null; },
      cell: r => {
        const x = r.runs[run];
        if (!x || !x.explanation) return '<span class="dim">–</span>';
        const warn = x.problems && x.problems.length
          ? ` <span class="pwarn" title="${esc(x.problems.join("; "))}">⚠</span>` : "";
        const pred = x.prediction ? `<div class="ppred" title="the teacher's one-sentence prediction (outcome words allowed here)">→ ${esc(x.prediction)}</div>` : "";
        return `<div class="ptext">${esc(x.explanation)}${warn}</div>${pred}<div class="ptags">${esc(x.tags || "")}</div>`;
      }}));
    cols.push({key: "hv", label: "Hitter visible? (you)", type: "none", title: "your reading: is the road user that causes the collision visible in this window? (crash windows only)",
      cell: r => {
        if (r.label !== 1) return '<span class="dim">–</span>';
        const v = hvOf(r);
        const opt = (k, l) => `<option value="${k}"${v === k ? " selected" : ""}>${l}</option>`;
        return `<select class="hvsel${v ? " has-note" : ""}" data-hv="${esc(r.key)}">${opt("", "?")}${opt("yes", "yes")}${opt("partly", "partly (≤3 frames)")}${opt("no", "no")}</select>`;
      }});
    cols.push({key: "note", label: "Comments", type: "text", get: r => noteOf(r) || null,
      cell: r => {
        const v = noteOf(r);
        return `<textarea class="noteinput${v ? " has-note" : ""}" data-key="${esc(r.key)}" rows="3" placeholder="add a note…">${esc(v)}</textarea>`;
      }});
    return cols;
  }
  let COLS = columns();

  function passes(r){
    if (state.failsOnly && r.set !== "rescue" && !e3Wrong(r)) return false;
    if (state.disagree){
      const vs = RUNS.map(run => verdictGet(run)(r)).filter(v => v !== null);
      if (new Set(vs).size < 2) return false;
    }
    for (const c of COLS){
      if (c.type === "none") continue;
      const f = state.filters[c.key];
      if (f === undefined || f === "" || f === null) continue;
      if (c.type === "verdict"){
        const v = c.get(r), ok = isRight(r, c.run);
        if (f === "none"){ if (v !== null) return false; }
        else if (f === "correct"){ if (ok !== true) return false; }
        else if (f === "wrong"){ if (ok !== false) return false; }
        else if (v !== f) return false;
        continue;
      }
      const v = c.get(r);
      if (c.type === "select"){ if (String(v) !== f) return false; }
      else if (c.type === "text"){ if (!String(v || "").toLowerCase().includes(String(f).toLowerCase())) return false; }
      else {
        if (f.min !== "" && f.min !== undefined && (v === null || v < parseFloat(f.min))) return false;
        if (f.max !== "" && f.max !== undefined && (v === null || v > parseFloat(f.max))) return false;
      }
    }
    return true;
  }
  function visible(){
    let rows = D.rows.filter(passes);
    if (state.sort){
      const col = COLS.find(c => c.key === state.sort);
      if (col) rows = rows.slice().sort((a, b) => {
        const av = col.get(a), bv = col.get(b);
        const an = av === null || av === undefined, bn = bv === null || bv === undefined;
        if (an && bn) return 0;
        if (an) return 1;                         // nulls always last, whatever the direction
        if (bn) return -1;
        if (typeof av === "string") return av.localeCompare(bv) * state.dir;
        return (av - bv) * state.dir;
      });
    }
    return rows;
  }

  function rowHTML(r){
    const g = r.grid
      ? `<div class="pgrid" data-key="${esc(r.key)}"><img loading="lazy" src="${esc(r.grid)}" alt="${esc(r.key)}">
           <button class="pplus" title="enlarge" data-key="${esc(r.key)}">＋</button></div>`
      : '<span class="dim">frames pending</span>';
    return "<tr><td>" + g + "</td>" + COLS.slice(1).map(c =>
      `<td class="${c.type === "num" ? "num" : ""}${c.reason ? " reason" : ""}">${c.cell(r)}</td>`).join("") + "</tr>";
  }

  function renderHead(){
    $("pilotHead").innerHTML =
      `<tr>${COLS.map(c => c.type === "none" ? `<th>${esc(c.label)}</th>`
        : `<th class="sortable ${state.sort === c.key ? (state.dir === 1 ? "sort-asc" : "sort-desc") : ""}" data-col="${esc(c.key)}"
              ${c.title ? `title="${esc(c.title)}"` : ""}>${esc(c.label)}<span class="caret"></span></th>`).join("")}</tr>
       <tr class="filter-row">${COLS.map(c => {
         if (c.type === "none") return "<th></th>";
         if (c.type === "select") return `<th><select data-col="${esc(c.key)}">${c.opts.map(([v, l]) =>
           `<option value="${esc(v)}"${state.filters[c.key] === v ? " selected" : ""}>${esc(l)}</option>`).join("")}</select></th>`;
         if (c.type === "verdict"){
           const f = state.filters[c.key] || "";
           return `<th><select data-col="${esc(c.key)}">${[["", "All"], ["yes", "yes"], ["no", "no"], ["correct", "✓ matches GT"], ["wrong", "✗ differs"], ["none", "– none"]]
             .map(([v, l]) => `<option value="${v}"${f === v ? " selected" : ""}>${l}</option>`).join("")}</select></th>`;
         }
         if (c.type === "text") return `<th><input type="text" placeholder="contains…" data-col="${esc(c.key)}" value="${esc(state.filters[c.key] || "")}"></th>`;
         const f = state.filters[c.key] || {};
         return `<th><span class="range"><input type="text" placeholder="min" data-col="${esc(c.key)}" data-b="min" value="${esc(f.min || "")}">
                 <input type="text" placeholder="max" data-col="${esc(c.key)}" data-b="max" value="${esc(f.max || "")}"></span></th>`;
       }).join("")}</tr>`;
    const head = $("pilotHead");
    head.querySelectorAll("th.sortable").forEach(th => th.addEventListener("click", () => {
      const k = th.dataset.col;
      if (state.sort === k){ state.dir = state.dir === 1 ? -1 : 0; if (state.dir === 0){ state.sort = null; state.dir = 1; } }
      else { state.sort = k; state.dir = 1; }
      head.querySelectorAll("th.sortable").forEach(x => x.classList.remove("sort-asc", "sort-desc"));
      if (state.sort === k) th.classList.add(state.dir === 1 ? "sort-asc" : "sort-desc");
      renderBody();
    }));
    head.querySelectorAll("select[data-col]").forEach(s => s.addEventListener("change", () => {
      state.filters[s.dataset.col] = s.value; renderBody();
    }));
    head.querySelectorAll("input[data-col]").forEach(i => i.addEventListener("input", () => {
      const k = i.dataset.col;
      if (i.dataset.b){ state.filters[k] = Object.assign({min: "", max: ""}, state.filters[k]); state.filters[k][i.dataset.b] = i.value.trim(); }
      else state.filters[k] = i.value.trim();
      renderBody();
    }));
  }
  function renderBody(){
    const rows = visible();
    $("pilotBody").innerHTML = rows.map(rowHTML).join("");
    $("pilotCount").textContent = `${rows.length} of ${D.rows.length} windows`;
    updateNoteCount();
  }

  function renderRuns(){
    const f = ([a, b], ci) => b ? `${a}/${b} (${Math.round(100 * a / b)}%)${ci ? ` <span class="dim">[${ci[0]}–${ci[1]}]</span>` : ""}` : "–";
    const t = s => s ? `${s.toFixed(0)} s` : "–";
    const rows = ALL_RUNS.map(run => {
      const s = D.summary[run], r = s.run;
      return `<tr class="${RUNS.includes(run) ? "" : "dimrow"}"><td>${run}</td><td class="hpdesc">${esc(s.setting)}</td><td>${s.parsed}/${s.n || 0}</td>
        <td>${f(s.acc_all)}</td><td>${f(s.acc_crash, s.tp_ci)}</td><td>${f(s.acc_normal, s.tn_ci)}</td>
        <td>${s.risk_auc !== undefined && s.risk_auc !== null ? s.risk_auc.toFixed(3) : "–"}</td>
        <td>${r ? t(r.elapsed_total_s) : "–"}</td><td>${r ? `${r.latency_median_s} / ${r.latency_p95_s} s` : "–"}</td>
        <td>${r ? "$" + r.cost_per_window_usd.toFixed(4) : "–"}</td><td>${r ? "$" + r.cost_total_usd.toFixed(2) : "–"}</td>
        <td>${r ? r.with_problems : "–"}</td></tr>`;
    }).join("");
    $("pilotRuns").innerHTML = `<thead><tr><th>run</th><th>setting</th><th>windows parsed</th><th title="verdict equals the GT label, all windows it ran on">verdict = GT (all)</th>
      <th title="TP: crash windows (label 1) the teacher called yes; [95% Wilson]">crash → yes (TP)</th><th title="TN: normal windows (label 0) the teacher called no; [95% Wilson]">normal → no (TN)</th>
      <th title="rank AUC of risk_score, crash vs normal">risk AUC</th><th>total time</th><th>latency median / p95</th><th>cost / window</th><th>cost total</th><th>soft problems</th></tr></thead><tbody>${rows}</tbody>`;
    const mc = D.mcnemar || {};
    $("pilotPaired").innerHTML = Object.keys(mc).length ? "Paired tests on the same 51 windows (verdict = GT): " + Object.entries(mc).map(([k, v]) => {
      const [a, b] = k.split("|");
      return `<b>${a} vs ${b}</b> only-${a}-right ${v.only_first}, only-${b}-right ${v.only_second}, exact p = ${v.p}`;
    }).join(" · ") + ". p ≥ 0.1 = not distinguishable, not equal. E3 → E34 is label-routed and therefore not a perception test." : "";
  }

  function renderChips(){
    $("pilotChips").innerHTML = '<span class="dim">columns:</span> ' + ALL_RUNS.map(run =>
      `<button class="pill chip-run${RUNS.includes(run) ? " active" : ""}" data-run="${run}" title="${esc(D.runs[run])}">${run}</button>`).join(" ");
    $("pilotChips").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      const run = b.dataset.run;
      RUNS = RUNS.includes(run) ? RUNS.filter(x => x !== run) : ALL_RUNS.filter(x => RUNS.includes(x) || x === run);
      COLS = columns();
      if (state.sort && !COLS.find(c => c.key === state.sort)){ state.sort = null; state.dir = 1; }
      renderChips(); renderRuns(); renderHead(); renderBody();
    }));
  }

  function openGrid(key){
    const r = D.rows.find(x => x.key === key);
    if (!r || !(r.grid_large || r.grid)) return;
    $("gImg").src = r.grid_large || r.grid;
    $("gTitle").textContent = `#${r.video_id} · ${r.key}`;
    $("gMeta").textContent = `GT ${r.pre_alert ? "1 → " + r.label_vis + " (pre-alert)" : r.label === 1 ? "1 (crash)" : "0 (normal)"} · ${r.label === 1 ? "TTE" : "mid"} ${r.tte.toFixed(1)} s · ` +
      RUNS.map(run => `${run} ${(r.runs[run] && r.runs[run].verdict) || "–"}`).join(" · ");
    $("gridOverlay").classList.add("open");
  }
  const closeGrid = () => $("gridOverlay").classList.remove("open");

  function build(){
    built = true;
    $("pilotNote").textContent = D.note;
    renderChips();
    renderRuns();
    renderHead();
    $("pilotClear").addEventListener("click", () => { state.filters = {}; state.sort = null; state.dir = 1; state.disagree = false; state.failsOnly = false;
      $("pilotDisagree").classList.remove("active"); $("pilotFails").classList.remove("active"); renderHead(); renderBody(); });
    $("pilotFails").classList.toggle("active", state.failsOnly);
    $("pilotFails").addEventListener("click", e => { state.failsOnly = !state.failsOnly; e.currentTarget.classList.toggle("active", state.failsOnly); renderBody(); });
    if (D.routing_fig){ $("pilotRoute").src = D.routing_fig; $("pilotRouteLink").href = D.routing_fig; }
    $("pilotBody").addEventListener("change", e => {
      const s = e.target.closest("select[data-hv]");
      if (!s) return;
      setHV(s.dataset.hv, s.value);
      s.classList.toggle("has-note", !!s.value);
    });
    $("pilotDisagree").addEventListener("click", e => { state.disagree = !state.disagree; e.currentTarget.classList.toggle("active", state.disagree); renderBody(); });
    $("pilotBody").addEventListener("input", e => {
      const ta = e.target.closest("textarea[data-key]");
      if (!ta) return;
      const text = ta.value.trim();
      setNote(ta.dataset.key, text);
      ta.classList.toggle("has-note", !!text);
      updateNoteCount();
    });
    $("pilotBody").addEventListener("click", e => {
      const el = e.target.closest("[data-key]");
      if (el && (el.classList.contains("pplus") || el.classList.contains("pgrid") || e.target.tagName === "IMG")) openGrid(el.dataset.key);
    });
    $("gClose").addEventListener("click", closeGrid);
    $("gridOverlay").addEventListener("click", e => { if (e.target === $("gridOverlay")) closeGrid(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") closeGrid(); });
    $("pilotExport").addEventListener("click", deps.exportNotes);
    $("pilotImport").addEventListener("click", () => $("pilotFile").click());
    $("pilotFile").addEventListener("change", e => { if (e.target.files[0]) deps.importNotes(e.target.files[0]); e.target.value = ""; });
  }
  return {show(){ if (!built) build(); renderBody(); }, renderBody};
}
