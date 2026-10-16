/* p2f.js - the "Reasoning model - 3 formats" view of experiments.html (night run 2026-10-15/16: P2R reason-first, P2V verdict-first, P2F facts-first).

   window.P2F_DATA (built by build_p2f_data.py): summary tables + one row per validation window (524, pre-alert removed) with the crash model and,
   per run, the written verdict, P(yes), reasoning, outcome, facts line, the fact card of the reasoning and flags (missed / fa / rescue / lost / misaligned).
   Per-window table: sortable columns (click a header), group / label filter, flag filters per run, search over the id and all texts, 30 rows per page.
   The enlarge overlay (#gridOverlay) is the page's shared one.

   createP2FView({data, esc}) -> {show()} */
function createP2FView(deps){
  const D = deps.data, esc = deps.esc, PAGE = 30;
  const $ = id => document.getElementById(id);
  const RK = D.runs.map(r => r.key);
  const pct = v => (v === null || v === undefined) ? "–" : (100 * v).toFixed(0) + "%";
  const f3 = v => (v === null || v === undefined) ? "–" : Number(v).toFixed(3);
  const f2 = v => (v === null || v === undefined) ? "–" : Number(v).toFixed(2);
  const GL = {"normal": "normal", "tte0.5": "TTE 0.5 s", "tte1.0": "TTE 1.0 s", "tte1.5": "TTE 1.5 s"};
  const FLAGS = [["missed", "missed crash"], ["fa", "false alarm"], ["rescue", "right where crash model is wrong"], ["lost", "wrong where crash model is right"],
                 ["misaligned", "verdict and reasoning disagree"]];
  const state = {group: "", label: "", run: "P2V", flag: "", crashWrong: false, disagree: false, q: "", sort: {key: "id", dir: 1}, page: 0};
  let built = false;

  /* ----------------------------------------------------------- summary tables */
  function renderTables(){
    $("p2fDecision").innerHTML = `<thead><tr><th>model</th><th title="rank AUC vs the visible-hazard label, all 524 windows">AUC</th><th>AP</th>
      <th title="crash windows called yes / no at P = 0.5">TP / FN</th><th title="normal windows called yes / no at P = 0.5">FP / TN</th>
      <th>recall</th><th>specificity</th><th>precision</th><th>missed crashes all</th><th>0.5 s</th><th>1.0 s</th><th>1.5 s</th>
      <th title="median P(yes) on crash / normal windows">median P crash / normal</th></tr></thead><tbody>${D.decision.map(d => `<tr>
      <td>${esc(d.label)}</td><td class="num">${f3(d.auc)}</td><td class="num">${f3(d.ap)}</td><td class="num">${d.TP} / ${d.FN}</td><td class="num">${d.FP} / ${d.TN}</td>
      <td class="num">${f2(d.recall)}</td><td class="num">${f2(d.specificity)}</td><td class="num">${f2(d.precision)}</td><td class="num">${pct(d.missed_all)}</td>
      ${["tte0.5", "tte1.0", "tte1.5"].map(g => `<td class="num">${pct(d.missed[g])}</td>`).join("")}
      <td class="num">${f2(d.median_p_crash)} / ${f2(d.median_p_normal)}</td></tr>`).join("")}</tbody>`;
    $("p2fEqual").innerHTML = `<thead><tr><th>false alarms on normal windows</th><th>Crash model</th>${RK.map(k => `<th>${k}</th>`).join("")}</tr></thead><tbody>${
      D.equal_fa.map(r => `<tr><td>${pct(r.fa)}</td><td class="num">${pct(r.crash)}</td>${RK.map(k => `<td class="num">${pct(r[k])}</td>`).join("")}</tr>`).join("")}</tbody>`;
    $("p2fRescue").innerHTML = `<thead><tr><th>run</th><th title="crash windows the run calls yes while the crash model says no">right, crash model wrong: crash windows</th>
      <th title="normal windows the run calls no while the crash model raises a false alarm">… normal windows</th><th title="of those, how many have reasoning that matches the label (gap closing on a crash, not closing on normal)">… with consistent reasoning</th>
      <th>wrong, crash model right: crash windows</th><th>… normal windows</th></tr></thead><tbody>${D.rescues.map(r => `<tr><td>${esc(r.label)}</td>
      <td class="num">${r.rescue_crash}</td><td class="num">${r.rescue_normal}</td><td class="num">${r.rescue_reason_consistent}</td><td class="num">${r.lost_crash}</td><td class="num">${r.lost_normal}</td></tr>`).join("")}</tbody>`;
    const cells = ["crash|yes|closing", "crash|yes|other", "crash|no|closing", "crash|no|other", "normal|yes|closing", "normal|yes|other", "normal|no|closing", "normal|no|other"];
    const hdr = {"crash|yes|closing": "crash · yes · closing", "crash|yes|other": "crash · yes · <b>not closing</b>", "crash|no|closing": "crash · no · <b>closing</b>",
                 "crash|no|other": "crash · no · not closing", "normal|yes|closing": "normal · yes · closing", "normal|yes|other": "normal · yes · <b>not closing</b>",
                 "normal|no|closing": "normal · no · closing", "normal|no|other": "normal · no · not closing"};
    $("p2fVR").innerHTML = `<thead><tr><th>run</th>${cells.map(c => `<th>${hdr[c]}</th>`).join("")}<th title="verdict yes with reasoning not closing, or verdict no on a crash with reasoning closing">misaligned</th></tr></thead><tbody>${
      D.verdict_reasoning.map(r => `<tr><td>${esc(r.label)}</td>${cells.map(c => `<td class="num">${r.cells[c] || 0}</td>`).join("")}<td class="num">${r.misaligned}</td></tr>`).join("")}</tbody>`;
    $("p2fFacts").innerHTML = `<thead><tr><th>fact</th><th>windows</th><th>n</th><th>always guessing</th>${RK.map(k => `<th>${k}</th>`).join("")}</tr></thead><tbody>${
      D.facts.map(f => { const best = Math.max(...RK.map(k => f[k]));
        return `<tr><td>${esc(f.slot.replace("_", " "))}</td><td>${esc(f.set)}</td><td class="num">${f.n}</td><td class="num">${f2(f.guess_acc)} (${esc(f.guess)})</td>${
        RK.map(k => `<td class="num${f[k] === best ? " hero-metric" : ""}${f[k] < f.guess_acc ? " bad" : ""}">${f2(f[k])}</td>`).join("")}</tr>`; }).join("")}</tbody>`;
    $("p2fText").innerHTML = `<thead><tr><th>run</th><th>BERTScore (floor)</th><th>wrong-video gap</th><th>retrieval@1 of 10</th><th>distinct explanations</th></tr></thead><tbody>${
      D.text.map(t => `<tr><td>${esc(t.label)}</td><td class="num">${f3(t.bert)} (${f3(t.bert_floor)})</td><td class="num">${f3(t.gap_wrong)}</td><td class="num">${f2(t.retrieval)}</td><td class="num">${t.distinct} / ${D.n}</td></tr>`).join("")}</tbody>`;
  }

  /* ----------------------------------------------------------- per-window table */
  const hay = r => r._hay || (r._hay = [r.id, r.t_expl, r.t_out, ...RK.map(k => (r.runs[k].expl || "") + " " + (r.runs[k].out || ""))].join(" ").toLowerCase());
  const passes = r => (!state.group || r.group === state.group) && (state.label === "" || String(r.label) === state.label)
    && (!state.flag || r.runs[state.run][state.flag]) && (!state.crashWrong || r.crash_wrong)
    && (!state.disagree || new Set(RK.map(k => r.runs[k].v)).size > 1) && (!state.q || hay(r).includes(state.q));
  const SORT = {id: r => r.id, group: r => r.group, label: r => r.label, crash_p: r => r.crash_p,
                ...Object.fromEntries(RK.map(k => [k, r => r.runs[k].p]))};
  function sorted(rows){
    const g = SORT[state.sort.key], d = state.sort.dir;
    return rows.slice().sort((a, b) => { const x = g(a), y = g(b); return (x < y ? -1 : x > y ? 1 : 0) * d; });
  }
  const card = c => c ? `<span class="dim">[${esc(c.road_user)} · ${esc(c.side)} · ${esc(c.motion)} · ${esc(c.gap)}]</span>` : "";
  function cellRun(r, k){
    const e = r.runs[k], ok = e.v === r.label;
    const tags = FLAGS.filter(([f]) => e[f]).map(([, lab]) => `<span class="chip neg">${esc(lab)}</span>`).join(" ");
    const facts = e.facts ? `<div class="dim">Facts: ${esc(Object.entries(e.facts).map(([s, v]) => s.replace("_", " ") + " = " + (v || "–")).join("; "))}</div>` : "";
    return `<div class="p2fcell ${ok ? "" : "wrongcell"}"><b>${k}</b> <span class="chip ${e.v ? "pos" : "neg"}">${e.v ? "yes" : "no"}</span> <span class="dim">P ${f2(e.p)}</span> ${tags}
      ${facts}<div class="ptext">${esc(e.expl || "")}</div><div class="ptext dim">→ ${esc(e.out || "")}</div><div>reasoning card ${card(e.card)}</div></div>`;
  }
  function renderBody(){
    const all = sorted(D.rows.filter(passes)), pages = Math.max(1, Math.ceil(all.length / PAGE));
    state.page = Math.min(state.page, pages - 1);
    const part = all.slice(state.page * PAGE, (state.page + 1) * PAGE);
    $("p2fCount").textContent = `${all.length} of ${D.n} windows · page ${state.page + 1} / ${pages}`;
    $("p2fBody").innerHTML = part.map(r => `<tr>
      <td><div class="thumb" data-id="${esc(r.id)}" title="enlarge">${r.grid ? `<img loading="lazy" src="${esc(r.grid)}" alt="">` : ""}</div>
        <div class="dim">${esc(r.id)}</div><div class="dim">${esc(GL[r.group] || r.group)} · <b>${r.label ? "crash" : "normal"}</b></div>
        <div class="dim ${r.crash_wrong ? "bad" : ""}">crash model P ${f2(r.crash_p)}${r.crash_wrong ? " (wrong)" : ""}</div></td>
      <td class="hpdesc"><div class="p2fcell"><b>Teacher</b> <div class="ptext">${esc(r.t_expl)}</div><div class="ptext dim">→ ${esc(r.t_out)}</div><div>card ${card(r.t_card)}</div></div>
        ${RK.map(k => cellRun(r, k)).join("")}</td></tr>`).join("");
    $("p2fBody").querySelectorAll(".thumb").forEach(t => t.addEventListener("click", () => openGrid(t.dataset.id)));
    $("p2fPrev").disabled = state.page === 0;
    $("p2fNext").disabled = state.page >= pages - 1;
    $("p2fHead").querySelectorAll("button[data-sort]").forEach(b => {
      const on = b.dataset.sort === state.sort.key;
      b.classList.toggle("active", on);
      b.textContent = b.dataset.label + (on ? (state.sort.dir > 0 ? " ▲" : " ▼") : "");
    });
  }
  function openGrid(id){
    const r = D.rows.find(x => x.id === id);
    if (!r || !(r.grid_large || r.grid)) return;
    $("gImg").src = r.grid_large || r.grid;
    $("gTitle").textContent = r.id;
    $("gMeta").textContent = `${GL[r.group] || r.group} · ${r.label ? "crash" : "normal"} · crash model P ${f2(r.crash_p)} · ` + RK.map(k => `${k} ${r.runs[k].v ? "yes" : "no"} (${f2(r.runs[k].p)})`).join(" · ");
    $("gridOverlay").classList.add("open");
  }

  function build(){
    built = true;
    renderTables();
    const sel = (id, opts, key) => {
      $(id).innerHTML = opts.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("");
      $(id).addEventListener("change", e => { state[key] = e.target.value; state.page = 0; renderBody(); });
    };
    sel("p2fGroup", [["", "all groups"], ["normal", "normal"], ["tte0.5", "TTE 0.5 s"], ["tte1.0", "TTE 1.0 s"], ["tte1.5", "TTE 1.5 s"]], "group");
    sel("p2fLabel", [["", "crash + normal"], ["1", "crash windows"], ["0", "normal windows"]], "label");
    sel("p2fRun", RK.map(k => [k, `flag of ${k}`]), "run");
    $("p2fRun").value = state.run;
    sel("p2fFlag", [["", "no flag filter"], ...FLAGS], "flag");
    $("p2fCrashWrong").addEventListener("change", e => { state.crashWrong = e.target.checked; state.page = 0; renderBody(); });
    $("p2fDisagree").addEventListener("change", e => { state.disagree = e.target.checked; state.page = 0; renderBody(); });
    $("p2fSearch").addEventListener("input", e => { state.q = e.target.value.trim().toLowerCase(); state.page = 0; renderBody(); });
    $("p2fHead").innerHTML = [["id", "window"], ["group", "group"], ["label", "label"], ["crash_p", "crash model P"], ...RK.map(k => [k, `${k} P(yes)`])]
      .map(([k, l]) => `<button class="pill" data-sort="${k}" data-label="${esc(l)}">${esc(l)}</button>`).join(" ");
    $("p2fHead").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      state.sort = {key: b.dataset.sort, dir: state.sort.key === b.dataset.sort ? -state.sort.dir : 1}; state.page = 0; renderBody();
    }));
    $("p2fPrev").addEventListener("click", () => { state.page--; renderBody(); });
    $("p2fNext").addEventListener("click", () => { state.page++; renderBody(); });
    const close = () => $("gridOverlay").classList.remove("open");
    $("gClose").addEventListener("click", close);
    renderBody();
  }
  return {show(){ if (!built) build(); }};
}
