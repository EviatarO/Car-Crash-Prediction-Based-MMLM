/* r2.js - the "Reasoning model - Run 1" view of experiments.html.

   Result tables (decision scores, per group, text, training curve) are filled from window.R2_DATA (built by build_r2_data.py), then a
   paged per-window table (50 rows per page) with a group filter and four toggles: paths disagree / reasoning wrong / crash model wrong /
   generic explanation. The enlarge overlay (#gridOverlay) is the one shared with the pilot view.

   createR2View({data, esc}) -> {show()} */
function createR2View(deps){
  const D = deps.data, esc = deps.esc, PAGE = 50;
  const $ = id => document.getElementById(id);
  const f3 = v => (v === null || v === undefined) ? "–" : Number(v).toFixed(3);
  const f2 = v => (v === null || v === undefined) ? "–" : Number(v).toFixed(2);
  const pct = v => (v === null || v === undefined) ? "–" : (100 * v).toFixed(1) + "%";
  const sgn = v => (v > 0 ? "+" : "") + v.toFixed(3);
  const ci = c => `[${sgn(c[0])}, ${sgn(c[1])}]`;
  const GROUP_LABEL = Object.fromEntries(D.by_group.map(g => [g.key, g.label]));
  const state = {group: "", flags: {disagree: false, reason_wrong: false, crash_wrong: false, generic: false}, page: 0};
  let built = false;

  function renderDecision(){
    const best = k => Math.max(...D.decision.filter(d => d.key !== "s_teacher").map(d => d[k]));
    $("r2Decision").innerHTML = `<thead><tr><th>score</th><th>what it is</th>
      <th title="rank AUC against the visible-hazard label (label_vis); all 612 windows">AUC vs visible-hazard label</th>
      <th title="average precision against the visible-hazard label">AP</th>
      <th title="pre-alert windows left out, scored against the Nexar event label">AUC no pre-alert vs Nexar label</th>
      <th title="decision at P = 0.5 (log-odds 0)">accuracy @ 0.5</th><th>balanced accuracy @ 0.5</th>
      <th>TP</th><th>FN</th><th>FP</th><th>TN</th></tr></thead><tbody>${D.decision.map(d => {
        const c = d.confusion_at_thr, up = d.key === "s_teacher";
        const hm = k => (!up && d[k] === best(k)) ? " hero-metric" : "";
        return `<tr class="${up ? "dimrow" : ""}"><td>${esc(d.label)}</td><td class="hpdesc" style="min-width:190px">${esc(d.desc)}</td>
          <td class="num${hm("auc_vs_label_vis")}">${f3(d.auc_vs_label_vis)}</td><td class="num${hm("ap_vs_label_vis")}">${f3(d.ap_vs_label_vis)}</td>
          <td class="num${hm("auc_no_prealert_vs_nexar_label")}">${f3(d.auc_no_prealert_vs_nexar_label)}</td>
          <td class="num${hm("acc_at_thr")}">${pct(d.acc_at_thr)}</td><td class="num${hm("balanced_acc_at_thr")}">${pct(d.balanced_acc_at_thr)}</td>
          <td class="num">${c.TP}</td><td class="num">${c.FN}</td><td class="num">${c.FP}</td><td class="num">${c.TN}</td></tr>`;
      }).join("")}</tbody>`;
  }

  function renderGroups(){
    const cm = (c, ok) => `<td class="num">${c.TP}</td><td class="num">${c.FN}</td><td class="num">${c.FP}</td><td class="num">${c.TN}</td>`;
    const tot = k => D.by_group.reduce((s, g) => s + g[k], 0);
    $("r2Groups").innerHTML = `<thead><tr><th rowspan="2">group</th><th rowspan="2">n</th><th rowspan="2" title="windows with visible-hazard label 1">positives</th>
      <th colspan="5">crash model @ 0.5</th><th colspan="5">S_after @ 0</th><th colspan="2">paired, S_after vs crash model</th></tr>
      <tr><th>TP</th><th>FN</th><th>FP</th><th>TN</th><th>accuracy</th><th>TP</th><th>FN</th><th>FP</th><th>TN</th><th>accuracy</th>
      <th title="windows where the reasoning path is right and the crash model is wrong">rescued</th><th title="windows where the crash model is right and the reasoning path is wrong">lost</th></tr></thead>
      <tbody>${D.by_group.map(g => `<tr><td>${esc(g.label)}</td><td class="num">${g.n}</td><td class="num">${g.n_pos}</td>
        ${cm(g.crash.confusion)}<td class="num hero-metric">${pct(g.crash.acc)}</td>${cm(g.after.confusion)}<td class="num hero-metric">${pct(g.after.acc)}</td>
        <td class="num good">${g.rescue}</td><td class="num bad">${g.lost}</td></tr>`).join("")}
        <tr class="dimrow"><td>all</td><td class="num">${tot("n")}</td><td class="num">${tot("n_pos")}</td><td colspan="5"></td><td colspan="5"></td>
        <td class="num good">${tot("rescue")}</td><td class="num bad">${tot("lost")}</td></tr></tbody>`;
    const p = D.paired, mc = p.mcnemar;
    $("r2Paired").innerHTML = `<b>Paired, S_after vs crash model</b> (accuracy at the thresholds above, same 612 windows): only S_after right ${mc.only_first_right},
      only the crash model right ${mc.only_second_right}, McNemar p = ${mc.p.toFixed(3)}. AUC difference (S_after minus crash model) ${sgn(p.auc_diff.diff)},
      95% CI ${ci(p.auc_diff.ci95)}. The reasoning path removes false alarms on normal and pre-alert windows and loses crash windows at TTE 1.5, 1.0 and 0.5 s;
      per group it is a swap of errors, not a uniform gain. Groups with no positives have no AUC.`;
  }

  function renderText(){
    const t = D.text, row = (lab, val, hint) => `<tr><td>${lab}</td><td class="num">${val}</td><td class="hpdesc" style="min-width:320px">${hint}</td></tr>`;
    $("r2Overlap").innerHTML = `<thead><tr><th>text metric, model explanation vs teacher explanation</th><th>model</th><th>floor</th><th>model − floor</th></tr></thead><tbody>${
      t.overlap.map(o => `<tr><td>${esc(o.label)}</td><td class="num hero-metric">${f3(o.model)}</td><td class="num">${f3(o.floor)}</td>
        <td class="num good">${sgn(o.model - o.floor)}</td></tr>`).join("")}</tbody>`;
    $("r2Other").innerHTML = `<thead><tr><th>check</th><th>value</th><th>reading</th></tr></thead><tbody>
      ${row("road-user match", pct(t.road_user_match), "the explanation names the same road user as the teacher's")}
      ${row("format ok", pct(t.format_ok), "Reasoning / Collision / Outcome all present and parseable")}
      ${row("distinct explanations", `${t.distinct} of ${t.n}`, "a low count means many windows get the same sentence (see the “generic explanation” filter below)")}
      ${row("perplexity, after vs Phase 1", `${f2(t.ppl_after)} vs ${f2(t.ppl_before)}`, "teacher text under the trained model vs the model before Phase 2")}
      ${row("wrong-video gap", `${t.gap_wrong.toFixed(3)} <span class="dim">${ci(t.gap_wrong_ci)}</span>`,
            `loss with another video's frames minus loss with the right frames, nats per token; before Phase 2 ${t.gap_wrong_before.toFixed(3)} ${ci(t.gap_wrong_before_ci)}. Above 0 = the text depends on the frames`)}
      ${row("blank-video gap", `${t.gap_blank.toFixed(3)} <span class="dim">${ci(t.gap_blank_ci)}</span>`,
            `loss with blank frames minus loss with the right frames; before Phase 2 ${t.gap_blank_before.toFixed(3)} ${ci(t.gap_blank_before_ci)}`)}
      ${row("retrieval @ 1 of 10", pct(t.retrieval_top1), `pick the right video among 10 by explanation likelihood; chance ${t.retrieval_chance.toFixed(2)}`)}
      </tbody>`;
  }

  function renderCurve(){
    $("r2Curve").innerHTML = `<thead><tr><th>epoch</th><th>train loss</th><th title="loss of the teacher explanation on the 612 validation windows">validation loss</th>
      <th title="wrong-video gap with 95% CI">wrong-video gap</th><th>S_before AUC</th><th>S_teacher AUC</th></tr></thead><tbody>${D.curve.map(c => {
        const tags = (c.epoch === D.best_epoch ? ' <span class="chip pos" title="best.pt: chosen by the wrong-video gap">best.pt</span>' : "")
          + (c.epoch === D.val_min_epoch ? ' <span class="chip neg" title="lowest validation loss">lowest val loss</span>' : "");
        return `<tr><td>${c.epoch}${tags}</td><td class="num">${f3(c.train_loss)}</td><td class="num${c.epoch === D.val_min_epoch ? " hero-metric" : ""}">${f3(c.val_loss)}</td>
          <td class="num${c.epoch === D.best_epoch ? " hero-metric" : ""}">${f3(c.gap_wrong)} <span class="dim">${ci(c.gap_wrong_ci)}</span></td>
          <td class="num">${f3(c.auc_before)}</td><td class="num">${f3(c.auc_teacher)}</td></tr>`;
      }).join("")}</tbody>`;
    $("r2CurveNote").innerHTML = `<b>best.pt = epoch ${D.best_epoch}</b>, chosen by the wrong-video gap (how much the text depends on the frames), while the
      <b>validation loss was lowest at epoch ${D.val_min_epoch}</b> and the train loss kept falling: from epoch ${D.val_min_epoch} on the model fits the training text
      slightly better and the held-out text slightly worse.`;
  }

  /* ----------------------------------------------------------- per-window table */
  const passes = r => (!state.group || r.group === state.group) && Object.keys(state.flags).every(k => !state.flags[k] || r[k]);
  const visible = () => D.rows.filter(passes);
  const ok = (r, yes) => (yes === (r.label === 1));
  const pBadge = (p, yes) => `<span class="pv ${yes ? "ok" : "no"}" title="${yes ? "matches" : "differs from"} the visible-hazard label">${p.toFixed(2)}</span>`;
  const answer = (expl, out) => `<div class="ptext">${esc(expl)}</div><div class="ppred" title="the one-sentence outcome">→ ${esc(out)}</div>`;

  function rowHTML(r){
    const g = r.grid
      ? `<div class="pgrid" data-key="${esc(r.key)}"><img loading="lazy" src="${esc(r.grid)}" alt="${esc(r.key)}"><button class="pplus" title="enlarge" data-key="${esc(r.key)}">＋</button></div>`
      : '<span class="dim">frames pending</span>';
    const lab = r.label === r.label_nexar
      ? `<span class="chip ${r.label === 1 ? "pos" : "neg"}">${r.label}</span>`
      : `<span class="chip pos" title="Nexar label ${r.label_nexar}, visible-hazard label ${r.label}">${r.label_nexar} → ${r.label}</span><div class="dim hvhint">pre-alert</div>`;
    const w = r.verdict === "yes";
    return `<tr><td>${g}</td><td>${esc(r.key)}</td><td>${esc(GROUP_LABEL[r.group] || r.group)}</td><td>${lab}</td>
      <td class="num">${pBadge(r.crash_p, r.crash_p >= 0.5)}</td><td class="num">${pBadge(r.p_before, r.p_before >= 0.5)}</td><td class="num">${pBadge(r.p_after, r.p_after >= 0.5)}</td>
      <td><span class="pv ${ok(r, w) ? "ok" : "no"}" title="the written Collision: line; it equals the S_after decision">${r.verdict} ${ok(r, w) ? "✓" : "✗"}</span>${r.generic ? ' <span class="pflip" title="the same explanation is written for ' + D.generic_min + ' or more windows">generic</span>' : ""}</td>
      <td class="reason">${answer(r.t_expl, r.t_out)}</td><td class="reason">${answer(r.m_expl, r.m_out)}</td></tr>`;
  }

  function renderHead(){
    $("r2Head").innerHTML = `<tr><th>Frames 4×4 (red box)</th><th>Window</th><th>Group</th><th title="visible-hazard label (Nexar label → label when they differ)">Label</th>
      <th title="crash model probability of a collision">crash model P</th><th title="P(yes) = sigmoid(S_before): decision with the reasoning cut out">P(yes) S_before</th>
      <th title="P(yes) = sigmoid(S_after): decision after the model's own reasoning">P(yes) S_after</th><th title="the model's written Collision: answer">Written verdict</th>
      <th>Teacher: explanation → outcome</th><th>Model: explanation → outcome</th></tr>`;
  }

  function renderBody(keepPage){
    const rows = visible();
    if (!keepPage) state.page = 0;
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    state.page = Math.min(state.page, pages - 1);
    const shown = rows.slice(state.page * PAGE, (state.page + 1) * PAGE);
    $("r2Body").innerHTML = shown.map(rowHTML).join("");
    $("r2Count").textContent = `${rows.length} of ${D.rows.length} windows` + (shown.length ? ` · showing ${state.page * PAGE + 1}–${state.page * PAGE + shown.length}` : "");
    ["r2Pager", "r2Pager2"].forEach(pid => {
      const pg = $(pid);
      pg.innerHTML = pages > 1
        ? `<button class="pill" data-pg="first" ${state.page === 0 ? "disabled" : ""}>« first</button> <button class="pill" data-pg="prev" ${state.page === 0 ? "disabled" : ""}>‹ prev</button>
           <span class="rowcount">page ${state.page + 1} / ${pages}</span>
           <button class="pill" data-pg="next" ${state.page >= pages - 1 ? "disabled" : ""}>next ›</button> <button class="pill" data-pg="last" ${state.page >= pages - 1 ? "disabled" : ""}>last »</button>` : "";
      pg.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
        const k = b.dataset.pg;
        state.page = k === "first" ? 0 : k === "last" ? pages - 1 : state.page + (k === "next" ? 1 : -1);
        renderBody(true);
        $("r2Head").closest("table").scrollIntoView({block: "start"});
      }));
    });
  }

  function openGrid(key){
    const r = D.rows.find(x => x.key === key);
    if (!r || !(r.grid_large || r.grid)) return;
    $("gImg").src = r.grid_large || r.grid;
    $("gTitle").textContent = `#${r.video} · ${r.key}`;
    $("gMeta").textContent = `label ${r.label} · ${GROUP_LABEL[r.group] || r.group} · crash model ${r.crash_p.toFixed(2)} · S_before ${r.p_before.toFixed(2)} · S_after ${r.p_after.toFixed(2)} · written ${r.verdict}`;
    $("gridOverlay").classList.add("open");
  }

  function build(){
    built = true;
    $("r2Group").innerHTML = '<option value="">All groups</option>' + D.by_group.map(g => `<option value="${esc(g.key)}">${esc(g.label)}</option>`).join("");
    $("r2Group").addEventListener("change", e => { state.group = e.target.value; renderBody(); });
    document.querySelectorAll("#r2Flags [data-flag]").forEach(b => {
      b.textContent += ` (${D.counts[b.dataset.flag]})`;
      b.addEventListener("click", () => { state.flags[b.dataset.flag] = !state.flags[b.dataset.flag]; b.classList.toggle("active", state.flags[b.dataset.flag]); renderBody(); });
    });
    $("r2Clear").addEventListener("click", () => {
      state.group = ""; $("r2Group").value = "";
      Object.keys(state.flags).forEach(k => state.flags[k] = false);
      document.querySelectorAll("#r2Flags [data-flag]").forEach(b => b.classList.remove("active"));
      renderBody();
    });
    $("r2Body").addEventListener("click", e => {
      const el = e.target.closest("[data-key]");
      if (el && (el.classList.contains("pplus") || el.classList.contains("pgrid") || e.target.tagName === "IMG")) openGrid(el.dataset.key);
    });
    const closeGrid = () => $("gridOverlay").classList.remove("open");
    $("gClose").addEventListener("click", closeGrid);
    $("gridOverlay").addEventListener("click", e => { if (e.target === $("gridOverlay")) closeGrid(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") closeGrid(); });
    renderDecision(); renderGroups(); renderText(); renderCurve(); renderHead();
  }
  return {show(){ if (!built) build(); renderBody(); }};
}
