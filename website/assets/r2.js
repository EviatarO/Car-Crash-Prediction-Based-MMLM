/* r2.js - the "Reasoning model - runs" view of experiments.html (Run 1, the three follow-ups and the four Session-1 runs B3/B4/B5/B45).

   window.R2_DATA (built by build_r2_data.py) holds the 612 windows once (D.rows: frames, label, crash model, teacher text) and, per run
   (D.runs[i]), its tables and its run-specific window columns (D.runs[i].rows, same order). A run selector switches the run; a comparison
   table with one row per run is always visible. Per-run sections: decision scores, per group, text, training curve, then a paged per-window
   table (50 rows per page) with a group filter and four toggles: paths disagree / reasoning wrong / crash model wrong / generic explanation.
   The enlarge overlay (#gridOverlay) is the one shared with the pilot view.

   createR2View({data, esc}) -> {show()} */
function createR2View(deps){
  const D = deps.data, esc = deps.esc, PAGE = 50;
  const $ = id => document.getElementById(id);
  const f3 = v => (v === null || v === undefined) ? "–" : Number(v).toFixed(3);
  const f2 = v => (v === null || v === undefined) ? "–" : Number(v).toFixed(2);
  const pct = v => (v === null || v === undefined) ? "–" : (100 * v).toFixed(1) + "%";
  const sgn = v => (v > 0 ? "+" : "") + v.toFixed(3);
  const ci = c => `[${sgn(c[0])}, ${sgn(c[1])}]`;
  const GROUP_LABEL = Object.fromEntries(D.groups.map(g => [g.key, g.label]));
  const state = {run: D.runs[0].key, group: "", flags: {disagree: false, reason_wrong: false, crash_wrong: false, generic: false}, page: 0,
                 q: "", sort: {key: null, dir: 1}};
  let built = false, R = D.runs[0], ROWS = [];
  const rowCache = {};
  // shared window fields + the run's own columns
  const runRows = run => rowCache[run.key] || (rowCache[run.key] = D.rows.map((r, i) => Object.assign({}, r, run.rows[i])));
  const counts = () => Object.assign({}, R.counts, D.counts);

  /* ----------------------------------------------------------- per-run result tables */
  function renderDecision(){
    const best = k => Math.max(...R.decision.filter(d => d.key !== "s_teacher").map(d => d[k]));
    $("r2Decision").innerHTML = `<thead><tr><th>score</th><th>what it is</th>
      <th title="rank AUC against the visible-hazard label (label_vis); all 612 windows">AUC vs visible-hazard label</th>
      <th title="average precision against the visible-hazard label">AP</th>
      <th title="pre-alert windows left out, scored against the Nexar event label">AUC no pre-alert vs Nexar label</th>
      <th title="decision at P = 0.5 (log-odds 0)">accuracy @ 0.5</th><th>balanced accuracy @ 0.5</th>
      <th>TP</th><th>FN</th><th>FP</th><th>TN</th></tr></thead><tbody>${R.decision.map(d => {
        const c = d.confusion_at_thr, up = d.key === "s_teacher";
        const hm = k => (!up && d[k] === best(k)) ? " hero-metric" : "";
        return `<tr class="${up ? "dimrow" : ""}"><td>${esc(d.label)}</td><td class="hpdesc" style="min-width:190px">${esc(d.desc)}</td>
          <td class="num${hm("auc_vs_label_vis")}">${f3(d.auc_vs_label_vis)}</td><td class="num${hm("ap_vs_label_vis")}">${f3(d.ap_vs_label_vis)}</td>
          <td class="num${hm("auc_no_prealert_vs_nexar_label")}">${f3(d.auc_no_prealert_vs_nexar_label)}</td>
          <td class="num${hm("acc_at_thr")}">${pct(d.acc_at_thr)}</td><td class="num${hm("balanced_acc_at_thr")}">${pct(d.balanced_acc_at_thr)}</td>
          <td class="num">${c.TP}</td><td class="num">${c.FN}</td><td class="num">${c.FP}</td><td class="num">${c.TN}</td></tr>`;
      }).join("")}</tbody>`;
  }

  // Session-1 runs: S_before read at a threshold chosen on 600 training windows (metrics.json "train_threshold")
  function renderThr(){
    const t = R.train_thr;
    $("r2ThrWrap").hidden = !t;
    if (!t) return;
    $("r2ThrNote").innerHTML = `<b>Decision before reasoning at a train-chosen threshold.</b> S_before is read at P = ${t.p.toFixed(3)} (log-odds ${t.log_odds.toFixed(3)})
      instead of 0.5; the threshold was picked on ${t.n_train} training windows, so the validation windows did not choose it.`;
    const c = t.confusion;
    $("r2Thr").innerHTML = `<thead><tr><th>threshold P</th><th>log-odds</th><th title="windows from the training set used to pick the threshold">chosen on</th>
      <th>balanced acc there</th><th>validation accuracy</th><th>validation balanced accuracy</th><th>TP</th><th>FN</th><th>FP</th><th>TN</th></tr></thead>
      <tbody><tr><td class="num">${t.p.toFixed(3)}</td><td class="num">${t.log_odds.toFixed(3)}</td><td class="num">${t.n_train} train windows</td>
      <td class="num">${pct(t.bacc_train)}</td><td class="num">${pct(t.acc)}</td><td class="num hero-metric">${pct(t.bacc)}</td>
      <td class="num">${c.TP}</td><td class="num">${c.FN}</td><td class="num">${c.FP}</td><td class="num">${c.TN}</td></tr></tbody>`;
  }

  function renderGroups(){
    const cm = (c, ok) => `<td class="num">${c.TP}</td><td class="num">${c.FN}</td><td class="num">${c.FP}</td><td class="num">${c.TN}</td>`;
    const tot = k => R.by_group.reduce((s, g) => s + g[k], 0);
    $("r2Groups").innerHTML = `<thead><tr><th rowspan="2">group</th><th rowspan="2">n</th><th rowspan="2" title="windows with visible-hazard label 1">positives</th>
      <th colspan="5">crash model @ 0.5</th><th colspan="5">S_after @ 0</th><th colspan="2">paired, S_after vs crash model</th></tr>
      <tr><th>TP</th><th>FN</th><th>FP</th><th>TN</th><th>accuracy</th><th>TP</th><th>FN</th><th>FP</th><th>TN</th><th>accuracy</th>
      <th title="windows where the reasoning path is right and the crash model is wrong">rescued</th><th title="windows where the crash model is right and the reasoning path is wrong">lost</th></tr></thead>
      <tbody>${R.by_group.map(g => `<tr><td>${esc(g.label)}</td><td class="num">${g.n}</td><td class="num">${g.n_pos}</td>
        ${cm(g.crash.confusion)}<td class="num hero-metric">${pct(g.crash.acc)}</td>${cm(g.after.confusion)}<td class="num hero-metric">${pct(g.after.acc)}</td>
        <td class="num good">${g.rescue}</td><td class="num bad">${g.lost}</td></tr>`).join("")}
        <tr class="dimrow"><td>all</td><td class="num">${tot("n")}</td><td class="num">${tot("n_pos")}</td><td colspan="5"></td><td colspan="5"></td>
        <td class="num good">${tot("rescue")}</td><td class="num bad">${tot("lost")}</td></tr></tbody>`;
    const p = R.paired, mc = p.mcnemar;
    const head = `<b>Paired, S_after vs crash model</b> (accuracy at the thresholds above, same 612 windows): only S_after right ${mc.only_first_right},
      only the crash model right ${mc.only_second_right}, McNemar p = ${mc.p.toFixed(3)}. AUC difference (S_after minus crash model) ${sgn(p.auc_diff.diff)},
      95% CI ${ci(p.auc_diff.ci95)}. `;
    // Run 1: the reading written when the tab was first built. Other runs: only what their own per-group counts say.
    const tail = R.key === "run1"
      ? "The reasoning path removes false alarms on normal and pre-alert windows and loses crash windows at TTE 1.5, 1.0 and 0.5 s; per group it is a swap of errors, not a uniform gain. Groups with no positives have no AUC."
      : `Per group, rescued minus lost windows: ${R.by_group.map(g => `${esc(g.label)} ${g.rescue - g.lost > 0 ? "+" : ""}${g.rescue - g.lost}`).join(", ")}. Groups with no positives have no AUC.`;
    $("r2Paired").innerHTML = head + tail;
  }

  function renderText(){
    const t = R.text, row = (lab, val, hint) => `<tr><td>${lab}</td><td class="num">${val}</td><td class="hpdesc" style="min-width:320px">${hint}</td></tr>`;
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

  // one training-curve table; tag(c) -> chips for the epoch row, hl(c) -> row highlighted
  function curveTable(id, curve, bestEp, minEp, tag, hl){
    $(id).innerHTML = `<thead><tr><th>epoch</th><th>train loss</th><th title="loss of the teacher explanation on the 612 validation windows">validation loss</th>
      <th title="wrong-video gap with 95% CI">wrong-video gap</th><th>S_before AUC</th><th>S_teacher AUC</th></tr></thead><tbody>${curve.map(c => {
        const isBest = c.epoch === bestEp, isMin = c.epoch === minEp;
        return `<tr class="${hl && isBest ? "hl" : ""}"><td>${c.epoch}${tag(isBest, isMin)}</td><td class="num">${f3(c.train_loss)}</td><td class="num${isMin ? " hero-metric" : ""}">${f3(c.val_loss)}</td>
          <td class="num${isBest ? " hero-metric" : ""}">${f3(c.gap_wrong)} <span class="dim">${ci(c.gap_wrong_ci)}</span></td>
          <td class="num">${f3(c.auc_before)}</td><td class="num">${f3(c.auc_teacher)}</td></tr>`;
      }).join("")}</tbody>`;
  }

  function renderCurve(){
    const two = !!R.curve_p1;            // Session-1 runs: a Phase-1 table and a Phase-2 table
    ["r2Curve1Head", "r2Curve1Wrap", "r2Curve2Head"].forEach(id => { $(id).hidden = !two; });
    const bestTag = R.key === "run1" ? ' <span class="chip pos" title="best.pt: chosen by the wrong-video gap">best.pt</span>'
      : R.key === "opt1" ? ' <span class="chip pos" title="the checkpoint evaluated in this run (Option 1)">evaluated here</span>'
      : ' <span class="chip pos" title="best.pt: the retrains keep the epoch with the lowest validation loss">best.pt</span>';
    const minTag = ' <span class="chip neg" title="lowest validation loss">lowest val loss</span>';
    const tag = two
      ? ((b, m) => b ? ' <span class="chip pos" title="best.pt: the epoch with the lowest validation loss">best.pt</span>' : "")
      : ((b, m) => (b ? bestTag : "") + (m ? minTag : ""));
    if (two) curveTable("r2Curve1", R.curve_p1, R.best_epoch_p1, R.best_epoch_p1, tag, false);
    curveTable("r2Curve", R.curve, R.best_epoch, R.val_min_epoch, tag, R.key === "opt1");
    const last = R.curve[R.curve.length - 1], r1 = D.runs[0];
    $("r2CurveSrc").textContent = two
      ? `This run's own training logs: Phase 1 (${R.curve_p1.length} epochs) and Phase 2 (${R.curve.length} epochs).`
      : R.curve_own
      ? (R.key === "run1" ? `Run 1's own training log (${R.curve.length} epochs).` : `This run's own training log (${R.curve.length} epochs).`)
      : "Option 1 has no training log of its own: this is Run 1's log, and the row of the evaluated checkpoint (epoch 3) is highlighted.";
    if (two){
      const p1 = R.curve_p1, p1f = p1[0], p1b = p1.find(c => c.epoch === R.best_epoch_p1), b2 = R.curve.find(c => c.epoch === R.best_epoch);
      $("r2CurveNote").innerHTML = `<b>Phase 1</b> best.pt = epoch ${R.best_epoch_p1} of ${p1.length} (validation loss ${f3(p1b.val_loss)}, wrong-video gap ${f3(p1f.gap_wrong)} → ${f3(p1b.gap_wrong)}).
        <b>Phase 2</b> best.pt = epoch ${R.best_epoch}, the epoch with the lowest validation loss (${f3(b2.val_loss)})` +
        (last.epoch === R.best_epoch ? "; it is also the last epoch of the log." : `; the last epoch (${last.epoch}) has train loss ${f3(last.train_loss)} and validation loss ${f3(last.val_loss)}.`) +
        ` S_before AUC = the decision read straight after the frames, measured on the 612 validation windows.`;
      return;
    }
    $("r2CurveNote").innerHTML = (R.key === "run1" || R.key === "opt1")
      ? `<b>Run 1: best.pt = epoch ${r1.best_epoch}</b>, chosen by the wrong-video gap (how much the text depends on the frames), while the
      <b>validation loss was lowest at epoch ${r1.val_min_epoch}</b> and the train loss kept falling: from epoch ${r1.val_min_epoch} on the model fits the training text
      slightly better and the held-out text slightly worse.` + (R.key === "opt1" ? ` Option 1 evaluates exactly epoch ${R.best_epoch}.` : "")
      : `<b>best.pt = epoch ${R.best_epoch}</b>, the epoch with the lowest validation loss (${f3(R.curve.find(c => c.epoch === R.best_epoch).val_loss)});
      the last epoch (${last.epoch}) has train loss ${f3(last.train_loss)} and validation loss ${f3(last.val_loss)}.`;
  }

  /* ----------------------------------------------------------- per-window table */
  // search text: window id + teacher and model text, lower-cased once per row
  const hay = r => r._hay || (r._hay = [r.key, r.t_expl, r.t_out, r.m_expl, r.m_out].join(" ").toLowerCase());
  const passes = r => (!state.group || r.group === state.group) && Object.keys(state.flags).every(k => !state.flags[k] || r[k])
    && (!state.q || hay(r).includes(state.q));
  // sortable columns: [state key, header, tooltip, value getter, numeric?]
  const COLS = {
    key: r => r.key, group: r => GROUP_LABEL[r.group] || r.group, label: r => r.label,
    crash_p: r => r.crash_p, p_before: r => r.p_before, p_after: r => r.p_after, verdict: r => r.verdict,
  };
  const NUMERIC = {label: 1, crash_p: 1, p_before: 1, p_after: 1};
  // filter, then sort the filtered set (stable: ties keep the original window order); the pager slices the result
  function visible(){
    const rows = ROWS.filter(passes), k = state.sort.key;
    if (!k) return rows;
    const get = COLS[k], d = state.sort.dir, num = !!NUMERIC[k];
    return rows.map((r, i) => [r, i, get(r)]).sort((a, b) => {
      const x = a[2], y = b[2], nx = x === null || x === undefined, ny = y === null || y === undefined;
      if (nx || ny) return nx === ny ? a[1] - b[1] : (nx ? 1 : -1);   // missing values last in both directions
      const c = num ? x - y : String(x).localeCompare(String(y), undefined, {numeric: true, sensitivity: "base"});
      return c ? c * d : a[1] - b[1];
    }).map(t => t[0]);
  }
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
    return `<tr><td>${g}</td><td class="r2id">${esc(r.key)}</td><td>${esc(GROUP_LABEL[r.group] || r.group)}</td><td>${lab}</td>
      <td class="num">${pBadge(r.crash_p, r.crash_p >= 0.5)}</td><td class="num">${pBadge(r.p_before, r.p_before >= 0.5)}</td><td class="num">${pBadge(r.p_after, r.p_after >= 0.5)}</td>
      <td><span class="pv ${ok(r, w) ? "ok" : "no"}" title="the written Collision: line; it equals the S_after decision">${r.verdict} ${ok(r, w) ? "✓" : "✗"}</span>${r.generic ? ' <span class="pflip" title="the same explanation is written for ' + D.generic_min + ' or more windows">generic</span>' : ""}</td>
      <td class="r2txt"><div class="r2blk"><span class="r2lab">Teacher</span>${answer(r.t_expl, r.t_out)}</div>
        <div class="r2blk model"><span class="r2lab">Model</span>${answer(r.m_expl, r.m_out)}</div></td></tr>`;
  }

  function renderHead(){
    // [sort key | null, text, tooltip]
    const H = [[null, "Frames 4×4 (red box)", ""], ["key", "Window", "window id"], ["group", "Group", ""],
      ["label", "Label", "visible-hazard label (Nexar label → label when they differ)"],
      ["crash_p", "Crash P", "crash model probability of a collision"],
      ["p_before", "P before", "P(yes) = sigmoid(S_before): decision with the reasoning cut out"],
      ["p_after", "P after", "P(yes) = sigmoid(S_after): decision after the model's own reasoning"],
      ["verdict", "Written verdict", "the model's written Collision: answer"],
      [null, "Teacher / model: explanation → outcome", ""]];
    $("r2Head").innerHTML = "<tr>" + H.map(([k, t, tip]) => {
      if (!k) return `<th>${t}</th>`;
      const on = state.sort.key === k, cls = on ? (state.sort.dir > 0 ? "sort-asc" : "sort-desc") : "";
      return `<th class="sortable ${cls}" data-sort="${k}" title="${esc(tip ? tip + " - " : "")}click to sort" aria-sort="${on ? (state.sort.dir > 0 ? "ascending" : "descending") : "none"}">${t}<span class="caret">${on ? (state.sort.dir > 0 ? "▲" : "▼") : "⇅"}</span></th>`;
    }).join("") + "</tr>";
  }

  function renderBody(keepPage){
    const rows = visible();
    if (!keepPage) state.page = 0;
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    state.page = Math.min(state.page, pages - 1);
    const shown = rows.slice(state.page * PAGE, (state.page + 1) * PAGE);
    $("r2Body").innerHTML = shown.map(rowHTML).join("");
    $("r2Count").textContent = `${rows.length} of ${ROWS.length} windows` + (shown.length ? ` · showing ${state.page * PAGE + 1}–${state.page * PAGE + shown.length}` : "");
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
    const r = ROWS.find(x => x.key === key);
    if (!r || !(r.grid_large || r.grid)) return;
    $("gImg").src = r.grid_large || r.grid;
    $("gTitle").textContent = `#${r.video} · ${r.key}`;
    $("gMeta").textContent = `label ${r.label} · ${GROUP_LABEL[r.group] || r.group} · crash model ${r.crash_p.toFixed(2)} · S_before ${r.p_before.toFixed(2)} · S_after ${r.p_after.toFixed(2)} · written ${r.verdict} (${R.label})`;
    $("gridOverlay").classList.add("open");
  }

  /* ----------------------------------------------------------- comparison table + run selector */
  function renderCompare(){
    // [header, tooltip, formatter, field for "best of the runs", direction]
    const C = [
      ["val loss", "loss of the teacher explanation on the 612 validation windows (nats per token); lower = fits the teacher text better", c => f3(c.val_loss), "val_loss", "min"],
      ["perplexity", "exp(val loss)", c => f2(c.ppl), "ppl", "min"],
      ["BERTScore model", "BERTScore F1 (rescaled), model explanation vs teacher explanation", c => f3(c.bert), "bert", "max"],
      ["BERTScore floor", "the same score for the baseline floor", c => f3(c.bert_floor), null, null],
      ["ROUGE-L", "ROUGE-L F1, model vs teacher", c => f3(c.rouge), "rouge", "max"],
      ["road-user match", "the explanation names the same road user as the teacher's", c => pct(c.road_user_match), "road_user_match", "max"],
      ["distinct explanations", "number of different explanation sentences among the 612 windows", c => c.distinct, null, null],
      ["most repeated", "how many windows get the single most repeated explanation", c => c.top_repeat, "top_repeat", "min"],
      ["wrong-video gap [95% CI]", "loss with another video's frames minus loss with the right frames, nats per token; above 0 = the text depends on the frames",
        c => `${sgn(c.gap_wrong)} <span class="dim">${ci(c.gap_wrong_ci)}</span>`, "gap_wrong", "max"],
      ["retrieval @1 of 10", "pick the right video among 10 by explanation likelihood; chance 0.10", c => pct(c.retrieval_top1), "retrieval_top1", "max"],
      ["AUC S_after", "AUC of the decision after the model's own reasoning, vs the visible-hazard label", c => f3(c.auc_after), "auc_after", "max"],
      ["AUC S_before", "AUC of the decision with the reasoning cut out", c => f3(c.auc_before), "auc_before", "max"],
      ["acc S_after @0.5", "accuracy of S_after at P = 0.5", c => pct(c.acc_after), "acc_after", "max"],
      ["bal. acc S_after @0.5", "balanced accuracy of S_after at P = 0.5", c => pct(c.bacc_after), "bacc_after", "max"],
      ["acc S_before @0.5", "accuracy of S_before at P = 0.5", c => pct(c.acc_before), "acc_before", "max"],
      ["bal. acc S_before @0.5", "balanced accuracy of S_before at P = 0.5", c => pct(c.bacc_before), "bacc_before", "max"],
      ["balanced acc before reasoning (train-chosen threshold)", "balanced accuracy of S_before at a threshold chosen on 600 training windows (Session-1 runs only; blank for the older runs)",
        c => c.bacc_before_train_thr === null ? "" : pct(c.bacc_before_train_thr), "bacc_before_train_thr", "max"],
      ["crash model AUC", "the frozen crash model on the same windows, for reference (the same in every run)", c => f3(c.auc_crash), null, null],
    ];
    // fact agreement: share of windows where the explanation names the same type / side / motion / gap as the teacher's (windows where the teacher text states it)
    const FACTS = [["type", "road-user type"], ["side", "side"], ["motion", "motion"], ["gap", "gap"]];
    FACTS.forEach(([f, name]) => C.push([`fact: ${f}`, `fact agreement, ${name}: share of windows where the model's explanation states the same ${name} as the teacher's (among windows where the teacher text states it); majority baseline ${f2(D.fact_baseline[f])}`,
      c => f2(c.fact[f]), `fact_${f}`, "max"]));
    const rows = D.runs.map(r => Object.assign({}, r.compare, Object.fromEntries(FACTS.map(([f]) => [`fact_${f}`, r.compare.fact[f]]))));
    const best = {};
    C.forEach(([, , , k, dir]) => { if (k){ const v = rows.map(c => c[k]).filter(x => x !== null && x !== undefined); best[k] = dir === "max" ? Math.max(...v) : Math.min(...v); } });
    const base = `<tr class="dimrow"><td title="always answering the most common value of the fact">majority baseline</td>${C.map(([, , , k]) =>
      `<td class="num">${k && k.startsWith("fact_") ? f2(D.fact_baseline[k.slice(5)]) : ""}</td>`).join("")}</tr>`;
    $("r2Compare").innerHTML = `<thead><tr><th>run</th>${C.map(([h, t]) => `<th title="${esc(t)}">${h}</th>`).join("")}</tr></thead><tbody>${rows.map(c =>
      `<tr data-run="${esc(c.key)}" class="cmprow"><td>${esc(c.label)}</td>${C.map(([, , f, k]) =>
        `<td class="num${k && c[k] === best[k] ? " hero-metric" : ""}">${f(c)}</td>`).join("")}</tr>`).join("")}${base}</tbody>`;
    $("r2Compare").querySelectorAll("tr[data-run]").forEach(tr => tr.addEventListener("click", () => selectRun(tr.dataset.run)));
    const by = k => rows.find(c => c.key === k), r1 = by("run1"), b3 = by("b3"), p0 = x => (100 * x).toFixed(0) + "%";
    $("r2CompareNote").innerHTML = `<b>Reading (Session 1).</b> The fresh Phase 1 on the Nexar reasoning texts (B3 vs Run 1) removed the template collapse
      (${b3.distinct} vs ${r1.distinct} distinct explanations; most repeated ${b3.top_repeat} vs ${r1.top_repeat}) and made the text depend about 3x more on the video
      (wrong-video gap ${sgn(b3.gap_wrong)} vs ${sgn(r1.gap_wrong)}); side named correctly ${p0(b3.fact.side)} vs ${p0(r1.fact.side)}.<br>
      Extra inputs (attention, layer 11, both) add little on top; road-user type stays at the baseline (${f2(D.fact_baseline.type)}) in every run, matching the probe.<br>
      The decision read before reasoning is still best in Run 1 (AUC ${f3(r1.auc_before)}) of Run 1 and the B runs (Option 2: ${f3(by("opt2").auc_before)}); no run reaches the crash model (AUC ${f3(r1.auc_crash)}).<br>
      <span class="dim">Bold = best of the eight (lowest for loss, perplexity, most repeated; highest for the rest). Thresholded columns are at P = 0.5 unless stated.
      Fact columns = share of windows where the explanation names the same type / side / motion / gap as the teacher's; the last row is the majority baseline
      (type ${f2(D.fact_baseline.type)}, side ${f2(D.fact_baseline.side)}, motion ${f2(D.fact_baseline.motion)}, gap ${f2(D.fact_baseline.gap)}).
      Option 1 is the same training run as Run 1 at an earlier epoch. Click a row or a run button to open that run below; the table scrolls sideways.</span>`;
  }

  // fact probe: balanced accuracy of a small classifier on the pooled tokens, chance in brackets
  function renderProbe(){
    const P = [["last", "last layer"], ["l11", "layer 11"], ["attn", "attention-weighted"]], F = [["type", "road-user type"], ["side", "side"], ["motion", "motion"], ["gap", "gap"]];
    $("r2Probe").innerHTML = `<thead><tr><th>fact</th>${P.map(([, h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${F.map(([f, name]) => {
      const best = Math.max(...P.map(([k]) => D.probe[k][f].bacc));
      return `<tr><td>${name}</td>${P.map(([k]) => { const x = D.probe[k][f];
        return `<td class="num${x.bacc === best ? " hero-metric" : ""}">${f2(x.bacc)} <span class="dim">(${f2(x.chance)})</span></td>`; }).join("")}</tr>`;
    }).join("")}</tbody>`;
  }

  function selectRun(key){
    R = D.runs.find(r => r.key === key) || D.runs[0];
    state.run = R.key;
    ROWS = runRows(R);
    document.querySelectorAll("#r2RunSel [data-run]").forEach(b => b.classList.toggle("active", b.dataset.run === R.key));
    document.querySelectorAll("#r2Compare tr[data-run]").forEach(tr => tr.classList.toggle("hl", tr.dataset.run === R.key));
    $("r2RunDesc").innerHTML = `<b>${esc(R.label)}.</b> ${esc(R.desc)}. Everything below (decision scores, per group, text, training curve, window table) is this run.`;
    document.querySelectorAll("#r2Flags [data-flag]").forEach(b => { b.textContent = `${b.dataset.base} (${counts()[b.dataset.flag]})`; });
    renderDecision(); renderThr(); renderGroups(); renderText(); renderCurve();
    renderBody();
  }

  function build(){
    built = true;
    $("r2Group").innerHTML = '<option value="">All groups</option>' + D.groups.map(g => `<option value="${esc(g.key)}">${esc(g.label)}</option>`).join("");
    $("r2Group").addEventListener("change", e => { state.group = e.target.value; renderBody(); });
    document.querySelectorAll("#r2Flags [data-flag]").forEach(b => {
      b.dataset.base = b.textContent;
      b.addEventListener("click", () => { state.flags[b.dataset.flag] = !state.flags[b.dataset.flag]; b.classList.toggle("active", state.flags[b.dataset.flag]); renderBody(); });
    });
    $("r2Clear").addEventListener("click", () => {
      state.group = ""; $("r2Group").value = ""; state.q = ""; $("r2Search").value = "";
      Object.keys(state.flags).forEach(k => state.flags[k] = false);
      document.querySelectorAll("#r2Flags [data-flag]").forEach(b => b.classList.remove("active"));
      renderBody();
    });
    $("r2Head").addEventListener("click", e => {
      const th = e.target.closest("th[data-sort]");
      if (!th) return;
      const k = th.dataset.sort;
      state.sort = state.sort.key === k ? {key: k, dir: -state.sort.dir} : {key: k, dir: 1};   // 1st click ascending, 2nd descending
      renderHead(); renderBody();
    });
    $("r2Search").addEventListener("input", e => { state.q = e.target.value.trim().toLowerCase(); renderBody(); });
    $("r2Body").addEventListener("click", e => {
      const el = e.target.closest("[data-key]");
      if (el && (el.classList.contains("pplus") || el.classList.contains("pgrid") || e.target.tagName === "IMG")) openGrid(el.dataset.key);
    });
    const closeGrid = () => $("gridOverlay").classList.remove("open");
    $("gClose").addEventListener("click", closeGrid);
    $("gridOverlay").addEventListener("click", e => { if (e.target === $("gridOverlay")) closeGrid(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") closeGrid(); });
    $("r2RunSel").innerHTML = '<span class="rowcount">Run:</span> ' + D.runs.map(r =>
      `<button class="pill" data-run="${esc(r.key)}" title="${esc(r.desc)}">${esc(r.label)}</button>`).join(" ");
    $("r2RunSel").querySelectorAll("[data-run]").forEach(b => b.addEventListener("click", () => selectRun(b.dataset.run)));
    renderCompare(); renderProbe(); renderHead();
    selectRun(state.run);
  }
  return {show(){ if (!built) build(); else renderBody(true); }};
}
