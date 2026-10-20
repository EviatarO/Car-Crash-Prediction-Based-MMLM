/* aug.js - the "Reasoning dataset - augmented" view of dataset.html (plan 2026-10-14: Phase 1 on the enlarged dataset).

   window.AUG_DATA (built by build_aug_data.py): summary + one row per original window (4,169) with the teacher's text, every rephrased
   version ever written for it (writer claude | gemini; status rephrased_v1..v3 = used in training, spare, judge = rejected by the Sonnet judge,
   auto = rejected by the automatic check, unjudged) with the Opus audit verdict when sampled, and the frame-shifted window copies.
   Filters: split, group, number of used versions, writer, status; "show rejected" toggle; search over id and all texts; 50 rows per page.
   The enlarge overlay (#gridOverlay) is the page's shared one.

   createAugView({data, esc}) -> {show()} */
function createAugView(deps){
  const D = deps.data, esc = deps.esc, PAGE = 50, P = deps.prefix || "aug";
  const $ = id => document.getElementById(id);
  const S = D.summary;
  const state = {split: "", grp: "", nu: "", writer: "", status: "", rejected: false, q: "", page: 0};
  let built = false, ROWS = D.rows;
  const GROUPS = deps.groups || ["normal", "pre_alert", "tte0.5", "tte1.0", "tte1.5"];
  const WRITERS = deps.writers || [["claude", "Claude-written"], ["gemini", "Gemini-written"]];
  const STATUS_LABEL = {used: "used in training", spare: "judge ok, not used (4th+)", judge: "rejected by the judge", auto: "rejected by the automatic check", removed: "removed after the Opus audit (contact word)", unjudged: "not judged"};
  const stKind = st => st.startsWith("rephrased") ? "used" : st;
  const fmt = n => Number(n).toLocaleString("en-US");

  function renderSummary(){
    const t = S.tally, a = S.audit, h = S.good_hist;
    if (S.kind === "extra"){
      $(P + "Summary").innerHTML = `<thead><tr><th>what</th><th>count</th><th class="hpdesc">meaning</th></tr></thead><tbody>
      <tr><td>new windows</td><td class="num">${fmt(S.windows)} (${fmt(S.n_train)} train)</td><td class="hpdesc">crash windows at TTE 0.75 / 1.25 s; the teacher's text (tag <b>teacher</b>) is always kept</td></tr>
      <tr><td>versions written by Gemini</td><td class="num">${fmt(S.written)}</td><td class="hpdesc">3 per window, plus one rewrite round for windows with fewer than 3 accepted versions (identical texts counted once)</td></tr>
      <tr><td>accepted (used)</td><td class="num">${fmt(t.used || 0)}</td><td class="hpdesc">passed the automatic check AND the Sonnet 5.5 judge; tags rephrased_v1..v3. <b>${fmt(S.used_train)}</b> of them are in the TRAIN windows = the training input; the validation windows' versions are not used (validation scores the teacher's text only)</td></tr>
      <tr><td>judge ok, not used (4th+)</td><td class="num">${fmt(t.spare || 0)}</td><td class="hpdesc">a window keeps at most 3 versions</td></tr>
      <tr><td>rejected by the judge</td><td class="num">${fmt(t.judge || 0)}</td><td class="hpdesc">any changed / added / dropped fact or nuance (the judge's reason is shown)</td></tr>
      <tr><td>rejected by the automatic check</td><td class="num">${fmt(t.auto || 0)}</td><td class="hpdesc">length, outcome word, added type / side, polarity, near copy</td></tr>
      <tr><td>removed after the Opus audit</td><td class="num">${fmt(t.removed || 0)}</td><td class="hpdesc">accepted by the judge, but the text states contact ("bumpers meet") that the original does not: it leaks the outcome</td></tr>
      <tr><td>windows with 0 / 1 / 2 / 3 used versions</td><td class="num">${[0, 1, 2, 3].map(k => fmt(h[k] || 0)).join(" / ")}</td><td class="hpdesc">windows with fewer than 3 train with what they have (plus the teacher's text)</td></tr>
      <tr><td>Opus audit, judge-accepted</td><td class="num">${a.acc_ok} ok · ${a.acc_minor} minor · ${a.acc_major} major (of ${a.n_accepted})</td>
          <td class="hpdesc">blind re-check by Opus 5.5 of 300 sampled versions; minor = a nuance ("approaches" -> "enters"), major = a fact a reader would get wrong. Filter "in the Opus audit" shows them</td></tr>
      <tr><td>Opus audit, judge-rejected</td><td class="num">${a.rej_opus_ok} of ${a.n_rejected} judged ok by Opus</td><td class="hpdesc">the judge is stricter than Opus: it costs versions, not quality</td></tr>
    </tbody>`;
      return;
    }
    $(P + "Summary").innerHTML = `<thead><tr><th>what</th><th>count</th><th class="hpdesc">meaning</th></tr></thead><tbody>
      <tr><td>original windows</td><td class="num">${fmt(S.windows)}</td><td class="hpdesc">the teacher's text (tag <b>teacher</b>) is always kept</td></tr>
      <tr><td>shifted window copies</td><td class="num">${fmt(S.shifted)}</td><td class="hpdesc">the same scene, 1-3 source frames earlier (frames are on the pod, not shown)</td></tr>
      <tr><td>rephrased versions used</td><td class="num">${fmt(t.used || 0)}</td><td class="hpdesc">passed the automatic check AND the Sonnet 5.5 judge; tags rephrased_v1..v3</td></tr>
      <tr><td>rejected by the judge</td><td class="num">${fmt(t.judge || 0)}</td><td class="hpdesc">any changed / added / dropped fact or nuance</td></tr>
      <tr><td>rejected by the automatic check</td><td class="num">${fmt(t.auto || 0)}</td><td class="hpdesc">length, outcome word, added type / side, polarity, near copy</td></tr>
      <tr><td>windows with 0 / 1 / 2 / 3 used versions</td><td class="num">${[0, 1, 2, 3].map(k => fmt(h[k] || 0)).join(" / ")}</td><td class="hpdesc">windows with fewer than 3 train with what they have (plus the teacher's text)</td></tr>
      <tr><td>Opus audit, judge-accepted</td><td class="num">${a.acc_ok} ok · ${a.acc_minor} minor · ${a.acc_major} major (of ${a.n_accepted})</td>
          <td class="hpdesc">blind re-check by Opus 5.5; minor = a nuance ("approaches" -> "enters"), major = a fact a reader would get wrong</td></tr>
      <tr><td>Opus audit, judge-rejected</td><td class="num">${a.rej_opus_ok} of ${a.n_rejected} judged ok by Opus</td><td class="hpdesc">the judge is stricter than Opus: it costs versions, not quality</td></tr>
    </tbody>`;
  }

  function chip(group, value, label, active){
    return `<button class="pill${active ? " active" : ""}" data-g="${group}" data-v="${esc(value)}">${esc(label)}</button>`;
  }
  function renderChips(){
    const c = [];
    c.push(chip("split", "", "all splits", state.split === ""), chip("split", "train", "train", state.split === "train"), chip("split", "val", "val", state.split === "val"));
    c.push("<span class='sep'></span>", chip("grp", "", "all groups", state.grp === ""), ...GROUPS.map(g => chip("grp", g, g, state.grp === g)));
    c.push("<span class='sep'></span>", chip("nu", "", "any # used", state.nu === ""), ...[0, 1, 2, 3].map(k => chip("nu", String(k), `${k} used`, state.nu === String(k))));
    c.push("<span class='sep'></span>", chip("writer", "", D.summary.kind === "extra" ? "any round" : "any writer", state.writer === ""), ...WRITERS.map(w => chip("writer", w[0], w[1], state.writer === w[0])));
    c.push("<span class='sep'></span>", chip("status", "", "any status", state.status === ""), ...Object.keys(STATUS_LABEL).map(k => chip("status", k, STATUS_LABEL[k], state.status === k)),
           chip("audit", "1", "in the Opus audit", state.status === "audit"));
    $(P + "Chips").innerHTML = c.join(" ");
    $(P + "Chips").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      if (b.dataset.g === "audit"){ state.status = state.status === "audit" ? "" : "audit"; }
      else state[b.dataset.g] = b.dataset.v;
      if (state.status && state.status !== "used" && state.status !== "audit") state.rejected = true;
      $(P + "ShowRej").checked = state.rejected;
      state.page = 0; renderChips(); renderBody();
    }));
  }

  function filtered(){
    const q = state.q.trim().toLowerCase();
    return ROWS.filter(r => {
      if (state.split && r.split !== state.split) return false;
      if (state.grp && r.grp !== state.grp) return false;
      if (state.nu !== "" && r.nu !== Number(state.nu)) return false;
      if (state.writer && !r.v.some(v => v.w === state.writer)) return false;
      if (state.status === "audit" && !r.v.some(v => v.au)) return false;
      if (state.status && state.status !== "audit" && !r.v.some(v => stKind(v.st) === state.status)) return false;
      if (q && !(r.id.toLowerCase().includes(q) || r.e.toLowerCase().includes(q) || r.v.some(v => v.e.toLowerCase().includes(q)))) return false;
      return true;
    });
  }

  function verRow(v){
    const k = stKind(v.st);
    const tag = k === "used" ? v.st.replace("rephrased_", "rephrased ") : STATUS_LABEL[k];
    const cls = k === "used" ? "good" : (k === "spare" ? "" : "bad");
    const au = v.au ? `<span class="dim">Opus audit: ${esc(v.au)}</span>` : "";
    return `<div class="augver ${k === "used" ? "" : "dimrow"}"><span class="chip ${k === "used" ? "pos" : "neg"}">${esc(tag)}</span>
      <span class="chip neg">${esc(v.w)}</span> <span class="ptext">${esc(v.e)}</span> <span class="ptext dim">→ ${esc(v.o)}</span>
      ${v.why ? `<span class="${cls} dim">✗ ${esc(v.why)}</span>` : ""} ${au}</div>`;
  }

  function renderBody(){
    const rows = filtered(), n = rows.length, pages = Math.max(1, Math.ceil(n / PAGE));
    state.page = Math.min(state.page, pages - 1);
    const part = rows.slice(state.page * PAGE, (state.page + 1) * PAGE);
    $(P + "Count").textContent = `${fmt(n)} of ${fmt(ROWS.length)} windows · page ${state.page + 1} / ${pages}`;
    $(P + "Body").innerHTML = part.map(r => {
      const vs = state.rejected ? r.v : r.v.filter(v => stKind(v.st) === "used");
      const gt = r.pre ? `pre-alert → ${r.lv}` : (r.nl ? "crash" : "normal");
      return `<tr>
        <td><div class="thumb" data-id="${esc(r.id)}" title="enlarge">${r.grid ? `<img loading="lazy" src="${esc(r.grid)}" alt="">` : ""}</div>
            <div class="dim">${esc(r.id)}</div><div class="dim">${esc(r.split)} · ${esc(r.grp)} · ${gt}</div>
            <div class="dim">verdict <b>${r.lv ? "yes" : "no"}</b> · windows: original${r.sh.length ? " + shift " + r.sh.join(", ") : ""}</div></td>
        <td class="hpdesc"><div class="augver"><span class="chip pos">teacher</span> <span class="ptext">${esc(r.e)}</span> <span class="ptext dim">→ ${esc(r.o)}</span></div>
            ${vs.map(verRow).join("")}
            ${!state.rejected && r.v.length > vs.length ? `<div class="dim">${r.v.length - vs.length} rejected / spare version(s) hidden</div>` : ""}</td></tr>`;
    }).join("");
    $(P + "Body").querySelectorAll(".thumb").forEach(t => t.addEventListener("click", () => openGrid(t.dataset.id)));
    $(P + "Prev").disabled = state.page === 0;
    $(P + "Next").disabled = state.page >= pages - 1;
  }

  function openGrid(id){
    const r = ROWS.find(x => x.id === id);
    if (!r || !(r.gridL || r.grid)) return;
    $("gImg").src = r.gridL || r.grid;
    $("gTitle").textContent = r.id;
    $("gMeta").textContent = `${r.split} · ${r.grp} · verdict ${r.lv ? "yes" : "no"} · ${r.nu} rephrased version(s) used`;
    $("gridOverlay").classList.add("open");
  }

  function build(){
    built = true;
    renderSummary(); renderChips();
    $(P + "Search").addEventListener("input", e => { state.q = e.target.value; state.page = 0; renderBody(); });
    $(P + "ShowRej").addEventListener("change", e => { state.rejected = e.target.checked; renderBody(); });
    $(P + "Prev").addEventListener("click", () => { state.page--; renderBody(); window.scrollTo({top: $(P + "Table").offsetTop - 80}); });
    $(P + "Next").addEventListener("click", () => { state.page++; renderBody(); window.scrollTo({top: $(P + "Table").offsetTop - 80}); });
    $(P + "Clear").addEventListener("click", () => {
      Object.assign(state, {split: "", grp: "", nu: "", writer: "", status: "", rejected: false, q: "", page: 0});
      $(P + "Search").value = ""; $(P + "ShowRej").checked = false; renderChips(); renderBody();
    });
    const close = () => $("gridOverlay").classList.remove("open");
    $("gClose").addEventListener("click", close);
    $("gridOverlay").addEventListener("click", e => { if (e.target === $("gridOverlay")) close(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
    renderBody();
  }

  return {show(){ if (!built) build(); }};
}
