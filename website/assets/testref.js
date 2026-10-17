/* testref.js - the "Teacher reasoning - test (private)" view of dataset.html (plan 2026-10-17, step 7).

   window.TEST_TEACHER_DATA (build_test_teacher_data.py): summary + one row per private-test window (677) with the boxed sheet, label, TTE, the crash model's P, the
   blind first look, the label-informed second look (with hazard_visible), the merged final reference and its fact card.
   Filters: TTE group, label, blind verdict wrong, has second look, hazard visible, source of the final text, review flag, search. Sort by clicking a header chip.
   A notes column (saved in this browser, Export / Import) for the review of step 8. 30 rows per page. The enlarge overlay (#gridOverlay) is the page's shared one.

   createTestRefView({data, esc}) -> {show()} */
function createTestRefView(deps){
  const D = deps.data, esc = deps.esc, PAGE = 30;
  const $ = id => document.getElementById(id);
  const f2 = v => (v === null || v === undefined) ? "–" : Number(v).toFixed(2);
  const NOTES_LS = "ccp:test-teacher-notes:v1";
  let NOTES = {};
  try { NOTES = JSON.parse(localStorage.getItem(NOTES_LS)) || {}; } catch (e) { NOTES = {}; }
  let saveT = null;
  const save = () => { clearTimeout(saveT); saveT = setTimeout(() => { try { localStorage.setItem(NOTES_LS, JSON.stringify(NOTES)); } catch (e) {} }, 300); };
  const state = {group: "", label: "", flag: "", hv: "", src: "", review: "", q: "", sort: {key: "key", dir: 1}, page: 0};
  let built = false;
  const GL = {"normal": "normal", "tte0.5": "TTE 0.5 s", "tte1": "TTE 1.0 s", "tte1.0": "TTE 1.0 s", "tte1.5": "TTE 1.5 s"};

  const srcOf = r => r.final ? r.final.source : "";
  const flagged = r => !!(NOTES[r.key] && (NOTES[r.key].flag || NOTES[r.key].note));
  const hay = r => r._hay || (r._hay = [r.key, r.blind && r.blind.expl, r.blind && r.blind.pred, r.second && r.second.expl, r.second && r.second.pred, r.final && r.final.expl,
                                         r.second && r.second.agent].join(" ").toLowerCase());
  function passes(r){
    if (state.group && r.group !== state.group) return false;
    if (state.label !== "" && String(r.label) !== state.label) return false;
    if (state.flag === "blind_wrong" && !r.blind_wrong) return false;
    if (state.flag === "second" && !r.second) return false;
    if (state.flag === "no_blind" && r.blind) return false;
    if (state.hv && !(r.second && r.second.hv === state.hv)) return false;
    if (state.src && !srcOf(r).startsWith(state.src)) return false;
    if (state.review === "flagged" && !flagged(r)) return false;
    if (state.review === "unreviewed" && flagged(r)) return false;
    if (state.q && !hay(r).includes(state.q.toLowerCase())) return false;
    return true;
  }
  const SORT = {key: r => r.key, tte: r => r.tte, label: r => r.label, a1: r => (r.a1_p === null ? -1 : r.a1_p), blind: r => (r.blind ? r.blind.risk : -1), second: r => (r.second ? r.second.risk : -1)};
  const sorted = rows => { const g = SORT[state.sort.key], d = state.sort.dir; return rows.slice().sort((a, b) => { const x = g(a), y = g(b); return (x < y ? -1 : x > y ? 1 : 0) * d; }); };

  function renderSummary(){
    const S = D.summary, g = S.groups || {};
    $("trSummary").innerHTML = `<thead><tr><th>what</th><th>count</th></tr></thead><tbody>
      <tr><td>private test windows</td><td class="num">${S.n} (${S.n_crash} crash · ${S.n_normal} normal)</td></tr>
      <tr><td>windows by group</td><td class="num">${Object.entries(g).map(([k, v]) => `${GL[k] || k}: ${v}`).join(" · ")}</td></tr>
      <tr><td>pass 1 (blind first look)</td><td class="num">${S.blind} answers; wrong verdict (needs pass 2) on ${S.blind_wrong}</td></tr>
      <tr><td>pass 2 (second look)</td><td class="num">${S.second} answers${S.second ? "; hazard visible: " + Object.entries(S.hv).map(([k, v]) => `${k} ${v}`).join(" · ") : ""}</td></tr>
      <tr><td>final reference texts</td><td class="num">${Object.keys(S.final || {}).length ? Object.entries(S.final).map(([k, v]) => `${k} ${v}`).join(" · ") : "not merged yet"}</td></tr></tbody>`;
  }

  const chip = (t, cls) => `<span class="chip ${cls}">${esc(t)}</span>`;
  function ansBlock(title, a, label){
    if (!a) return "";
    const ok = (a.v === "yes") === !!label;
    const p2 = (!ok && title.startsWith("Pass 1")) ? chip("pass 1 wrong → needs pass 2", "neg") : "";
    return `<div class="trans ${ok ? "" : "wrongcell"}"><b>${esc(title)}</b> ${chip(a.v, a.v === "yes" ? "pos" : "neg")} <span class="dim">risk ${a.risk === null || a.risk === undefined ? "–" : a.risk}</span> ${p2}
      ${a.hv ? chip("hazard visible: " + a.hv + (a.first ? " (from frame " + a.first + ")" : ""), a.hv === "no" ? "neg" : "pos") : ""}
      ${a.agent ? `<div class="dim">involved: ${esc(a.agent)}</div>` : ""}<div class="ptext">${esc(a.expl || "")}</div><div class="ptext dim">→ ${esc(a.pred || "")}</div>
      ${(a.problems && a.problems.length) ? `<div class="dim bad">flags: ${esc(a.problems.join("; "))}</div>` : ""}</div>`;
  }
  function renderBody(){
    const all = sorted(D.rows.filter(passes)), pages = Math.max(1, Math.ceil(all.length / PAGE));
    state.page = Math.min(state.page, pages - 1);
    const part = all.slice(state.page * PAGE, (state.page + 1) * PAGE);
    $("trCount").textContent = `${all.length} of ${D.rows.length} windows · page ${state.page + 1} / ${pages} · ${Object.values(NOTES).filter(n => n.flag || n.note).length} reviewed`;
    $("trBody").innerHTML = part.map(r => {
      const n = NOTES[r.key] || {};
      const c = r.card ? `<div class="dim">card: ${esc(r.card.road_user)} · ${esc(r.card.side)} · ${esc(r.card.motion)} · ${esc(r.card.gap)}</div>` : "";
      return `<tr><td><div class="thumb" data-id="${esc(r.key)}" title="enlarge">${r.grid ? `<img loading="lazy" src="${esc(r.grid)}" alt="">` : ""}</div>
        <div class="dim">${esc(r.key)}</div><div class="dim">${esc(GL[r.group] || r.group)} · <b>${r.label ? "crash" : "normal"}</b></div>
        <div class="dim">crash model P ${f2(r.a1_p)}${r.box_flag ? " · box " + esc(r.box_flag) : ""}</div></td>
        <td class="hpdesc">${ansBlock("Pass 1 (blind)", r.blind, r.label)}${ansBlock("Pass 2 (second look)", r.second, r.label)}
          ${r.final ? `<div class="trans final"><b>Reference</b> ${chip(r.final.source, "pos")} ${r.final.hv ? chip("hazard " + r.final.hv, r.final.hv === "no" ? "neg" : "pos") : ""}
            <div class="ptext">${esc(r.final.expl)}</div><div class="ptext dim">→ ${esc(r.final.pred)}</div></div>` : ""}${c}</td>
        <td><select class="trflag" data-k="${esc(r.key)}"><option value="">review…</option>${["ok", "wrong text", "hazard not visible", "label problem", "fix needed"]
          .map(o => `<option${n.flag === o ? " selected" : ""}>${o}</option>`).join("")}</select>
          <textarea class="trnote" data-k="${esc(r.key)}" rows="3" placeholder="note / corrected text">${esc(n.note || "")}</textarea></td></tr>`;
    }).join("");
    $("trBody").querySelectorAll(".thumb").forEach(t => t.addEventListener("click", () => openGrid(t.dataset.id)));
    $("trBody").querySelectorAll(".trflag").forEach(s => s.addEventListener("change", e => { (NOTES[s.dataset.k] = NOTES[s.dataset.k] || {}).flag = e.target.value; save(); }));
    $("trBody").querySelectorAll(".trnote").forEach(t => t.addEventListener("input", e => { (NOTES[t.dataset.k] = NOTES[t.dataset.k] || {}).note = e.target.value; save(); }));
    $("trPrev").disabled = state.page === 0;
    $("trNext").disabled = state.page >= pages - 1;
    $("trHead").querySelectorAll("button[data-sort]").forEach(b => {
      const on = b.dataset.sort === state.sort.key;
      b.classList.toggle("active", on);
      b.textContent = b.dataset.label + (on ? (state.sort.dir > 0 ? " ▲" : " ▼") : "");
    });
  }
  function openGrid(id){
    const r = D.rows.find(x => x.key === id);
    if (!r || !r.grid) return;
    $("gImg").src = r.grid;
    $("gTitle").textContent = r.key;
    $("gMeta").textContent = `${GL[r.group] || r.group} · ${r.label ? "crash" : "normal"} · crash model P ${f2(r.a1_p)}`;
    $("gridOverlay").classList.add("open");
  }

  function build(){
    built = true;
    renderSummary();
    const sel = (id, opts, key) => {
      $(id).innerHTML = opts.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("");
      $(id).addEventListener("change", e => { state[key] = e.target.value; state.page = 0; renderBody(); });
    };
    sel("trGroup", [["", "all groups"], ["normal", "normal"], ["tte0.5", "TTE 0.5 s"], ["tte1", "TTE 1.0 s"], ["tte1.5", "TTE 1.5 s"]], "group");
    sel("trLabel", [["", "crash + normal"], ["1", "crash windows"], ["0", "normal windows"]], "label");
    sel("trFlag", [["", "all windows"], ["blind_wrong", "pass 1 wrong (needs pass 2)"], ["second", "has a pass 2"], ["no_blind", "no pass-1 answer yet"]], "flag");
    sel("trHv", [["", "any hazard visibility"], ["yes", "hazard visible: yes"], ["partly", "partly"], ["no", "not visible"]], "hv");
    sel("trSrc", [["", "any reference source"], ["blind", "blind"], ["second_look", "second look"], ["edited", "edited"]], "src");
    sel("trReview", [["", "review: all"], ["flagged", "reviewed"], ["unreviewed", "not reviewed"]], "review");
    $("trSearch").addEventListener("input", e => { state.q = e.target.value.trim(); state.page = 0; renderBody(); });
    $("trHead").innerHTML = [["key", "window"], ["tte", "TTE"], ["label", "label"], ["a1", "crash model P"], ["blind", "blind risk"], ["second", "second-look risk"]]
      .map(([k, l]) => `<button class="pill" data-sort="${k}" data-label="${esc(l)}">${esc(l)}</button>`).join(" ");
    $("trHead").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      state.sort = {key: b.dataset.sort, dir: state.sort.key === b.dataset.sort ? -state.sort.dir : 1}; state.page = 0; renderBody();
    }));
    $("trPrev").addEventListener("click", () => { state.page--; renderBody(); });
    $("trNext").addEventListener("click", () => { state.page++; renderBody(); });
    $("trExport").addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(NOTES, null, 2)], {type: "application/json"}), a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = `ccp-test-review-notes-${new Date().toISOString().slice(0, 10)}.json`; a.click(); URL.revokeObjectURL(a.href);
    });
    $("trImport").addEventListener("click", () => $("trFile").click());
    $("trFile").addEventListener("change", e => {
      const f = e.target.files[0]; if (!f) return;
      const fr = new FileReader();
      fr.onload = () => { try { const inc = JSON.parse(fr.result); let n = 0; for (const k in inc) if (!NOTES[k]) { NOTES[k] = inc[k]; n++; } save(); renderBody(); alert(`Imported ${n} new note(s).`); } catch (er) { alert("Not a valid notes file."); } };
      fr.readAsText(f);
    });
    renderBody();
  }
  return {show(){ if (!built) build(); }};
}
