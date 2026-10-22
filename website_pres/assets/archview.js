/* ================================================================= archview.js
   buildArchView(host, spec) - renders ARCH_SPEC (archspec.js): a toggle between the two models, a colour legend, the main chain of blocks with the
   tensor size on every arrow, the training loss (dashed, training only), and a pop-up (native <dialog>) with the inner sub-blocks of the clicked block.
   Plain HTML + CSS (pres.css), so it stays crisp, wraps on narrow screens without side-scrolling, and every block is a real, keyboard-focusable button. */
(function (global) {
  const esc = s => String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const STATE_NAME = {trained: "trained", mixed: "frozen + trained LoRA", frozen: "frozen", fixed: "no weights", io: "input / output"};

  function card(b, idx) {
    return `<button type="button" class="ab ab-${b.state}" data-i="${idx}" aria-haspopup="dialog" title="${esc(STATE_NAME[b.state] || "")} · click for the inner layers">
      <span class="ab-name">${esc(b.name)}</span>${b.line ? `<span class="ab-line">${esc(b.line)}</span>` : ""}
      ${b.inner ? '<span class="ab-more" aria-hidden="true">⊕</span>' : ""}</button>`;
  }

  function chainHTML(path) {
    // rows: a block with `row: true` starts a new row; the row then opens with an arrow carrying the previous block's output size
    const rows = [[]];
    path.chain.forEach((b, i) => { if (b.row && rows[rows.length - 1].length) rows.push([]); rows[rows.length - 1].push(i); });
    return rows.map((idx, ri) => {
      const parts = idx.map((i, k) => {
        const b = path.chain[i];
        const main = b.side
          ? `<div class="ab-stack"><div class="ab ab-side ab-${b.side.state}" role="note"><span class="ab-name">${esc(b.side.name)}</span><span class="ab-line">${esc(b.side.line)}</span></div>
               <div class="ab-down"><span>${esc(b.side.out)}</span></div>${card(b, i)}</div>`
          : card(b, i);
        const arrow = k < idx.length - 1 ? `<div class="ab-arrow"><span>${esc(b.out)}</span></div>` : "";
        return `<div class="ab-cell">${main}</div>${arrow}`;
      }).join("");
      const lead = ri > 0 ? `<div class="ab-arrow ab-lead"><span>${esc(path.chain[idx[0] - 1].out)}</span></div>` : "";
      const hasSide = idx.some(i => path.chain[i].side);
      return `<div class="ab-row${hasSide ? " has-side" : ""}">${lead}${parts}</div>`;
    }).join("");
  }

  function innerHTML(b) {
    const inn = b.inner;
    const rows = inn.blocks.map((s, k) => `
      ${k ? '<div class="sb-arrow" aria-hidden="true">↓</div>' : ""}
      <div class="sb sb-${s.state || "fixed"}"><div class="sb-main"><span class="sb-name">${esc(s.name)}</span>${s.line ? `<span class="sb-line">${esc(s.line)}</span>` : ""}</div>
        <div class="sb-size">${s.in ? `<span class="sb-in">${esc(s.in)}</span><span class="sb-to">→</span>` : ""}${s.out ? `<span class="sb-out">${esc(s.out)}</span>` : ""}</div></div>`).join("");
    return `<div class="dlg-head"><div><div class="dlg-kicker">${esc(STATE_NAME[b.state] || "")}</div><h3>${esc(inn.title || b.name)}</h3></div>
              <button type="button" class="dlg-x" aria-label="Close">×</button></div>
            ${inn.note ? `<p class="dlg-note">${esc(inn.note)}</p>` : ""}<div class="sb-list">${rows}</div>`;
  }

  global.buildArchView = function (host, spec) {
    host.innerHTML = `<div class="arch-toggle" role="tablist">${spec.paths.map((p, i) =>
        `<button type="button" role="tab" class="ctl big arch-tab" data-p="${i}">${esc(p.tag)} · <b>${esc(p.title)}</b></button>`).join("")}</div>
      <div class="arch-legend">${spec.legend.map(([k, n]) => `<span class="lg"><i class="lg-${k}"></i>${esc(n)}</span>`).join("")}</div>
      <p class="arch-caption" id="archCaption"></p>
      <div class="arch-chain" id="archChain"></div>
      <div class="arch-loss" id="archLoss"></div>
      <dialog class="arch-dlg" id="archDlg"><div class="dlg-body" id="archDlgBody"></div></dialog>`;
    const chainEl = host.querySelector("#archChain"), dlg = host.querySelector("#archDlg"), body = host.querySelector("#archDlgBody");
    let cur = 0;
    function show(i) {
      cur = i;
      const p = spec.paths[i];
      host.querySelectorAll(".arch-tab").forEach(t => { const on = +t.dataset.p === i; t.classList.toggle("active", on); t.setAttribute("aria-selected", on); });
      host.querySelector("#archCaption").textContent = p.caption;
      chainEl.innerHTML = chainHTML(p);
      host.querySelector("#archLoss").innerHTML = p.loss ? `<span class="loss-tag">training only</span><b>${esc(p.loss.name)}:</b> ${esc(p.loss.line)}` : "";
    }
    host.querySelectorAll(".arch-tab").forEach(t => t.addEventListener("click", () => show(+t.dataset.p)));
    chainEl.addEventListener("click", e => {
      const btn = e.target.closest(".ab[data-i]");
      if (!btn) return;
      const b = spec.paths[cur].chain[+btn.dataset.i];
      if (!b.inner) return;
      body.innerHTML = innerHTML(b);
      body.querySelector(".dlg-x").addEventListener("click", () => dlg.close());
      dlg.showModal();
      body.querySelector(".dlg-x").focus();
    });
    dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });          // click on the backdrop closes; Esc is native
    show(0);                                                                               // the reasoning model on page load
  };
})(window);
