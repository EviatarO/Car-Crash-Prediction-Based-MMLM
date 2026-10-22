/* ================================================================= results.js
   sortTable(host, cols, rows, opts): a table whose every column sorts on a header click (asc / desc, arrow shown; missing values always last).
     cols: [{key, label, num: bool, fmt(v,row), best: "max"|"min"|undefined, title}]   rows: plain objects
     opts.best = "column": the best value of each `best` column is green (a missing value never wins)
     opts.rowClass(row) -> extra class; opts.cellClass(key,row) -> extra class (e.g. green when a model beats its baseline)
     opts.sort = {key, dir}: the initial order. */
(function (global) {
  const esc = s => String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const missing = v => v === null || v === undefined || (typeof v === "number" && isNaN(v));

  global.sortTable = function (host, cols, rows, opts = {}) {
    let sort = opts.sort || null;
    const bestOf = {};
    for (const c of cols) {
      if (!c.best) continue;
      const vals = rows.map(r => r[c.key]).filter(v => !missing(v));
      if (vals.length) bestOf[c.key] = c.best === "min" ? Math.min(...vals) : Math.max(...vals);
    }
    function render() {
      let rs = rows.slice();
      if (sort) {
        const c = cols.find(x => x.key === sort.key), sgn = sort.dir === "asc" ? 1 : -1;
        rs.sort((a, b) => {
          const x = a[sort.key], y = b[sort.key];
          if (missing(x) && missing(y)) return 0;
          if (missing(x)) return 1;
          if (missing(y)) return -1;
          return (c.num ? x - y : String(x).localeCompare(String(y))) * sgn;
        });
      }
      host.innerHTML = `<table class="ptable"><thead><tr>${cols.map(c => {
          const on = sort && sort.key === c.key;
          return `<th scope="col" class="${c.num ? "num" : ""}${on ? " sorted" : ""}" data-k="${c.key}" tabindex="0" title="${esc(c.title || "sort by " + c.label)}"
                    aria-sort="${on ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}">${c.label}<span class="sarr">${on ? (sort.dir === "asc" ? "▲" : "▼") : "↕"}</span></th>`;
        }).join("")}</tr></thead><tbody>${rs.map(r => `<tr class="${opts.rowClass ? opts.rowClass(r) : ""}">${cols.map(c => {
          const v = r[c.key];
          const best = c.best && !missing(v) && bestOf[c.key] !== undefined && Math.abs(v - bestOf[c.key]) < 1e-9;
          const extra = opts.cellClass ? opts.cellClass(c.key, r) : "";
          return `<td class="${c.num ? "num" : ""}${best ? " best" : ""} ${extra}">${c.fmt ? c.fmt(v, r) : (missing(v) ? "–" : esc(v))}</td>`;
        }).join("")}</tr>`).join("")}</tbody></table>`;
      host.querySelectorAll("th").forEach(th => {
        const go = () => {
          const k = th.dataset.k, c = cols.find(x => x.key === k);
          sort = sort && sort.key === k ? {key: k, dir: sort.dir === "asc" ? "desc" : "asc"} : {key: k, dir: c.num ? "desc" : "asc"};
          render();
        };
        th.addEventListener("click", go);
        th.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
      });
    }
    render();
  };
})(window);
