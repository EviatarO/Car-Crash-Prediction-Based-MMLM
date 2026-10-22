/* ================================================================= showcase.js
   "Model in action": plays a dashcam clip around the window the model saw and shows BADAS-JEPA-Reason's own answer for that window (reasoning,
   outcome, P(collision), verdict). Rows come from pres_data.js (only correct answers; see build_pres_data.py). Crash and normal clips are drawn 50/50.
   Segment rule as on the lab site's landing page: [anchor - 5 s, anchor + 2 s], anchor = the collision time (crash clips) or the window end (normal clips).
   When the segment ends, the next clip loads by itself. A thin timeline marks the 2-second window the model saw and its decision point. */
(function (global) {
  const $ = id => document.getElementById(id);

  global.initShowcase = function (rows) {
    const crash = rows.filter(r => r.label === 1), normal = rows.filter(r => r.label === 0);
    const video = $("scVideo"), hint = $("scTapHint");
    let cur = null, raf = null, lastId = null, unlockArmed = false;

    function pick() {
      for (let k = 0; k < 10; k++) {
        const pool = Math.random() < 0.5 ? crash : normal;
        const r = pool[Math.floor(Math.random() * pool.length)];
        if (r.id !== lastId) return r;
      }
      return rows[Math.floor(Math.random() * rows.length)];
    }
    function armUnlock() {
      if (unlockArmed) return;
      unlockArmed = true;
      const retry = () => { attempt(); document.removeEventListener("pointerdown", retry); document.removeEventListener("keydown", retry); };
      document.addEventListener("pointerdown", retry, {once: true});
      document.addEventListener("keydown", retry, {once: true});
    }
    function attempt() {
      const p = video.play();
      if (!p) return;
      p.then(() => { hint.classList.remove("show"); unlockArmed = false; }).catch(() => { hint.classList.add("show"); armUnlock(); });
    }

    function fill(r) {
      $("scReason").textContent = r.reasoning;
      $("scOutcome").textContent = r.outcome ? "→ " + r.outcome : "";
      const pct = Math.round(r.p * 1000) / 10;
      $("scPval").textContent = (pct >= 99.95 ? ">99.9" : pct <= 0.05 ? "<0.1" : pct.toFixed(1)) + "%";
      $("scPbar").style.width = Math.max(1.5, r.p * 100) + "%";
      $("scPbar").className = "pbar-fill " + (r.verdict ? "yes" : "no");
      const v = $("scVerdict");
      v.textContent = r.verdict ? "YES" : "NO";
      v.className = "verdict " + (r.verdict ? "yes" : "no");
      $("scClip").textContent = "#" + r.id;
      $("scHorizon").textContent = r.label ? `${r.tte.toFixed(r.tte % 1 ? 1 : 0)} s before the collision` : "normal driving";
      $("scTruth").textContent = r.label ? "collision" : "no collision";
      $("scCard").classList.remove("decided");
    }
    function load(r) {
      lastId = r.id;
      cur = {r, anchor: r.label ? r.event : r.t_end, segStart: 0, segEnd: null, decided: false};
      fill(r);
      video.src = r.video;
      video.load();
    }
    function next() { load(pick()); }

    video.addEventListener("loadedmetadata", () => {
      if (!cur) return;
      const dur = video.duration;
      cur.segStart = Math.max(0, cur.anchor - 5);
      cur.segEnd = Math.min(dur, cur.anchor + 2);
      const span = cur.segEnd - cur.segStart, pos = t => Math.min(100, Math.max(0, (t - cur.segStart) / span * 100));
      $("tlWin").style.left = pos(cur.r.t_end - 2) + "%";
      $("tlWin").style.width = (pos(cur.r.t_end) - pos(cur.r.t_end - 2)) + "%";
      $("tlDec").style.left = pos(cur.r.t_end) + "%";
      $("tlEvt").style.display = cur.r.label ? "block" : "none";
      if (cur.r.label) $("tlEvt").style.left = pos(cur.r.event) + "%";
      cur.pos = pos;
      video.currentTime = cur.segStart;
      attempt();
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    });
    function checkEnd() {
      if (cur && cur.segEnd !== null && video.currentTime >= cur.segEnd) next();
    }
    video.addEventListener("timeupdate", checkEnd);                 // also fires in a background tab, where rAF is paused
    video.addEventListener("ended", next);

    function tick() {
      if (!cur || cur.segEnd === null) { raf = requestAnimationFrame(tick); return; }
      const t = video.currentTime, r = cur.r;
      $("tlHead").style.left = cur.pos(t) + "%";
      const val = $("scTteVal"), sub = $("scTteSub");
      if (r.label) {
        const tte = r.event - t;
        $("scTteLbl").textContent = "TIME TO COLLISION";
        val.textContent = tte >= 0 ? tte.toFixed(2) + "s" : "+" + (-tte).toFixed(2) + "s";
        val.className = "val" + (tte < 0 ? " after" : tte < 1 ? " warn" : "");
        sub.textContent = tte < 0 ? "after the collision" : "";
      } else {
        const left = r.t_end - t;
        $("scTteLbl").textContent = "NORMAL DRIVING";
        val.textContent = left >= 0 ? left.toFixed(2) + "s" : "+" + (-left).toFixed(2) + "s";
        val.className = "val" + (left < 0 ? " after" : "");
        sub.textContent = left >= 0 ? "to the decision point" : "after the decision point";
      }
      const decided = t >= r.t_end;
      if (decided !== cur.decided) {
        cur.decided = decided;
        $("scDot").classList.toggle("on", decided);
        $("scCard").classList.toggle("decided", decided);
      }
      checkEnd();
      raf = requestAnimationFrame(tick);
    }

    $("shuffleBtn").addEventListener("click", next);
    $("scStage").addEventListener("click", () => { if (video.paused) attempt(); else video.pause(); });
    next();
  };
})(window);
