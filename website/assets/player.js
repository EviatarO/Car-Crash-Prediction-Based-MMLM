/* ================================================================ player.js
   The clip lightbox: segment playback, time-to-event countdown, alert-window
   light, speed control and an in-segment scrubber. Shared by dataset.html and
   experiments.html so the two cannot drift apart.

   Expects the standard overlay markup (see either page): #overlay, #pVideo,
   #pTitle, #pMeta, #tteLbl/#tteVal/#tteSub, #aDot, #endcard, #replayBtn,
   #pauseBtn, #pClose, #scrub, #scrubElapsed, #scrubTotal, #segInfo, and
   optionally #pCaption (a caption card shown under the video when the caller
   passes caption text).

   Usage:
     const player = createPlayer();
     player.open(clip, {title: "#00810", meta: "TRAIN · POSITIVE", caption: "...",
                        mode: "train"});

   `clip` needs: video, video_missing, time_of_event, time_of_alert, target.

   TWO PLAYBACK MODES, passed explicitly rather than inferred from a null field:

     "train"  positives play [event-5s, event+2s] and count down to the event, then
              show "+Xs after". Negatives get the SAME shape around their sampled
              window instead of the event: [anchor-5s, anchor+2s], counting down to
              the anchor and then "+Xs after". Without this a negative just counted
              elapsed time, so there was no way to see which moment the caption and
              the prediction actually refer to.

     "test"   the last 5 seconds of the clip, counting 5.00 -> 0.00, for BOTH classes.
              The test set has no usable event time: test.xlsx carries no
              time_of_event column at all, and the 284 clips that do have one get it
              from extract_tte_curve_test_frames.py, which DERIVES it as
              `total_frames/fps - 0.5` rather than reading an annotation. So the
              readout is labelled time-to-clip-end, which is what it actually
              measures, and there is no "+Xs after" branch because there is no
              post-event footage to show. All test clips are >= 6.57s, so 5s always
              exists.

   The mode is explicit because the old behaviour keyed off `time_of_event === null`,
   which silently lumped "this clip has no event" together with "we have no annotation
   for this clip" - two different situations that need different readouts.
   ================================================================ */
(function (global) {

  /* A negative has no event, but it DOES have a sampled window, and that window is
     what the caption describes and what the model scored. The extractor placed it at
     max(T_FLOOR, duration/2 + offset), offset -10/-4/-8 by bucket
     (semsup_extract_promptbakeoff_frames.py:238-256; T_FLOOR = 2.0). The offset is
     recoverable from the horizon label, so the anchor needs no new stored field -
     `t_seconds` is null for every MID row in the manifests anyway.

     Clip-level views fall back to the plain midpoint: the dataset page lists whole
     clips, and a negative clip has THREE windows (MID-4/-8/-10), so no single one is
     "the" window there. The midpoint is the reference all three offsets are measured
     from, which makes it the honest stand-in. */
  const MID_OFFSET = {"MID-4": -4, "MID-8": -8, "MID-10": -10};
  const T_FLOOR = 2.0;
  global.negAnchor = function negAnchor(duration, horizon){
    const off = MID_OFFSET[horizon] !== undefined ? MID_OFFSET[horizon] : 0;
    return Math.min(duration, Math.max(T_FLOOR, duration / 2 + off));
  };

  global.createPlayer = function createPlayer() {
    const $ = id => document.getElementById(id);
    const overlay = $("overlay"), video = $("pVideo");
    const tteVal = $("tteVal"), tteLbl = $("tteLbl"), tteSub = $("tteSub");
    const aDot = $("aDot"), endcard = $("endcard"), pauseBtn = $("pauseBtn");
    const scrub = $("scrub"), scrubElapsed = $("scrubElapsed"), scrubTotal = $("scrubTotal");
    const capBox = $("pCaption");

    let cur = null;            // {clip, segStart, segEnd}
    let rafId = null;
    let scrubbing = false;     // true while the user is dragging the range input
    let rate = 1;

    function safePlay(){
      const p = video.play();
      if (p) p.then(() => { pauseBtn.textContent = "⏸ pause"; })
              .catch(() => { pauseBtn.textContent = "▶ resume"; });
    }

    function open(clip, meta){
      if (!clip || clip.video_missing) return false;
      meta = meta || {};
      cur = { clip, segStart: 0, segEnd: null, anchor: null,
              mode: meta.mode === "test" ? "test" : "train",
              horizon: meta.horizon || null };
      $("pTitle").textContent = meta.title || ("#" + (clip.id || ""));
      $("pMeta").textContent = meta.meta || "";
      if (capBox){
        capBox.textContent = meta.caption || "";
        capBox.style.display = meta.caption ? "" : "none";
      }
      endcard.classList.remove("show");
      pauseBtn.textContent = "⏸ pause";
      overlay.classList.add("open");
      video.src = clip.video;
      video.load();
      return true;
    }

    video.addEventListener("loadedmetadata", () => {
      if (!cur) return;
      const c = cur.clip, dur = video.duration;
      if (cur.mode === "test"){
        cur.segStart = Math.max(0, dur - 5);
        cur.segEnd   = dur;
      } else if (c.time_of_event !== null && c.time_of_event !== undefined){
        cur.segStart = Math.max(0, c.time_of_event - 5);
        cur.segEnd   = Math.min(dur, c.time_of_event + 2);
      } else {
        cur.anchor   = global.negAnchor(dur, cur.horizon);
        cur.segStart = Math.max(0, cur.anchor - 5);
        cur.segEnd   = Math.min(dur, cur.anchor + 2);
      }
      $("segInfo").textContent =
        `segment ${cur.segStart.toFixed(2)}s → ${cur.segEnd.toFixed(2)}s of ${dur.toFixed(2)}s`;
      scrubTotal.textContent = (cur.segEnd - cur.segStart).toFixed(1) + "s";
      video.currentTime = cur.segStart;
      video.playbackRate = rate;
      safePlay();
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(tick);
    });

    function checkEnd(){
      if (!cur || cur.segEnd === null) return;
      if (video.currentTime >= cur.segEnd && !video.paused){
        video.pause();
        video.currentTime = cur.segEnd;
        endcard.classList.add("show");
        pauseBtn.textContent = "▶ resume";
      }
    }
    video.addEventListener("timeupdate", checkEnd);

    function updateScrubUI(t){
      if (!cur) return;
      const span = cur.segEnd - cur.segStart;
      if (!scrubbing) scrub.value = span > 0 ? (t - cur.segStart) / span : 0;
      scrubElapsed.textContent = Math.max(0, t - cur.segStart).toFixed(1) + "s";
    }

    function tick(){
      if (!cur) return;
      const t = video.currentTime, c = cur.clip;
      checkEnd();
      if (cur.mode === "test"){
        // counts to the end of available footage, NOT to the collision - see the mode
        // note at the top of this file for why no event time is trustworthy here.
        const left = Math.max(0, cur.segEnd - t);
        tteLbl.textContent = "TIME TO CLIP END";
        tteVal.textContent = left.toFixed(2) + "s";
        tteVal.className = "val" + (left < 1 ? " warn" : "");
        tteSub.textContent = "";
      } else if (c.time_of_event !== null && c.time_of_event !== undefined){
        const tte = c.time_of_event - t;
        tteLbl.textContent = "TIME TO EVENT";
        if (tte >= 0){
          tteVal.textContent = tte.toFixed(2) + "s";
          tteVal.className = "val" + (tte < 1 ? " warn" : "");
          tteSub.textContent = "";
        } else {
          tteVal.textContent = "+" + (-tte).toFixed(2) + "s";
          tteVal.className = "val after";
          tteSub.textContent = "after event";
        }
      } else {
        // counts to the sampled window, not to an event - a negative has none
        const left = (cur.anchor === null ? 0 : cur.anchor) - t;
        tteLbl.textContent = "TIME TO WINDOW";
        if (left >= 0){
          tteVal.textContent = left.toFixed(2) + "s";
          tteVal.className = "val" + (left < 1 ? " warn" : "");
          tteSub.textContent = cur.horizon ? cur.horizon : "";
        } else {
          tteVal.textContent = "+" + (-left).toFixed(2) + "s";
          tteVal.className = "val after";
          tteSub.textContent = "after window";
        }
      }
      const inWin = c.time_of_alert !== null && c.time_of_alert !== undefined &&
                    c.time_of_event !== null && c.time_of_event !== undefined &&
                    t > c.time_of_alert && t < c.time_of_event;
      aDot.classList.toggle("on", inWin);
      updateScrubUI(t);
      rafId = requestAnimationFrame(tick);
    }

    function close(){
      overlay.classList.remove("open");
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
      video.pause();
      video.removeAttribute("src");   // releases the file handle
      video.load();
      cur = null;
    }

    function togglePause(){
      if (!cur) return;
      if (video.paused){
        if (endcard.classList.contains("show")) return;
        safePlay();
      } else {
        video.pause(); pauseBtn.textContent = "▶ resume";
      }
    }

    function replay(){
      if (!cur) return;
      endcard.classList.remove("show");
      video.currentTime = cur.segStart;
      safePlay();
    }

    /* seek to a fraction [0,1] of the reviewed segment - shared by the scrubber and
       the arrow-key nudges. Backing out of the ended state (dragging left, or
       pressing Left) clears the end-card so playback can resume without a full
       replay, per spec. */
    function seekToFraction(frac){
      if (!cur) return;
      frac = Math.min(1, Math.max(0, frac));
      const t = cur.segStart + frac * (cur.segEnd - cur.segStart);
      video.currentTime = t;
      updateScrubUI(t);
      if (t < cur.segEnd) endcard.classList.remove("show");
    }
    function nudge(deltaS){
      if (!cur) return;
      const span = cur.segEnd - cur.segStart;
      seekToFraction(span > 0 ? (video.currentTime - cur.segStart + deltaS) / span : 0);
    }

    /* wiring */
    $("pClose").addEventListener("click", close);
    overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
    video.addEventListener("click", togglePause);
    pauseBtn.addEventListener("click", togglePause);
    $("replayBtn").addEventListener("click", replay);
    document.querySelectorAll(".spd").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".spd").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      rate = parseFloat(b.dataset.rate);
      video.playbackRate = rate;
    }));
    scrub.addEventListener("mousedown", () => scrubbing = true);
    scrub.addEventListener("touchstart", () => scrubbing = true);
    scrub.addEventListener("input", () => seekToFraction(parseFloat(scrub.value)));
    scrub.addEventListener("change", () => { scrubbing = false; });
    document.addEventListener("keydown", e => {
      if (!overlay.classList.contains("open")) return;
      if (e.key === "Escape") close();
      if (e.key === " "){ e.preventDefault(); togglePause(); }
      if (e.key === "ArrowLeft"){ e.preventDefault(); nudge(-0.25); }
      if (e.key === "ArrowRight"){ e.preventDefault(); nudge(0.25); }
    });

    return {open, close, isOpen: () => overlay.classList.contains("open")};
  };
})(window);
