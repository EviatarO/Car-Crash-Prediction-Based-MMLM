"use strict";
/* plexus.js - connected-dots hero animation, shared by index.html and dataset.html.
   Two particle streams drift toward the centre; when opposing particles meet, a red
   collision pulse ripples out - an abstract nod to the task (crash anticipation), not
   a starfield. Extracted from the stage-1 dataset page so both pages render the exact
   same motion instead of drifting apart via copy-paste. */
function initPlexus(cv){
  if (!cv || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const ctx = cv.getContext("2d");
  let W, H, parts = [], pulses = [];
  const N = 70;

  function resize(){
    W = cv.width = cv.offsetWidth * devicePixelRatio;
    H = cv.height = cv.offsetHeight * devicePixelRatio;
  }
  addEventListener("resize", resize); resize();

  function spawn(side){
    const fromLeft = side !== undefined ? side : Math.random() < .5;
    return {
      x: fromLeft ? Math.random() * W * .25 : W - Math.random() * W * .25,
      y: Math.random() * H,
      vx: (fromLeft ? 1 : -1) * (0.25 + Math.random() * 0.5) * devicePixelRatio,
      vy: (Math.random() - .5) * 0.35 * devicePixelRatio,
      left: fromLeft,
    };
  }
  for (let i = 0; i < N; i++){
    const p = spawn();
    p.x = Math.random() * W;
    parts.push(p);
  }

  function step(){
    ctx.clearRect(0, 0, W, H);
    const linkD = 130 * devicePixelRatio, crashD = 14 * devicePixelRatio;

    for (const p of parts){
      p.x += p.vx; p.y += p.vy;
      if (p.x < -20 || p.x > W + 20 || p.y < -20 || p.y > H + 20)
        Object.assign(p, spawn(p.left));
    }
    for (let i = 0; i < parts.length; i++){
      for (let j = i + 1; j < parts.length; j++){
        const a = parts[i], b = parts[j];
        const dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
        if (d < linkD){
          ctx.strokeStyle = `rgba(0,191,255,${(1 - d / linkD) * .28})`;
          ctx.lineWidth = devicePixelRatio;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
        if (d < crashD && a.left !== b.left && Math.abs(a.x - W / 2) < W * 0.3){
          pulses.push({x:(a.x+b.x)/2, y:(a.y+b.y)/2, r:0});
          Object.assign(a, spawn(a.left)); Object.assign(b, spawn(b.left));
        }
      }
    }
    for (const p of parts){
      ctx.fillStyle = "rgba(0,191,255,.85)";
      ctx.beginPath(); ctx.arc(p.x, p.y, 1.8 * devicePixelRatio, 0, 7); ctx.fill();
    }
    for (let i = pulses.length - 1; i >= 0; i--){
      const u = pulses[i];
      u.r += 2.2 * devicePixelRatio;
      const alpha = Math.max(0, 1 - u.r / (90 * devicePixelRatio));
      if (alpha <= 0){ pulses.splice(i, 1); continue; }
      ctx.strokeStyle = `rgba(255,107,107,${alpha * .8})`;
      ctx.lineWidth = 2 * devicePixelRatio;
      ctx.beginPath(); ctx.arc(u.x, u.y, u.r, 0, 7); ctx.stroke();
    }
    requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
