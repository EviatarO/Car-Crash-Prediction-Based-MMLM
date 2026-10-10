# Teacher reasoning dataset: the chosen flow

*CCP V-JEPA2 Reasoning · updated 2026-10-10 · status: tested on a 51-window pilot and a 30-window rescue test; the full run has not started.*

**Goal:** one natural-language explanation per training window, used as the target text for the student (V-JEPA2 → projector → Qwen3-VL-4B). Each window also gets a target verdict, `label_vis` (section 2).

---

## 1. Pipeline

**Step 1: windows**
- **Pool:** all 1,500 Nexar training videos, minus the 99 crash videos in which no window passes the visibility rule (98 + 00319 from the 18 former `val_e3a` videos). That leaves **1,401 videos × 3 = 4,203 windows**.
- **Window shape:** 16 frames at 1280×720, every 4th frame of the 30 fps video, so about 2 s.
- **Crash videos (651):** windows end 1.5 / 1.0 / 0.5 s before the collision (TTE).
- **Normal videos (750):** windows end 1.5 / 1.0 / 0.5 s before a fake event at the video's midpoint (with small per-video noise), the same protocol as the Nexar test set; all re-cut with it.
- **The 18 former `val_e3a` videos join the pool.** The validation split for training is taken later, by video, from the pooled 1,500.

**Step 2: detection and tracking**
- **Vehicles:** YOLOPv2. **People and two-wheelers:** Grounding-DINO-tiny (person / bicycle / motorcycle). **Tracking:** BoT-SORT with track stitching and an ego-hood filter, over 4.0 s to 0.5 s before the event, so one track ID spans the three windows of a video.

**Step 3: the object the crash model looks at**
- **Model:** our crash model A1 = BADAS-Open (V-JEPA2 ViT-L) + LoRA r16, compress256, AP 0.9128 on the private test, run frozen.
- **Scoring:** A1's crash head pools 2,560 tokens with attention. That attention, weighted toward the latest frames, is summed inside each tracked box.
- **Choice:** the track with the most attention in the video's latest window (TTE 0.5 / mid 0.5). The same track ID is used in the earlier windows when it is visible in at least 4 of their 16 frames, otherwise that window's own top track.
- **Drawing:** a thin red box on that road user in every frame where it is visible.

**Step 4: stage 1, the teacher on every window**
- Gemini 3.8 Flash through OpenRouter, Flex tier, temperature 0.1, 16–48 calls in parallel.
- Input: the 16 native-resolution boxed frames plus the **E3** prompt. The teacher is blind to the label.
- Then a text-only call adds the short prediction sentence to each E3 answer (the explanation is not touched).

**Step 5: stage 2, the flow** (section 4): windows whose E3 verdict does not match the visible-hazard label are re-run with E6, E7 or E7-pre.

---

## 2. Labels: `label` and `label_vis`

Every crash video has `time_of_alert` (when the danger becomes apparent) and `time_of_event` (the collision).
- **Visibility rule:** a crash window is *kept* if it ends at least 0.27 s (8 frames) after `time_of_alert`. Otherwise it is **pre-alert**: by the annotation, nothing announces the collision in that window yet.
- **`label`:** the Nexar event label (1 for every window of a crash video). It stays as it is, for the crash-score evaluation (A1 is frozen).
- **`label_vis`:** the visible-hazard label, used for routing and as the student's verdict target.

| Window | `label` | `label_vis` |
|---|---|---|
| Normal | 0 | 0 |
| Crash, kept by the rule | 1 | 1 |
| Crash, pre-alert | 1 | 0, unless E7-pre finds the hazard already visible (then 1) |

On the website a pre-alert window reads **1 → 0 · pre-alert**, or **1 → 1 · seen early**.

**Counts (4,203 windows):** 1,343 crash windows kept, 610 crash pre-alert windows and 2,250 normal windows. The visibility rule removes the 99 videos with no kept window entirely. Per TTE, the kept crash windows number 651 / 439 / 253 at TTE 0.5 / 1.0 / 1.5 s.

---

## 3. The prompts (what each one asks, in order)

### E3: blind prediction (stage 1, every window)
1. **Role and goal:** a calibrated driving-safety analyst. Will the ego car collide within 3 s after the last frame?
2. **Base rate:** about half the clips end in a collision, so neither answer is the safe default. An object being present or growing is not evidence; judge the motion.
3. **Box:** the boxed road user is the one a risk model focused on. Treat it as the main agent, never mention the box, set `box_ok = false` if it is not on a real road user.
4. **Analysis:** scene context → moving road users → early / middle / late frames → the boxed agent in frames 12–16: is it moving toward the ego lane, is the gap shrinking faster than before? Safe patterns vs conflict patterns, including a car pulling out of parking.
5. **Two readings of equal depth:** a safe reading and a collision reading, then a counterfactual.
6. **Symmetric decision gates:** *yes* if the gap is shrinking and the paths meet, a crossing / merge / pull-out is unresolved, the ego car closes on a stopped object, or a pedestrian moves toward the path; *no* if the late frames show a stable gap, a finished merge or diverging paths.
7. **Explanation:** 25–45 words of natural sentences about this clip. No colours, no times, no outcome words.
8. **Decision last:** `risk_score` 0–100, then `collision` yes / no (yes if the score is ≥ 50).
9. **Prediction, added afterwards** by a text-only call: only the outcome, 4–15 words ("A collision with the truck follows." / "No collision; the road stays clear."), not repeating the explanation. A retry happens once if it is too long or overlaps the explanation.

### E6: anchored second look (a wrong window with a valid correct sibling)
1. **Two clips of the same drive:** clip A is the reference (the nearest valid window E3 got right), clip B is the target, both boxed. The prompt says how many seconds earlier or later B ends.
2. **Reference analysis of clip A, marked verified correct:** E3's verdict and the outcome, plus E3's temporal analysis, its reasoning for that outcome, its summary and risk score.
3. **How to use it:** start from that reasoning, find the same road user and cues in clip B, describe only what B shows, say so if a cue is not visible yet, never mention clip A.
4. **Timing check:** the deciding road user may not be in clip B yet. Judge B only from what it shows, including the ego car's own motion. A low risk is right if nothing alarming is visible; keep the risk high if the ego car's own motion is already dangerous.
5. **Analysis and decision:** E3's steps on clip B, then the explanation, the decision and the outcome-only prediction.
6. **Extra fields:** `same_agent`, `hazard_visible` (yes / partly / no), `first_visible_frame`.

### E7: hindsight (no valid correct window in the video, at its TTE 0.5)
1. **Two clips:** clip B is the target (16 raw frames, no box); clip C is what happened next, up to just past the collision.
2. **The verified outcome in text.**
3. **Hindsight:** use clip C to identify the road user involved; then go back to clip B: is that road user visible, from which frame, which early cues (including ego motion) are already there? Describe only clip B and judge the risk as a driver would at B's last frame.
4. **Analysis, explanation, decision, prediction** as in E3 (the box wording is replaced by "the most relevant road user").
5. **Extra fields:** `involved_agent`, `hazard_visible`, `first_visible_frame`, `early_cues`.

### E7-pre: the same prompt for pre-alert windows
- It adds one note: the collision happens at the end of clip C, but by the annotation the danger becomes recognisable only after clip B; check whether it is already visible in B. If nothing in B announces the collision, a low risk is the right judgement.

---

## 4. The flowchart

![routing](../reports/figures/teacher_prompt_routing_2026-10-10.png)

**Stage 1:** E3 runs on all 3 windows of every video first. A window is routed only after the whole video has its E3 answers, because an anchor needs the sibling answers. The target for the comparison is the window's default `label_vis` (normal 0 · crash kept 1 · pre-alert 0).

1. **E3 verdict matches the target:** keep E3. A pre-alert window where E3 says "no" is therefore kept as it is: no risk is the right description before the hazard appears (e.g. 00480 TTE 1.5).
2. **Pre-alert window where E3 said "yes": E7-pre.** If it says the hazard is visible (yes / partly) and the verdict yes, the target becomes `label_vis` = 1 (seen early); otherwise it stays 0 and the E7-pre explanation describes the window as it is.
3. **Other wrong window (a kept crash window or a normal one):**
   - a **valid correct sibling** exists → **E6**, anchor = the nearest one (larger TTE on a tie). Pre-alert windows never serve as anchors.
   - no valid correct sibling → **E7 on the video's TTE 0.5 window**. If E7 says the hazard is visible and its verdict is right, **E6 chain** on the other wrong valid windows (TTE 1.0, then 1.5, each anchored on the previous answer, tagged E6C); otherwise the windows are flagged "hazard not visible: drop or review".
4. **Label-informed by design:** an anchor's verdict, or E7's stated outcome, reveals the label. Stage-2 answers are judged on whether the text matches the frames, not on accuracy.

**Code:** `r1_teacher_pilot.py` (calls), `r1_teacher_flow.py` (routing, writes `flow_final.jsonl`), `r1_add_prediction.py` (short predictions), `r1_after_frames.py` (clip C frames).

---

## 5. Tests so far

**Pilot (51 windows, 17 videos outside the 1,761 pool):** E3 was right on 36; the 15 wrong windows went to E6 (12), E7 (1, 00997 TTE 0.5) and the chain (2). The explanations were reviewed on the website.

**Rescue test (10 crash videos, 30 windows, 15 of them pre-alert):**
- E3 was right on 11 of the 15 pre-alert windows ("no") and wrong on 4 ("yes").
- E7-pre read those 4 as "no hazard visible" with honest descriptions, e.g. 00217 TTE 1.0: "a van barely appears at the far left edge in the final frame".
- All 4 stay `label_vis` = 0.
- Decisions taken: the pre-alert windows join the dataset with `label_vis` = 0 and E3's (or E7-pre's) honest reasoning; the 99 videos with no kept window stay out.

**Known limit:** where the hazard is not visible, E6 can write an explanation that the frames do not support (00673, 01028 TTE 1.5 in the pilot). E7 (hindsight) handles those windows honestly, but only as the chain start. A pre-alert window where E3 said "no" but the hazard is actually visible keeps `label_vis` 0 and is only spot-checked.

---

## 6. Full run: plan and cost

- **Windows:** `r1_teacher_windows.py` writes the 4,203 windows; the normal windows are re-cut with the midpoint rule.
- **Boxes:** detection, tracking and A1 attention are GPU work; the plan is a RunPod pod (about 4–5 h, $3–5) that pushes the boxed frames to a private HF repo. The teacher calls then run from the PC (OpenRouter Flex), not from the pod.
- **Teacher cost (Flex):** E3 $48, E6 about $14, E7 and the chain about $4, E7-pre about $3, the E3 prediction pass about $5: **≈ $74**, 2–3 h of calls. The E3 failure rate (29 %) comes from 51 windows, so the stage-2 cost is ±30 %.
- **Review output:** a report plus a website view of every stage-2 window and 200 random E3-kept windows.
