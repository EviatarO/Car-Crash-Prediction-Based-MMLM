# Results Website (`MMLM_AI/website/`)

## Goal
Local, offline site to review the dataset and present results — first step of a
multi-stage plan; more stages (full Experiments page) are expected later.

## Structure
3 pages, shared nav/palette/plexus animation in `website/assets/`:
- `index.html` — landing page: random clip showcase w/ V12 caption, project-goal cards,
  hand-drawn architecture SVG, A0/A1/v12 test-set results table.
- `dataset.html` — train/test clip browser: sortable+filterable table, video player.
- `experiments.html` — two hash-routed sub-views:
  - `#detail/<arm>` — one arm in full: description (hypothesis / aim / method), the
    caption prompt, dataset composition, the architecture SVG recoloured to show what
    that arm trains, hyperparameters, and Train + Test results with a
    `All | Compare | TTE 0.5s | 1s | 1.5s` mode selector.
  - `#compare` — Cross-Experiment Comparison: a shared dataset + up to 3 arms, a
    per-clip table (sort/filter/play, capped height with its own scrollbar), a
    free-text **Comments** column on the far right, and a 1x4 bar figure
    (TTE 0.5s / 1s / 1.5s / Negatives) of correct predictions over a dashed
    ground-truth ceiling.

    The bar ceiling is the **whole dataset's** count for that category and does not
    move when you filter — only the solid bars do — so you can see how much of a
    category the current selection still covers. Panel subtitles read
    "53 of 253 clips in filter" to make both numbers explicit.

    Comments live in `localStorage` under `ccp:review-notes:v1`, keyed by dataset +
    the row's `key` (`frames_dir` in the training pools, where one `video_id` spans up
    to three windows and would otherwise collide — `build_compare_data.py` asserts
    these keys are unique). The store, not the DOM, is the source of truth: the tbody
    is rebuilt on every sort/filter, so a note living only in a textarea would be lost.
    Editing a note deliberately does not re-render, which also keeps the caret. Empty
    notes sort last, so sorting the column surfaces everything reviewed so far.
    Export/Import buttons write and merge a JSON file — localStorage is per-browser and
    a "clear site data" would otherwise wipe a review session with no way back.

Shared modules in `website/assets/`, so no page carries a second copy:
`site.css` (all shared styles; font-sizes were scaled +10% via a one-off script, not
hand-tuned — keep the non-round values), `plexus.js` (hero animation),
`arch.js` (`buildArch(host, opts)` — the architecture SVG, parameterised by
`{semantic, loss, state:{module:"frozen"|"train"|"absent"}, note}`; calling it with no
opts reproduces the landing page's original figure byte-for-byte, which is the
regression check), `charts.js` (`lineChart` / `rocChart` / `confusionMatrix` /
`groupedBars` — hand-rolled SVG, no external library, since the site is offline),
`player.js` (`createPlayer()` — the clip lightbox; dataset.html and experiments.html
both use it).

## Data builders (rerun after source data changes — never hand-edit the `.js` outputs)
- `build_site_data.py` → `site_data.js` — train (1500) + test (1344) clip metadata,
  resolved video/thumb paths. Test positives' event times come from
  `dataset/manifests/test_tte_curve_{public,private}_manifest.jsonl`; only 284/672
  test positives are covered (no other source exists) — uncovered rows show a
  "no event time" badge and play with the TN rule.
- `build_landing_data.py` → `landing_data.js` — showcase clips (all 321
  `outputs/a1fail321/Caption_a1fail321_V12.jsonl` windows, joined to `train.xlsx`) +
  the A0/A1/v12 test-set metrics table. **Asserts computed metrics against known-good
  values** (`EXPECTED` dict in the script) — fails loudly rather than silently
  embedding drifted numbers if the score files ever change.
- `build_experiments_data.py` → `experiments_data.js` — the per-arm reports for
  A0/A1/B-v1/B-v2/B-v3/P1/V10/V12. Imports and CALLS the real prompt builders
  (`prompts/PROMPT_SEMSUP_V1{0,2}_*.py::build_prompt`) so a displayed prompt cannot
  drift from the one the captions were generated with.
- `build_compare_data.py` → `compare_data.js` — per-clip rows for the three comparable
  datasets (test677 / pool1761 / a1fail321). Imports `clip_level_split` from
  `build_pool1761_comparison.py` rather than re-deriving the train/val split.

All builders take every metric from `student_training/scripts/metrics_core.py::
metrics_from_arrays` — the function the training pipeline itself uses — so the site
cannot disagree with the run reports by using a different formula or threshold.

## Serving
`serve.py` — stdlib `http.server` + Range-request support (needed for video seeking;
plain `http.server` lacks it). Root = the `Thesis/` parent directory, because the raw
mp4s live in the sibling project
`Thesis/Data-Centric-Crash-Prediction-Using-3LC-and-MViT/src/Nexar_DataSet/{train,test}/*.mp4`.
Launch: double-click `start_website.bat`, or `python serve.py` then open
`http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/index.html`.
Binds 127.0.0.1 only.

## Playback rules (implemented identically in dataset.html's player and index.html's
showcase player)
- Positive w/ known event time: segment = `[event_time − 5s, event_time + 2s]`.
- Negative, or positive missing an event time: segment = `[0, duration/2 + 2s]`.
- TTE readout (left of video): counts down to 0, flips to `+Xs after event` in amber.
- Alert-window light (top-right circle): green iff `time_of_alert < t < time_of_event`,
  else red.
- Speed buttons 0.5×/0.75×/1× (`video.playbackRate`), pause/resume (button + click +
  spacebar), seek scrubber mapped to the segment only (not the full video), ←/→ nudge
  ±0.25s. Dataset table: every column except the thumbnail is sortable (click cycles
  asc→desc→off, nulls always sort last) and filterable (min/max for numeric, dropdown
  for categorical); filters compose with the top search box.

## Architecture SVG (index.html, hand-drawn, not the PNG)
Redrawn from `reports/figures/arch_L3_training_a1fail_2026-08-29.png`; vector-shape
labels copied verbatim from `make_arch_figures_2026-08-22.py::fig_L3`. Deliberate
deviations from that reference (per explicit review): no "patch grid" box (Predictor
branches directly off the trunk's own output, same source point the crash path uses);
teacher caption sits left of SigLIP with a left-to-right arrow (reference stacks them
vertically); the loss is a bold equation line under the diagram, not a boxed node.
`λ` (semantic loss weight) is carried as a `landing_data.js` field
(`semantic_lambda`, currently 0.2), not hardcoded into the SVG — it changed once
already between experiments (0.05 → 0.2).

## Two sources of truth for test metrics (read before "fixing" a number)

Where a run wrote `test_summary.json`, that file is authoritative for **AP/AUC**; the
per-clip `test_results_epNN.jsonl` dump is authoritative for the **confusion matrix**.
They disagree because `semsup_train.py` stores scores as `round(s, 4)`, and the resulting
ties perturb average precision. AUC and the 0.5-threshold confusion matrix are largely
insensitive to those ties, which is why they agree and AP does not.

How far apart depends on how saturated an arm's scores are:

| arm | rows in a tie (of 677) | published AP | AP recomputed from the dump |
|---|---|---|---|
| A1 | 208 | 0.9000 | 0.8986 |
| B-v3 | 409 (101 at exactly 1.0) | 0.8784 | 0.8655 |

`build_experiments_data.py` and `build_landing_data.py` both gate the override: before
using a summary's AP/AUC they assert its `f1`, `recall` and `specificity` match the dump's
to 1e-3. If those disagree, the two files are not the same checkpoint and the override
would be silently wrong. Per-horizon AP is taken from the summary's `per_tte_ap` for the
same reason, so the headline row and the bucket rows quote one source, not two.

**A0's `/2.0` softmax divisor does NOT break comparability at threshold 0.5.**
`softmax(z)[1] > 0.5 ⟺ z₁ > z₀ ⟺ softmax(z/2)[1] > 0.5`. It matters only at other
thresholds, and not at all for AP/AUC.

## What the data genuinely does not contain

Rendered as explicit "not available" notes. Do not interpolate these.

| Gap | Scope |
|---|---|
| accuracy-vs-epoch, AUC-vs-epoch | **every arm** — `epoch_metrics.jsonl` only ever logged `val_ap`. For V10/V12 they are *derived* here from `val_scores_ep*.jsonl` and labelled as derived. |
| train-split per-example scores | **every arm** — so no train ROC and no train confusion matrix exists anywhere |
| per-epoch val scores | pool-1761 arms (A1/B-v1/B-v2/B-v3/P1) predate `--dump-val-scores` |
| test scores | **V10** was never scored on the 677-clip set |
| hyperparameters | **B-v3** — `train_metrics.json` never synced; nothing is reconstructed |
| epochs 1–8 | **B-v3** — the 12-epoch continuation overwrote them; its curve starts at 9 |

**Per-TTE metrics are valid on the test set only.** In both training pools the horizon is
perfectly confounded with the label (every `TTE_*` window is positive, every `MID-*`
negative), so per-bucket AP/AUC is undefined there. Bucketed *counts* stay valid, which is
why the comparison page's bar figure works on all three datasets and the detail page's
mode selector is offered on the test block only.

## Integrity checks the pages reproduce independently

Useful as a smoke test that the join logic is right — both fall out of the pool designs:

- pool1761: **A0 is correct on exactly 1174 of 1761** windows (it is wrong on precisely the
  587 mined failures).
- a1fail321: **A1 is correct on 0 of 61** — every window in that pool is one A1 gets wrong.
- The per-horizon bars sum to the confusion matrix: A1's TP bars are 136+112+72 = 320 = its
  `tp`; the Negatives bar is 216 = its `tn`.

## Caching — the bug that wasted the most time here

`serve.py` had **two `end_headers` methods in the same class**. The later one won silently
(that is just how a Python class body works), so the only `Cache-Control` ever sent was
`max-age=3600` for `.jpg`/`.png`, and CSS/JS/HTML got **no cache header at all**. With only
`Last-Modified` present, browsers apply heuristic freshness and happily serve a stale
stylesheet or a stale `*_data.js` — which renders as a perfectly fine-looking page showing
last week's layout or yesterday's numbers. This was misdiagnosed as a site bug more than
once.

Now there is exactly one `end_headers`: `max-age=3600` for images, `no-store,
must-revalidate` for everything else. Verify with
`curl -sI http://localhost:8765/.../assets/site.css | grep -i cache` — it must say
`no-store`. If you add another `end_headers`, merge it into the existing one.

A browser that cached a file *before* this fix will still hold that stale copy; one hard
refresh (Ctrl+Shift+R) clears it, after which no-store keeps it correct.

## Layout constraint worth knowing

The comparison table's notes column is deliberately **not** `position:sticky`. Sticky on a
table cell in Chrome lands ~95px short of the scroll container's right edge here and
overlays the score columns, which is worse than scrolling. Instead the two caption columns
(~310px, the width hogs) toggle off via the **Captions** pill, which brings the table from
1579px to 1213px — under a typical container width — so the notes column sits on screen
beside the scores with no horizontal scroll at all.

## Known gotchas
- The in-app Browser pane's screenshot tool is unreliable on this page (stale/blank
  frames) because of the continuous rAF loops (plexus + showcase video tick) —
  verify via DOM/JS state inspection (`element.getBoundingClientRect()`,
  `getComputedStyle`, live attribute reads) instead of trusting screenshots; a
  static-HTML export of just the SVG markup into a fresh non-animated tab works when a
  real visual check is needed.
- Muted video autoplay can be silently blocked by the browser (Chrome's per-origin
  Media Engagement Index makes this flaky in testing specifically — real users hitting
  the page cold are the common case). Both players handle this: a visible tap-hint
  overlay appears on rejection, and the first real click/keydown anywhere unlocks and
  retries. A synthetic `element.click()` from a script does NOT count as a user
  gesture for this purpose — testing autoplay-unlock requires a real dispatched input
  (e.g. Claude_Browser's `computer` tool with a `ref`, not `javascript_tool`'s `.click()`).
- Sticky-offset CSS vars (`--navh`, `--topbarh` in `site.css`) are conservative fixed
  estimates sized for the topbar wrapping to 2 lines on narrow viewports; on wide
  viewports this leaves a harmless gap before the sticky `thead` engages. Not a bug,
  not tightened (would need a responsive/JS-measured offset instead of a CSS var).

## Commits (chronological)
`a42ebfc` stage 1 (single-page dataset explorer) · `8503609` stage 2 (3-page split,
sort/filter, scrubber) · `54a962c` README update · `3cf8336` architecture diagram
redraw + 10% font bump + best-AP highlight.

## Next step
- Score **V10** on the 677-clip test set so it stops being the one hole in the
  comparison view: `score_checkpoints_on_test.py` pointed at
  `outputs/a1fail321/results/v10/fold_01/epoch_10/lora_adapter`. Needs the pod.
- Everything under `website/` is committed; the training/analysis scripts this thread
  produced are still uncommitted (see PROJECT_STATE.md).
