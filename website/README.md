# CCP-MMLM website (local)

Run: double-click `start_website.bat` (starts the local server + opens the browser).
Or manually: `python serve.py`, then open
`http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/index.html`.

Three pages, linked via the top nav:
- `index.html` — landing page: live clip showcase, project goal, architecture diagram,
  test-set results table.
- `dataset.html` — the train/test clip browser (sort, filter, search, player).
- `experiments.html` — two sub-views, routed by the URL hash so a report is linkable:
  - `#detail/<arm>` — one arm in full: description, prompt, dataset, architecture
    (recoloured to show what that arm trains), hyperparameters, and results.
  - `#compare` — Cross-Experiment Comparison: pick a shared dataset and up to three
    arms, then a per-clip table (sort/filter/play) and per-horizon correct-prediction bars.

Rebuild the data after the source files change:
- `python build_site_data.py` — clip table (`site_data.js`), after `dataset/*.xlsx` changes.
- `python build_landing_data.py` — showcase + results table (`landing_data.js`), after new
  test scores or captions land under `outputs/`.
- `python build_experiments_data.py` — per-arm reports (`experiments_data.js`), after a
  new run, checkpoint or caption prompt.
- `python build_compare_data.py` — per-clip comparison tables (`compare_data.js`), after
  new per-clip score dumps.

All four builders assert what they expect before writing (row counts, class balance,
train/val split sizes, and agreement between a run's `test_summary.json` and its per-clip
dump). They fail loudly rather than emitting drifted numbers — if one raises, the data
changed and the claim on the page needs rechecking, not the assert loosening.

Shared front-end modules in `assets/`, so no page carries a second copy:
`arch.js` (the architecture SVG, parameterised by which modules are trainable),
`charts.js` (hand-rolled SVG line/ROC/confusion-matrix/bar charts, no external library),
`player.js` (the clip lightbox: segment playback, TTE countdown, alert light, scrubber),
`plexus.js` (hero animation), `site.css` (all shared styles).

Requires the sibling project's raw videos at
`Thesis/Data-Centric-Crash-Prediction-Using-3LC-and-MViT/src/Nexar_DataSet/{train,test}/*.mp4`.
Everything runs on 127.0.0.1 only — nothing leaves the machine.
