# CCP-MMLM website (local)

Run: double-click `start_website.bat` (starts the local server + opens the browser).
Or manually: `python serve.py`, then open
`http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/index.html`.

Three pages, linked via the top nav:
- `index.html` — landing page: live clip showcase, project goal, architecture diagram,
  test-set results table.
- `dataset.html` — the train/test clip browser (sort, filter, search, player).
- `experiments.html` — stub, to be filled in a later stage.

Rebuild the data after the source files change:
- `python build_site_data.py` — clip table (`site_data.js`), after `dataset/*.xlsx` changes.
- `python build_landing_data.py` — showcase + results table (`landing_data.js`), after new
  test scores or captions land under `outputs/`.

Requires the sibling project's raw videos at
`Thesis/Data-Centric-Crash-Prediction-Using-3LC-and-MViT/src/Nexar_DataSet/{train,test}/*.mp4`.
Everything runs on 127.0.0.1 only — nothing leaves the machine.
