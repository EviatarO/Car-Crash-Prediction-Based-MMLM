# Dataset Explorer (local website)

Run: double-click `start_website.bat` (starts the local server + opens the browser).
Or manually: `python serve.py`, then open
`http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/index.html`.

Rebuild the data after the xlsx files change: `python build_site_data.py` (writes `site_data.js`).

Requires the sibling project's raw videos at
`Thesis/Data-Centric-Crash-Prediction-Using-3LC-and-MViT/src/Nexar_DataSet/{train,test}/*.mp4`.
Everything runs on 127.0.0.1 only — nothing leaves the machine.
