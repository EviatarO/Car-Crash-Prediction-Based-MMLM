# Presentation website (`website_pres/`)

The clean site for outside viewers (plan `~/.claude/plans/CCP based BADAS/2026-10-20_Plan-Presentation-Website-Home.md`). The lab site `website/` keeps every
experiment; this site shows only what `promoted.json` lists.

**Promotion rule (like merging a branch into main):** new experiments go to `website/` only. When the user asks to promote something, edit `promoted.json`,
run the builder (its asserts check every number against its source file), show the diff and a screenshot, and only then keep the change.

| file | purpose |
|---|---|
| `promoted.json` | the curated list: showcase source, the models in the results table and their score files, the reasoning-metrics sources, the names |
| `build_pres_data.py` | `python website_pres/build_pres_data.py` -> `pres_data.js` (generated, git-ignored) |
| `index.html` | Home: hero, model in action, problem / our model + 3 numbers, architecture, results (vision + reasoning) |
| `assets/site.css`, `assets/plexus.js` | copies of the lab site's style (so the two sites never change each other) |
| `assets/pres.css` | presentation components |
| `assets/showcase.js` | the auto-advancing clip player with the model's own answer |
| `assets/archspec.js`, `assets/archview.js` | the two model diagrams as data + their renderer (toggle, legend, pop-ups with the inner layers) |
| `assets/results.js` | sortable tables with the best value of each column in green |

Open: run `python website/serve.py`, then `http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website_pres/index.html`.
