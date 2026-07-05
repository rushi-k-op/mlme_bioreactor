# MLME Bioreactor Project — Fixed & Rewritten Code

## What was actually wrong (found by analyzing your CSVs)

Your `exploratory_data.csv` lets us directly measure how well each cheap
scale predicts the pilot:

| Scale  | Fit                              | R²    | Cost/run |
|--------|-----------------------------------|-------|----------|
| bench  | pilot ≈ 0.485·bench − 0.255       | 0.998 | 500 EUR  |
| micro  | pilot ≈ 10.61·micro − 0.020       | 0.619 | 10 EUR   |

**Bench is a near-perfect, much cheaper stand-in for pilot. Micro is cheap
but noisy and only loosely related to pilot.** The old `phase3_bo.py` built
its entire surrogate model out of **micro only**, then blind-fired pilot
runs at whatever micro liked best — with no mechanism to correct for
micro's bias. That's the root cause it couldn't reliably beat 14 g/L.

The rewritten `phase3_bo.py` instead pools **all three scales** into one
Gaussian Process by converting every observation into a calibrated
"pilot-equivalent" value (recomputed online as more pilot data arrives),
weighted by how trustworthy that scale is. Bench does the heavy lifting for
exploration; pilot is used periodically to verify the best candidate and
keep the calibration honest; micro is used only for a very cheap initial
wide screening pass.

## File-by-file summary

- **`client.py`** — unchanged.
- **`phase1_exploration.py`** — added the missing per-variable dependence
  plot (`phase1_variable_dependence.png`, mandatory Task 1) and a
  bench/micro-vs-pilot calibration plot (`phase1_scale_calibration.png`).
  Noise/bias analysis (Task 2) kept and extended with the linear
  calibration fit.
- **`phase2_baseline.py`** — non-adaptive LHS baseline (Task 5), fixed to
  spend the full 15000 EUR budget and made reproducible via a seed.
- **`phase3_bo.py`** — full rewrite: multi-fidelity BO loop that picks
  **both** the next recipe and the next scale every iteration (Task 3),
  run across multiple seeds with aggregated reporting (Task 4).
- **`Beat-The-Felix/main.py`** — competition entry point. Reads credentials
  from environment variables (`MLME_USER`, `MLME_PASSWORD`,
  `MLME_BASE_URL`), runs the same multi-fidelity strategy, and prints/saves
  the required per-seed metrics.

## Running it

```bash
python phase1_exploration.py     # -> exploratory_data.csv, exploration_output.txt, *.png
python phase2_baseline.py        # -> baseline_results.csv, baseline_output.txt, *.png
python phase3_bo.py              # -> bo_all_trajectories.csv, bo_summary_per_seed.csv,
                                  #    bo_output.txt, *.png
```

For the competition entry:
```bash
export MLME_USER=group05
export MLME_PASSWORD=xxxxxxxx
python Beat-The-Felix/main.py    # -> beat_the_felix_results.csv
```

Every script writes its console output to a `.txt` log, its numeric
results to `.csv`, and its plots to `.png` automatically — no manual steps
needed.

## Honest caveat

I could not run any of this against the real `mlme26biosim.org` server (no
network access from here), so the numbers above are validated on a local
synthetic mock, not the real simulator. The **logic** (calibration,
GP fitting, scale selection, budget accounting) is verified to run without
errors and to correctly converge to a known optimum in that mock test. How
close you actually land to 14 g/L depends on the true (unknown) response
surface — if the first live run undershoots, the two easiest knobs to turn
in `phase3_bo.py` are `MICRO_SCREEN_BUDGET` and `VERIFY_EVERY` (more bench
exploration before verifying vs. more frequent pilot checks).
