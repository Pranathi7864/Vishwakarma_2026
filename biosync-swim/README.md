# BioSync Swim - analysis software prototype

Software half of **BioSync Swim**, a four-band wearable IMU system (both wrists + both ankles)
that logs swimmers' limb movement offline and compares it with physiotherapist-approved
rehabilitation templates. Vishwakarma Awards 2026-27, Stage 2 (Health Tech).

> **Scope / honesty note.** The hardware bands are not built yet. This repository contains the
> complete *analysis pipeline*, tested on **simulated** recordings written in the exact CSV
> format the ESP32 firmware will log. Nothing here is clinical validation and it is not a
> medical device. The simulator's numbers are assumptions, not measurements.

## What it does

| Step | File | What happens |
|---|---|---|
| 0 | `biosync/simulate.py` | Writes 4 band CSVs (`t_ms,ax,ay,az,gx,gy,gz`, 100 Hz): different start times, clap spike, noise, ~0-2 % lost rows, optional injected deviations |
| 1 | `biosync/sync.py` | Detects the jump-clap in each band, shifts every band to "time since clap", resamples to one 10 ms grid, measures **data completeness**, removes the noise offset, applies a zero-phase **4th-order Butterworth low-pass (6 Hz)** |
| 2 | `biosync/compare.py` | Cuts the session into repetitions, builds a template from approved reps, compares each rep with **Dynamic Time Warping** (hand-written, readable), flags reps above `mean + 4 sigma` of normal variation (leave-one-out calibration) |
| 3 | `biosync/report.py` | Produces the dashboard, figures, `session_report.html` and `session_report.json` |

## Run it

```bash
pip install -r requirements.txt
python run_demo.py                                  # full pipeline -> outputs/ and docs/
python -m unittest discover -s tests -t . -v        # 7 tests
```

Python 3.9+. Runtime is a couple of seconds.

## Result on the simulated demo session

* 12 / 12 repetitions segmented in both the calibration and the test session
* 5 of 5 injected deviations flagged (left wrist reps 7-9 reduced range; right ankle reps 4-5
  overshoot), 0 false alarms, although normal reps vary +-15 % in speed (DTW absorbs timing)
* Clap alignment error <= 1 sample (10 ms); data completeness 97.7-100 % per band
* Analysis time about 1 s for a 25 s session (project target: report within 2 min)

Screenshots: `docs/screenshot_1_sync.png`, `docs/screenshot_2_filter_dtw.png`,
`docs/screenshot_3_dashboard.png`.

## Known limitations (to be addressed in Stage 3)

* Simulated signals are cleaner and more regular than real swimming; thresholds must be
  re-learned from real, physiotherapist-labelled reps.
* The 90 % agreement target on the deck is **not** demonstrated here - it needs a real pilot.
* No clock-drift correction between bands over long sessions (one clap at the start only).
* Segmentation assumes a repetitive exercise with a still moment between reps.
* Uses gyro magnitude only (orientation-independent); joint-angle estimation is future work.

## Using real recordings

Put `band_L_wrist.csv`, `band_R_wrist.csv`, `band_L_ankle.csv`, `band_R_ankle.csv` in
`data/calibration/` (approved reps) and `data/session/` (session to review), then run
`python run_demo.py`. Delete the simulated `truth.json` files so the self-check is skipped.

## Reference

Butterworth filter: SciPy `scipy.signal`. DTW: Sakoe & Chiba (1978) band-constrained DTW.
