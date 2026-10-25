"""
Run the whole BioSync Swim software pipeline:   python run_demo.py

  1. create simulated calibration + session logs (skipped if data/ already exists)
  2. align four bands with the jump-clap and filter
  3. learn template + threshold from the calibration session
  4. score every repetition of the test session
  5. write figures, HTML report and JSON to outputs/

To use REAL band recordings later, put band_L_wrist.csv ... in data/session and
data/calibration (same columns) and run again.
"""
import json
import os
import time

from biosync import compare as C
from biosync import report as R
from biosync import simulate as S
from biosync import sync

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
DOCS = os.path.join(ROOT, "docs")


def main():
    if not os.path.isdir(os.path.join(DATA, "session")):
        print("Generating simulated data ...")
        S.generate_demo_data(DATA)
    t_start = time.time()
    bands = S.BANDS

    cal = sync.load_and_align(os.path.join(DATA, "calibration"), bands)
    ses = sync.load_and_align(os.path.join(DATA, "session"), bands)

    cal_cycles = C.segment_cycles(cal["speed"]["L_wrist"])      # L wrist = reference band
    ses_cycles = C.segment_cycles(ses["speed"]["L_wrist"])
    print(f"Repetitions found: calibration={len(cal_cycles)}, session={len(ses_cycles)}")

    cal_windows = {b: C.cut_windows(cal["speed"][b], cal_cycles) for b in bands}
    ses_windows = {b: C.cut_windows(ses["speed"][b], ses_cycles) for b in bands}
    model = C.calibrate(cal_windows)
    scores = C.score_session(ses_windows, model)

    flagged = [(i + 1, b) for i in range(len(ses_cycles)) for b in bands if scores[b][i]["flag"]]
    summary = {"n_reps": len(ses_cycles), "n_flagged": len(flagged), "flagged": flagged,
               "completeness": ses["completeness"]}

    # self-check against the simulator's known ground truth (only possible for simulated data)
    truth_path = os.path.join(DATA, "session", "truth.json")
    if os.path.exists(truth_path):
        truth = {(k + 1, b) for k, b, _ in json.load(open(truth_path))["deviations"]}
        found = set(flagged)
        tp, fp, fn = len(truth & found), len(found - truth), len(truth - found)
        summary["selfcheck"] = f"{tp} of {len(truth)} injected deviations found, {fp} false alarms"

    os.makedirs(DOCS, exist_ok=True)
    p1, p2, p3 = (os.path.join(DOCS, f"screenshot_{i}_{n}.png")
                  for i, n in ((1, "sync"), (2, "filter_dtw"), (3, "dashboard")))
    summary["analysis_seconds"] = time.time() - t_start     # load + align + score
    R.fig_sync(ses, bands, p1)
    R.fig_filter_dtw(ses, ses_cycles, ses_windows, model, p2)
    R.fig_dashboard(ses, bands, ses_cycles, scores, summary, p3)
    summary["report_seconds"] = time.time() - t_start
    R.write_report(OUT, bands, scores, summary, [p3, p1, p2])

    print("Completeness:", {b: f"{v * 100:.1f}%" for b, v in summary["completeness"].items()})
    print("Flagged (rep, band):", flagged)
    print("Self-check:", summary.get("selfcheck", "n/a"))
    print(f"Done in {summary['report_seconds']:.1f} s -> outputs/session_report.html")


if __name__ == "__main__":
    main()
