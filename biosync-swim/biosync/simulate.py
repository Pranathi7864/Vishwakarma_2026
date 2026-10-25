"""
Simulated 4-band IMU logs (stand-in for the ESP32 + MPU-6050 microSD files).

WHY SIMULATE?  The hardware bands are not built yet. The simulator writes CSV files
in the SAME format the firmware will log, so the whole analysis pipeline can be
developed and tested now, and later run unchanged on real recordings.

What is modelled (every number is an ASSUMPTION, not measured data):
  * 4 bands (both wrists, both ankles), 100 Hz, columns: t_ms,ax,ay,az,gx,gy,gz
  * every band starts recording at a different moment (own clock starting at 0)
  * a pre-session "jump-clap" spike seen by all bands at the same real moment
  * repetitions of an exercise cycle with different speeds (+-~15 %)
  * sensor noise, and a few percent of rows lost by slow SD writes
  * optional injected deviations (known ground truth) to test detection
"""
import json
import os

import numpy as np

FS = 100                       # samples per second
DT = 1.0 / FS
BANDS = ["L_wrist", "R_wrist", "L_ankle", "R_ankle"]
PEAK_DPS = {"L_wrist": 220.0, "R_wrist": 220.0, "L_ankle": 120.0, "R_ankle": 120.0}
START_OFFSET_S = {"L_wrist": 0.00, "R_wrist": 0.37, "L_ankle": 0.81, "R_ankle": 0.52}
DROP_FRACTION = {"L_wrist": 0.0, "R_wrist": 0.005, "L_ankle": 0.01, "R_ankle": 0.02}
CLAP_REAL_S = 2.0              # real-world moment of the jump-clap
EXERCISE_START_S = 3.5         # real-world moment the first repetition starts
NOMINAL_REP_S = 2.0
GYRO_NOISE_DPS = 6.0           # per axis
ACC_NOISE_G = 0.02


def cycle_profile(phase, band, deviation=None):
    """Angular-speed shape (0..~1) of ONE repetition; phase runs 0 -> 1."""
    is_wrist = band.endswith("wrist")
    if deviation == "reduced_range":      # compensation: smaller, flatter movement
        return 0.55 * np.sin(np.pi * phase)
    if is_wrist:
        shape = 0.75 * np.sin(np.pi * phase) ** 2 + 0.25 * np.sin(2 * np.pi * phase) ** 2
    else:                                  # ankle: two kicks per cycle
        shape = 0.9 * np.sin(2 * np.pi * phase) ** 2 + 0.1 * np.sin(np.pi * phase) ** 2
    if deviation == "overshoot":          # forceful, asymmetric kick
        shape = 1.9 * shape
    return shape


def make_session(n_reps, seed, deviations=None):
    """Return ({band: array[n,7]}, truth). deviations = {(rep_idx, band): kind}."""
    deviations = deviations or {}
    rng = np.random.default_rng(seed)
    durations = NOMINAL_REP_S * rng.uniform(0.85, 1.20, n_reps)   # speed varies per rep
    amps = rng.normal(1.0, 0.04, n_reps)                          # small strength variation

    t_end = EXERCISE_START_S + durations.sum() + 2.0
    t_fine = np.arange(0, t_end, 0.001)                           # 1 kHz "real world"
    speed = {b: np.zeros_like(t_fine) for b in BANDS}
    t0 = EXERCISE_START_S
    for k, dur in enumerate(durations):
        m = (t_fine >= t0) & (t_fine < t0 + dur)
        phase = (t_fine[m] - t0) / dur
        for b in BANDS:
            speed[b][m] = amps[k] * PEAK_DPS[b] * cycle_profile(phase, b, deviations.get((k, b)))
        t0 += dur

    out = {}
    for b in BANDS:
        n = int((t_end - START_OFFSET_S[b]) / DT)
        t_local = np.arange(n) * DT                  # the band's OWN clock, starts at 0
        t_real = START_OFFSET_S[b] + t_local
        s = np.interp(t_real, t_fine, speed[b])
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)       # unknown band orientation
        gyro = s[:, None] * direction + rng.normal(0, GYRO_NOISE_DPS, (n, 3))
        acc = (np.array([0, 0, 1.0]) + 0.15 * (s / PEAK_DPS[b])[:, None] * np.array([.3, .2, .1])
               + rng.normal(0, ACC_NOISE_G, (n, 3)))
        tau = t_real - CLAP_REAL_S                   # time since the clap
        acc[tau >= 0, 2] += 5.0 * np.exp(-tau[tau >= 0] / 0.03)   # clap spike (g)
        keep = rng.random(n) >= DROP_FRACTION[b]     # rows lost by slow SD writes
        keep[: int(2.5 / DT)] = True                 # (the clap itself stays intact)
        t_ms = np.round(t_local * 1000).astype(int)
        out[b] = np.column_stack([t_ms, acc, gyro])[keep]

    truth = {"start_offsets_s": START_OFFSET_S, "clap_real_s": CLAP_REAL_S,
             "rep_durations_s": durations.round(3).tolist(),
             "deviations": [[k, b, v] for (k, b), v in deviations.items()]}
    return out, truth


def write_session(folder, data, truth):
    os.makedirs(folder, exist_ok=True)
    for b, arr in data.items():
        np.savetxt(os.path.join(folder, f"band_{b}.csv"), arr, delimiter=",",
                   header="t_ms,ax,ay,az,gx,gy,gz", comments="",
                   fmt=["%d"] + ["%.4f"] * 3 + ["%.3f"] * 3)
    with open(os.path.join(folder, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)


# Ground-truth deviations injected into the demo session (0-based repetition index)
DEMO_DEVIATIONS = {(6, "L_wrist"): "reduced_range", (7, "L_wrist"): "reduced_range",
                   (8, "L_wrist"): "reduced_range",
                   (3, "R_ankle"): "overshoot", (4, "R_ankle"): "overshoot"}


def generate_demo_data(root):
    cal, t1 = make_session(12, seed=42)                              # "physio-approved" reps
    write_session(os.path.join(root, "calibration"), cal, t1)
    ses, t2 = make_session(12, seed=7, deviations=DEMO_DEVIATIONS)   # test session
    write_session(os.path.join(root, "session"), ses, t2)
