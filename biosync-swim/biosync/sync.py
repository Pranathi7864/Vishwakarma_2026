"""
Step 1 of the pipeline: load each band's CSV, find the jump-clap in every band,
and put all four bands on ONE shared, evenly spaced time axis.

Each band logs on its own clock (t_ms starts at 0 when THAT band is switched on),
so raw timestamps of different bands cannot be compared directly. The clap is one
physical event seen by all bands, so "time since clap" is a shared clock.
"""
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 100                      # nominal sampling rate (Hz)
DT = 1.0 / FS
CLAP_THRESHOLD_G = 2.0        # clap = acceleration magnitude jumps > 2 g from rest
ANALYSIS_DELAY_S = 1.0        # skip the first second after the clap (spike decay)


def load_band_csv(path):
    """Read one band file -> (t_seconds, acc[n,3] in g, gyro[n,3] in deg/s)."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:, 0] / 1000.0, arr[:, 1:4], arr[:, 4:7]


def detect_clap(t, acc, threshold_g=CLAP_THRESHOLD_G):
    """Time (band clock) of the first sample where |acc| deviates strongly from rest."""
    mag = np.linalg.norm(acc, axis=1)
    deviation = np.abs(mag - np.median(mag))     # median ~ 1 g (gravity) at rest
    hits = np.flatnonzero(deviation > threshold_g)
    if hits.size == 0:
        raise ValueError("No jump-clap found - repeat the session with a clap at the start.")
    return t[hits[0]]


def lowpass(x, fs=FS, cutoff_hz=6.0, order=4):
    """Zero-phase Butterworth low-pass. Swim/rehab limb motion lives below ~6 Hz,
    so higher frequencies are treated as sensor noise.  sosfiltfilt runs the filter
    forwards then backwards, which cancels the phase delay (no time shift)."""
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def load_and_align(folder, bands):
    """Load all bands in `folder` and return one aligned record.

    Returns dict with:
      t           shared time axis (s, 0 = clap), uniform 10 ms spacing
      speed[b]    gyro magnitude (deg/s), baseline-removed and low-pass filtered
      speed_raw[b] gyro magnitude before filtering (for plots)
      clap_local[b] clap time on each band's own clock
      completeness[b] fraction of expected samples that were really recorded
      acc_dev[b]  |acc| deviation (for plotting the clap)
    """
    raw, clap = {}, {}
    for b in bands:
        t, acc, gyro = load_band_csv(os.path.join(folder, f"band_{b}.csv"))
        raw[b] = (t, acc, gyro)
        clap[b] = detect_clap(t, acc)

    # common window in "time since clap": latest start .. earliest end
    start = max(raw[b][0][0] - clap[b] for b in bands)
    start = max(start, ANALYSIS_DELAY_S)
    end = min(raw[b][0][-1] - clap[b] for b in bands)
    grid = np.arange(start, end, DT)

    out = {"t": grid, "speed": {}, "speed_raw": {}, "clap_local": clap,
           "completeness": {}, "raw": raw}
    for b in bands:
        t, acc, gyro = raw[b]
        rel = t - clap[b]                                   # band clock -> clap clock
        in_win = (rel >= start) & (rel < end)
        out["completeness"][b] = float(in_win.sum() / len(grid))   # rows kept / rows expected
        # fill the few missing rows by straight-line interpolation onto the shared grid
        g = np.column_stack([np.interp(grid, rel, gyro[:, i]) for i in range(3)])
        mag = np.linalg.norm(g, axis=1)                     # orientation-independent speed
        idle = np.median(mag[: int(0.5 * FS)])              # first 0.5 s = band at rest
        mag = np.clip(mag - idle, 0, None)                  # remove the noise floor offset
        out["speed_raw"][b] = mag
        out["speed"][b] = lowpass(mag)
    return out
