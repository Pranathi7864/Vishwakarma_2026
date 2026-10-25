"""
Step 2 of the pipeline: cut the session into repetitions, compare every repetition
of every band with that band's reference template using Dynamic Time Warping (DTW),
and flag repetitions that are further from the template than normal variation allows.

Everything here is plain Python/numpy on purpose, so each line can be read and checked.
"""
import numpy as np
from scipy.signal import find_peaks

TEMPLATE_LEN = 100            # template samples (50 Hz after decimation = 2 s cycle)
DECIMATE = 2                  # 100 Hz -> 50 Hz (signal is already low-passed at 6 Hz)
K_SIGMA = 4.0                 # flag if distance > mean + K_SIGMA * std of normal reps
STD_FLOOR = 0.01              # never let the threshold collapse to ~0


# ----------------------------------------------------------------------------- DTW
def dtw_matrix(a, b, window_frac=0.3):
    """Cumulative-cost matrix D. D[i,j] = cheapest total cost of lining up a[:i] with b[:j].

    Each step may move diagonally (both advance), or advance only one series
    (one series "waits" = it is stretched or squeezed in time). The window keeps
    the path near the diagonal so wildly different alignments are not allowed."""
    a, b = list(map(float, a)), list(map(float, b))
    n, m = len(a), len(b)
    r = max(int(window_frac * max(n, m)), abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        centre = i * m / n
        for j in range(max(1, int(centre - r)), min(m, int(centre + r)) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D


def dtw_distance(a, b, window_frac=0.3):
    """Average cost per step along the best alignment (length-independent)."""
    D = dtw_matrix(a, b, window_frac)
    return D[len(a), len(b)] / (len(a) + len(b))


def dtw_path(D):
    """Walk back from the end of D to the start to get the matched index pairs."""
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        steps = [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)]
        _, i, j = min(steps)
        path.append((i - 1, j - 1))
    return path[::-1]


# ------------------------------------------------------------------ segmentation
def segment_cycles(ref_speed, fs=100):
    """Repetition boundaries (sample indices) from the reference band's speed signal.

    A repetition starts and ends where the limb is momentarily still, i.e. at the
    valleys of the speed curve. Active region = where speed exceeds 8 % of its 95th
    percentile."""
    p95 = np.percentile(ref_speed, 95)
    active = np.flatnonzero(ref_speed > 0.08 * p95)
    start, end = active[0], active[-1]
    valleys, _ = find_peaks(-ref_speed[start:end], distance=int(0.8 * fs),
                            prominence=0.25 * p95)
    bounds = [start] + (valleys + start).tolist() + [end]
    return list(zip(bounds[:-1], bounds[1:]))


def cut_windows(speed, cycles):
    """Cut one band's speed signal at the (shared) repetition boundaries, decimated."""
    return [speed[s:e][::DECIMATE] for s, e in cycles]


# ------------------------------------------------------------- template / threshold
def build_template(windows, length=TEMPLATE_LEN):
    """Average of repetitions after stretching each to the same length."""
    x_new = np.linspace(0, 1, length)
    stretched = [np.interp(x_new, np.linspace(0, 1, len(w)), w) for w in windows]
    return np.mean(stretched, axis=0)


def normalised_distance(window, template):
    """DTW distance divided by the template's peak -> 0.05 means ~5 % of peak speed."""
    return dtw_distance(window, template) / template.max()


def calibrate(cal_windows):
    """Learn template + threshold per band from physio-approved repetitions.

    The distance of each calibration rep is measured against a template built from
    the OTHER reps (leave-one-out), otherwise a rep would be compared with a template
    that already contains itself and the threshold would be too tight."""
    model = {}
    for band, wins in cal_windows.items():
        loo = []
        for i, w in enumerate(wins):
            template_wo = build_template(wins[:i] + wins[i + 1:])
            loo.append(normalised_distance(w, template_wo))
        loo = np.array(loo)
        model[band] = {"template": build_template(wins),
                       "mean": float(loo.mean()), "std": float(loo.std()),
                       "threshold": float(loo.mean() + K_SIGMA * max(loo.std(), STD_FLOOR)),
                       "cal_distances": loo.tolist()}
    return model


def score_session(session_windows, model):
    """distance, severity (= distance / threshold) and flag for every band and rep."""
    result = {}
    for band, wins in session_windows.items():
        rows = []
        for w in wins:
            d = normalised_distance(w, model[band]["template"])
            rows.append({"distance": float(d),
                         "severity": float(d / model[band]["threshold"]),
                         "flag": bool(d > model[band]["threshold"])})
        result[band] = rows
    return result
