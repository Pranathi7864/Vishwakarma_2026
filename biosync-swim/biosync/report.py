"""
Step 3 of the pipeline: turn the scores into figures, an HTML report and a JSON file.
The three figures are the project's "screenshots" (docs/screenshot_*.png).
"""
import base64
import html
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from . import compare as C

NICE = {"L_wrist": "Left wrist", "R_wrist": "Right wrist",
        "L_ankle": "Left ankle", "R_ankle": "Right ankle"}
COL = {"L_wrist": "#1f77b4", "R_wrist": "#ff7f0e", "L_ankle": "#2ca02c", "R_ankle": "#9467bd"}
DISCLAIMER = ("Prototype software evaluated on SIMULATED data only. Not a medical device and "
              "not clinically validated. Flags are prompts for physiotherapist review.")


# ------------------------------------------------------------------ figure 1
def fig_sync(aligned, bands, path):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), constrained_layout=True)
    for b in bands:
        t, acc, _ = aligned["raw"][b]
        dev = np.abs(np.linalg.norm(acc, axis=1) - np.median(np.linalg.norm(acc, axis=1)))
        ax[0].plot(t, dev, color=COL[b], lw=1.4, label=NICE[b])
        ax[0].axvline(aligned["clap_local"][b], color=COL[b], ls=":", lw=1)
        tc = aligned["clap_local"][b]
        ax[1].plot(t - tc, dev, color=COL[b], lw=1.4, label=NICE[b])
    ax[0].set(xlim=(0, 3.2), title="BEFORE: each band's own clock - the clap appears at 4 different times",
              xlabel="time since that band was switched on (s)", ylabel="|acc| change (g)")
    ax[1].set(xlim=(-1.0, 1.2), title="AFTER: shifted by the detected clap - all four bands share t = 0",
              xlabel="time since clap (s)", ylabel="|acc| change (g)")
    ax[1].axvline(0, color="k", lw=1)
    ax[0].legend(ncol=2, loc="upper left")
    for a in ax:
        a.grid(alpha=.25)
    fig.suptitle("BioSync Swim - Step 1: four-band synchronisation by jump-clap", fontweight="bold")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_filter_dtw(aligned, cycles, windows, model, path, band="L_wrist", rep=6):
    fig = plt.figure(figsize=(12, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    # (a) raw vs filtered
    a = fig.add_subplot(gs[0, :])
    t = aligned["t"]
    sel = (t > t[0] + 1) & (t < t[0] + 12)
    a.plot(t[sel], aligned["speed_raw"][band][sel], color="#bbbbbb", lw=1, label="raw gyro magnitude")
    a.plot(t[sel], aligned["speed"][band][sel], color=COL[band], lw=2, label="Butterworth low-pass (6 Hz)")
    a.set(title=f"{NICE[band]}: sensor noise removed without shifting the movement in time",
          xlabel="time since clap (s)", ylabel="angular speed (deg/s)")
    a.legend(loc="upper right")
    a.grid(alpha=.25)
    # (b) template vs reps
    tmpl = model[band]["template"]
    bx = fig.add_subplot(gs[1, 0])
    normal = windows[band][rep - 1]
    dev = windows[band][rep]
    bx.plot(np.arange(len(tmpl)) / 50, tmpl, "k", lw=2.5, label="approved template")
    bx.plot(np.arange(len(normal)) / 50, normal, color="#2ca02c", lw=1.8, label=f"rep {rep} (normal)")
    bx.plot(np.arange(len(dev)) / 50, dev, color="#d62728", lw=1.8, label=f"rep {rep + 1} (deviating)")
    bx.set(title="One repetition vs template", xlabel="time within repetition (s)",
           ylabel="angular speed (deg/s)")
    bx.legend()
    bx.grid(alpha=.25)
    # (c) DTW cost matrix + path
    cx = fig.add_subplot(gs[1, 1])
    D = C.dtw_matrix(dev, tmpl)
    Dp = np.where(np.isinf(D[1:, 1:]), np.nan, D[1:, 1:])
    cx.imshow(Dp.T, origin="lower", aspect="auto", cmap="viridis")
    p = np.array(C.dtw_path(D))
    cx.plot(p[:, 0], p[:, 1], color="w", lw=2)
    cx.set(title="DTW: cheapest alignment path (white)", xlabel=f"rep {rep + 1} (samples)",
           ylabel="template (samples)")
    d_n = C.normalised_distance(dev, tmpl)
    d_o = C.normalised_distance(normal, tmpl)
    fig.suptitle(f"BioSync Swim - Step 2: filtering and DTW comparison   "
                 f"(distance: normal rep {d_o:.3f}, deviating rep {d_n:.3f}, "
                 f"threshold {model[band]['threshold']:.3f})", fontweight="bold")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_dashboard(aligned, bands, cycles, scores, summary, path):
    n = len(cycles)
    fig = plt.figure(figsize=(13, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], height_ratios=[1, 1.15])
    # heatmap
    h = fig.add_subplot(gs[0, 0])
    sev = np.array([[scores[b][i]["severity"] for i in range(n)] for b in bands])
    cmap = LinearSegmentedColormap.from_list("g2r", ["#2e9d57", "#f2e86d", "#d62728"])
    h.imshow(np.clip(sev, 0, 3), cmap=cmap, aspect="auto", vmin=0, vmax=2)
    for r, b in enumerate(bands):
        for c in range(n):
            flag = scores[b][c]["flag"]
            h.text(c, r, f"{sev[r, c]:.1f}", ha="center", va="center", fontsize=9,
                   fontweight="bold" if flag else "normal", color="w" if flag else "k")
            if flag:
                h.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False, ec="k", lw=2.5))
    h.set_xticks(range(n), [str(i + 1) for i in range(n)])
    h.set_yticks(range(len(bands)), [NICE[b] for b in bands])
    h.set(xlabel="repetition", title="Deviation severity (1.0 = flag threshold; boxed = FLAGGED)")
    # timeline
    tl = fig.add_subplot(gs[1, 0])
    t = aligned["t"]
    for k, b in enumerate(bands):
        s = aligned["speed"][b]
        tl.plot(t, s / s.max() * 0.85 + (len(bands) - 1 - k), color=COL[b], lw=1.2)
        for i, (s0, e0) in enumerate(cycles):
            if scores[b][i]["flag"]:
                tl.add_patch(Rectangle((t[s0], len(bands) - 1 - k), t[e0] - t[s0], 0.95,
                                       color="#d62728", alpha=.22, lw=0))
    tl.set_yticks([len(bands) - 1 - k + .4 for k in range(len(bands))], [NICE[b] for b in bands])
    tl.set(xlim=(t[cycles[0][0]] - .3, t[cycles[-1][1]] + .3), xlabel="time since clap (s)",
           title="Synchronised session timeline (red = flagged repetition)")
    # summary text
    s_ax = fig.add_subplot(gs[:, 1])
    s_ax.axis("off")
    lines = ["SESSION SUMMARY", "",
             f"Repetitions found:   {summary['n_reps']}",
             f"Flagged (rep, band): {summary['n_flagged']}", ""]
    lines += ["Flagged items:"] + [f"  - Rep {r}: {NICE[b]}" for r, b in summary["flagged"]] + [""]
    lines += ["Data completeness:"] + [f"  {NICE[b]:<12} {v * 100:5.1f} %"
                                        for b, v in summary["completeness"].items()] + [""]
    selfcheck = summary.get("selfcheck", "n/a").replace(", ", ",\n  ")
    lines += [f"Analysis time: {summary['analysis_seconds']:.1f} s", "",
              "Simulation self-check:", f"  {selfcheck}", "",
              "SIMULATED DATA - not a", "medical device."]
    s_ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10.2)
    fig.suptitle("BioSync Swim - Step 3: physiotherapist dashboard", fontweight="bold")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ HTML / JSON
def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def write_report(out_dir, bands, scores, summary, image_paths):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "session_report.json"), "w") as f:
        json.dump({"summary": summary, "scores": scores, "disclaimer": DISCLAIMER}, f, indent=1)
    rows = "".join(
        f"<tr><td>{r}</td><td>{html.escape(NICE[b])}</td>"
        f"<td>{scores[b][r - 1]['severity']:.2f}</td>"
        f"<td>Movement differs from approved template - review with physiotherapist.</td></tr>"
        for r, b in summary["flagged"]) or "<tr><td colspan=4>No deviations flagged.</td></tr>"
    comp = "".join(f"<li>{NICE[b]}: {v * 100:.1f}%</li>" for b, v in summary["completeness"].items())
    imgs = "".join(f'<img style="max-width:100%;margin:12px 0" src="data:image/png;base64,{_b64(p)}">'
                   for p in image_paths)
    page = f"""<!doctype html><meta charset="utf-8"><title>BioSync Swim session report</title>
<body style="font-family:system-ui;max-width:980px;margin:24px auto;padding:0 12px">
<h1>BioSync Swim - session report</h1>
<p style="background:#fff3cd;padding:8px;border-radius:6px">{html.escape(DISCLAIMER)}</p>
<p>Repetitions analysed: <b>{summary['n_reps']}</b> &middot; Flagged: <b>{summary['n_flagged']}</b>
&middot; Report generated in <b>{summary['report_seconds']:.1f} s</b></p>
<h2>Flagged repetitions</h2>
<table border="1" cellpadding="6" style="border-collapse:collapse">
<tr><th>Rep</th><th>Limb</th><th>Severity (x threshold)</th><th>Note</th></tr>{rows}</table>
<h2>Data completeness</h2><ul>{comp}</ul>
<p>Simulation self-check: {html.escape(str(summary.get('selfcheck', 'n/a')))}</p>
<h2>Figures</h2>{imgs}</body>"""
    with open(os.path.join(out_dir, "session_report.html"), "w") as f:
        f.write(page)
