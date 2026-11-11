"""Visit-to-visit consistency for a multi-round recording (default: session_0-9_glenda_2.csv).

Each contiguous labelled block is one "visit". Evaluation is leave-one-visit-out:
train on every other visit, test on the held-out one (±TRIM s around label switches dropped).

Usage:  python3 visit_consistency.py [csv name in ../]
Outputs: <stem>_consistency.png, <stem>_timeline.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
FNAME = sys.argv[1] if len(sys.argv) > 1 else "session_0-9_glenda_2.csv"
STEM = Path(FNAME).stem
TRIM = 2.0
MIN_PKTS = 20  # visits with fewer steady packets are skipped

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------------- load ----------------
df = pd.read_csv(HERE.parent / FNAME)
raw = np.array([json.loads(s) for s in df.data], dtype=float)
amp = np.abs(raw[:, 1::2] + 1j * raw[:, 0::2])
amp = amp[:, amp.mean(0) > 0.5]
X = amp / np.linalg.norm(amp, axis=1, keepdims=True)
lab = df.label.astype(str).values
t = (df.local_timestamp.values - df.local_timestamp.values[0]) / 1e6
visit = np.cumsum(np.r_[0, lab[1:] != lab[:-1]])
sw = np.where(lab[1:] != lab[:-1])[0] + 1
steady = np.min(np.abs(t[:, None] - t[sw][None]), axis=1) > TRIM
use = steady & (lab != "none")

vis = [v for v in np.unique(visit) if (use & (visit == v)).sum() >= MIN_PKTS]
vpos = {v: lab[visit == v][0] for v in vis}
vt = {v: t[visit == v][0] for v in vis}
C = {v: X[use & (visit == v)].mean(0) for v in vis}
empty = X[lab == "none"].mean(0)

# ---------------- leave-one-visit-out ----------------
acc = {}
for v in vis:
    te, tr = use & (visit == v), use & (visit != v)
    p = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(X[tr], lab[tr]).predict(X[te])
    acc[v] = (p == lab[te]).mean()
n_te = {v: (use & (visit == v)).sum() for v in vis}
overall = sum(acc[v] * n_te[v] for v in vis) / sum(n_te.values())
print(f"leave-one-visit-out accuracy: {overall:.3f} (chance {1 / len(set(vpos.values())):.3f})")

positions = sorted(set(vpos.values()), key=int)
print("per position:", {P: round(np.mean([acc[v] for v in vis if vpos[v] == P]), 2) for P in positions})

# ---------------- figure: consistency ----------------
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1], hspace=0.45, wspace=0.28)
TIME_CMAP = LinearSegmentedColormap.from_list("time", ["#86b6ef", "#2a78d6", "#104281", "#0d366b"])
tnorm = plt.Normalize(0, t[-1])


def jitter(v):
    # spread repeated visits of a position horizontally, earliest on the left
    same = sorted([u for u in vis if vpos[u] == vpos[v]], key=lambda u: vt[u])
    k = same.index(v)
    return positions.index(vpos[v]) + (k - (len(same) - 1) / 2) * 0.22


def dots(ax, yfun):
    sc = ax.scatter([jitter(v) for v in vis], [yfun(v) for v in vis], s=70, c=[vt[v] for v in vis],
                    cmap=TIME_CMAP, norm=tnorm, edgecolor=SURFACE, linewidth=2, zorder=3)
    return sc


# (a) distance from empty room
ax = fig.add_subplot(gs[0, 0])
sc = dots(ax, lambda v: np.linalg.norm(C[v] - empty))
ax.set_xticks(range(len(positions)), positions)
ax.set_xlabel("Position")
ax.set_ylabel("Distance from empty-room CSI")
ax.set_ylim(0, None)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_title("(a) How much each visit changes the CSI vs an empty room", loc="left")
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Visit start time (s)", color=INK2)
cb.outline.set_visible(False)

# (b) leave-one-visit-out accuracy per visit
ax = fig.add_subplot(gs[1, 0])
sc = dots(ax, lambda v: acc[v])
chance = 1 / len(positions)
ax.axhline(chance, ls="--", lw=1.2, color=INK2, zorder=2)
ax.text(len(positions) - 0.5, chance + 0.02, f"chance {chance:.0%}", ha="right", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(positions)), positions)
ax.set_ylim(-0.03, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_xlabel("Position")
ax.set_ylabel("Accuracy on held-out visit")
ax.set_title(f"(b) Train on all other visits, test on this one — overall {overall:.0%}", loc="left")
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Visit start time (s)", color=INK2)
cb.outline.set_visible(False)

# (c) visit x visit distance, sorted by position then time
ax = fig.add_subplot(gs[:, 1])
order = sorted(vis, key=lambda v: (int(vpos[v]), vt[v]))
Dm = np.array([[np.linalg.norm(C[a] - C[b]) for b in order] for a in order])
im = ax.imshow(Dm, cmap=SEQ.reversed(), vmin=0, vmax=np.percentile(Dm, 95))
ticks = [f"pos {vpos[v]} @{vt[v]:.0f}s" for v in order]
ax.set_xticks(range(len(order)), ticks, rotation=90, fontsize=8)
ax.set_yticks(range(len(order)), ticks, fontsize=8)
bounds = [i for i in range(1, len(order)) if vpos[order[i]] != vpos[order[i - 1]]]
for b in bounds:
    ax.axhline(b - 0.5, color=SURFACE, lw=2)
    ax.axvline(b - 0.5, color=SURFACE, lw=2)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("(c) CSI distance between visits\n(dark = similar; consistent positions form dark squares on the diagonal)",
             loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Distance between visit-average CSI", color=INK2)
cb.outline.set_visible(False)
fig.suptitle(f"{STEM}: is the same position consistent across visits?", x=0.07, y=0.97, ha="left",
             fontsize=13, fontweight="bold")
fig.savefig(HERE / f"{STEM}_consistency.png", dpi=150, bbox_inches="tight")

# ---------------- figure: timeline ----------------
fig, ax = plt.subplots(figsize=(16, 4.2))
grid = np.arange(0, t[-1], 0.1)
bins = np.searchsorted(grid, t, side="right") - 1
G = np.full((len(grid), X.shape[1]), np.nan)
for b_ in np.unique(bins):
    G[b_] = X[bins == b_].mean(0)
G = pd.DataFrame(G).ffill().rolling(5, center=True, min_periods=1).mean().values
ax.imshow((G / np.nanmean(G, 0)).T, aspect="auto", cmap=SEQ, vmin=0.4, vmax=1.6, origin="lower",
          extent=(0, grid[-1] + 0.1, 0, X.shape[1]), interpolation="nearest")
edges = np.r_[0, sw, len(lab)]
for e in sw:
    ax.axvspan(t[e] - TRIM, t[e] + TRIM, color="#eb6834", alpha=0.25, lw=0)
    ax.axvline(t[e], color=INK, lw=1)
for e0, e1 in zip(edges[:-1], edges[1:]):
    ax.text((t[e0] + t[e1 - 1]) / 2, X.shape[1] + 1, lab[e0], ha="center", va="bottom",
            fontsize=9, color=INK, clip_on=False)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Subcarrier")
ax.set_title(f"{STEM}: CSI amplitude vs time (top labels = position; shaded = ±{TRIM:g} s dropped)",
             loc="left", pad=16)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.savefig(HERE / f"{STEM}_timeline.png", dpi=150, bbox_inches="tight")
print("saved", f"{STEM}_consistency.png", f"{STEM}_timeline.png")
