"""CSI position-classification analysis + plots.

Model: RandomForestClassifier (300 trees) on per-packet CSI amplitude
(valid subcarriers, each packet L2-normalised to remove gain/AGC changes).
Packets within TRIM seconds of any label switch are dropped (walking between spots).

Usage:  python3 plot_localization.py   (run from this folder)
Outputs: accuracy.png, cross_session.png, confusion.png, csi_timeline.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
TRIM = 2.0  # seconds dropped either side of each label switch
SESSIONS = {  # name -> file, in recording order
    "Glenda": "session_0-9_glenda.csv",
    "Siqi 1": "session_0-9_siqi.csv",
    "Siqi 2": "session_0-9_siqi_2.csv",
}
COMMON = [str(i) for i in range(1, 10)]  # positions present in every session (Glenda has no 0)

# palette (light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TRIM_SHADE = "#eb6834"
SEQ = LinearSegmentedColormap.from_list(
    "seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def rf():
    return RandomForestClassifier(300, n_jobs=-1, random_state=0)


def block_frac(lab, mask):
    """Position (0..1) of each masked packet within its label's block."""
    fr = np.zeros(len(lab))
    for L in np.unique(lab):
        ii = np.where((lab == L) & mask)[0]
        fr[ii] = np.arange(len(ii)) / max(len(ii), 1)
    return fr


def load(fname):
    df = pd.read_csv(DATA / fname)
    raw = np.array([json.loads(s) for s in df.data], dtype=float)
    amp = np.abs(raw[:, 1::2] + 1j * raw[:, 0::2])  # ESP32 order: [imag, real]
    lab = df.label.astype(str).values
    t = (df.local_timestamp.values - df.local_timestamp.values[0]) / 1e6
    sw = np.where(lab[1:] != lab[:-1])[0] + 1
    steady = np.min(np.abs(t[:, None] - t[sw][None]), axis=1) > TRIM
    pos = lab != "none"
    return dict(amp=amp, lab=lab, t=t, switches=sw, steady=steady, pos=pos,
                frac=block_frac(lab, np.ones(len(lab), bool)), frac_steady=block_frac(lab, steady),
                positions=[l for l in pd.unique(lab) if l != "none"])


S = {k: load(v) for k, v in SESSIONS.items()}
names = list(S)
valid = np.logical_and.reduce([s["amp"].mean(0) > 0.5 for s in S.values()])
for s in S.values():
    a = s["amp"][:, valid]
    s["X"] = a / np.linalg.norm(a, axis=1, keepdims=True)

# ---------------- within-session evaluations ----------------
res = {}
for n, s in S.items():
    X, y, pos = s["X"], s["lab"], s["pos"]
    Xa, Xb, ya, yb = train_test_split(X[pos], y[pos], test_size=0.3, stratify=y[pos], random_state=0)
    rand = accuracy_score(yb, rf().fit(Xa, ya).predict(Xb))
    tr, te = pos & (s["frac"] < 0.65), pos & (s["frac"] >= 0.7)
    blocked = accuracy_score(y[te], rf().fit(X[tr], y[tr]).predict(X[te]))
    st = pos & s["steady"]
    tr, te = st & (s["frac_steady"] < 0.65), st & (s["frac_steady"] >= 0.7)
    pred = rf().fit(X[tr], y[tr]).predict(X[te])
    res[n] = dict(random=rand, blocked=blocked, trimmed=accuracy_score(y[te], pred),
                  cm=confusion_matrix(y[te], pred, labels=s["positions"]),
                  chance=1 / len(s["positions"]))

# ---------------- cross-session (train on all steady packets of A, test on B), positions 1-9 ----------------
cross = np.full((len(names), len(names)), np.nan)
cross_cm = {}
for i, a in enumerate(names):
    A = S[a]
    ma = np.isin(A["lab"], COMMON) & A["steady"]
    model = rf().fit(A["X"][ma], A["lab"][ma])
    for j, b in enumerate(names):
        B = S[b]
        if i == j:  # diagonal: within-session time-blocked on the same 9 positions
            m = np.isin(B["lab"], COMMON) & B["steady"]
            tr, te = m & (B["frac_steady"] < 0.65), m & (B["frac_steady"] >= 0.7)
            cross[i, j] = accuracy_score(B["lab"][te], rf().fit(B["X"][tr], B["lab"][tr]).predict(B["X"][te]))
            continue
        mb = np.isin(B["lab"], COMMON) & B["steady"]
        p = model.predict(B["X"][mb])
        cross[i, j] = accuracy_score(B["lab"][mb], p)
        cross_cm[(a, b)] = confusion_matrix(B["lab"][mb], p, labels=COMMON)

empty = {n: S[n]["X"][S[n]["lab"] == "none"].mean(0) for n in names}
for n, r in res.items():
    print(f"{n:7s} random={r['random']:.3f} blocked={r['blocked']:.3f} trimmed(±{TRIM:g}s)={r['trimmed']:.3f}")
print("cross-session (rows=train, cols=test, positions 1-9; diagonal=within):")
print(pd.DataFrame(cross, index=names, columns=names).round(3))
print("empty-room centroid distance:")
print(pd.DataFrame([[np.linalg.norm(empty[a] - empty[b]) for b in names] for a in names],
                   index=names, columns=names).round(3))


def pct_axis(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)


# ---------------- 1. within-session accuracy ----------------
fig, ax = plt.subplots(figsize=(9, 4.8))
kinds = [("random", "Random 70/30 split (leaky — neighbouring packets in train & test)"),
         ("blocked", "Time-blocked split (train first 65%, test last 30% of each position)"),
         ("trimmed", f"Time-blocked, ±{TRIM:g} s around label switches dropped")]
w = 0.25
for k, (key, lbl) in enumerate(kinds):
    xs = np.arange(len(names)) + (k - 1) * w
    vals = [res[n][key] for n in names]
    ax.bar(xs, vals, w, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=lbl, zorder=2)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK)
for j, n in enumerate(names):
    c = res[n]["chance"]
    ax.plot([j - 1.7 * w, j + 1.7 * w], [c, c], ls="--", lw=1.2, color=INK2, zorder=3,
            label="Chance (1 / #positions)" if j == 0 else None)
ax.set_xticks(range(len(names)), [f"{n}\n({len(S[n]['positions'])} positions)" for n in names])
ax.set_ylim(0, 1.0)
pct_axis(ax)
ax.set_ylabel("Accuracy")
ax.set_title("Within-session position accuracy (Random Forest, per-packet CSI amplitude)", loc="left")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "accuracy.png", dpi=160)

# ---------------- 2. cross-session matrix ----------------
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.imshow(cross, cmap=SEQ, vmin=0, vmax=1)
for i in range(len(names)):
    for j in range(len(names)):
        v = cross[i, j]
        ax.text(j, i, f"{v:.0%}" + ("\nwithin" if i == j else ""), ha="center", va="center",
                fontsize=10, color="white" if v > 0.55 else INK)
ax.set_xticks(range(len(names)), names)
ax.set_yticks(range(len(names)), names)
ax.set_xlabel("Tested on")
ax.set_ylabel("Trained on")
ax.set_title(f"Train on one session, test on another\n(positions 1–9, chance {1/len(COMMON):.0%}; "
             f"±{TRIM:g} s transitions dropped)", loc="left")
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "cross_session.png", dpi=160)

# ---------------- 3. confusion matrices ----------------
panels = [(n, res[n]["cm"], S[n]["positions"], f"{n}: within-session (time-blocked)\naccuracy {res[n]['trimmed']:.0%}")
          for n in names]
a, b = "Siqi 1", "Siqi 2"
panels.append((None, cross_cm[(a, b)], COMMON,
               f"Trained on {a} → tested on {b}\naccuracy {cross[names.index(a), names.index(b)]:.0%}"))
fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
for ax, (n, cm, labs, title) in zip(axes.flat, panels):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cmn[i, j] >= 0.05:
                ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labs)), labs)
    ax.set_yticks(range(len(labs)), labs)
    ax.set_xlabel("Predicted position")
    ax.set_ylabel("True position" + (" (in recording order)" if n else ""))
    ax.set_title(title, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.text(0.01, 0.01, f"Cells: % of each true position's test packets (row-normalised). "
         f"Packets within ±{TRIM:g} s of a label switch excluded.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(HERE / "confusion.png", dpi=160)

# ---------------- 4. CSI amplitude over time ----------------
fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.6 * len(names)), gridspec_kw=dict(hspace=0.5))
for ax, n in zip(axes, names):
    s = S[n]
    X, t, lab = s["X"], s["t"], s["lab"]
    # packet rate is not constant, so bin onto a uniform 0.1 s time grid
    grid = np.arange(0, t[-1], 0.1)
    bins = np.searchsorted(grid, t, side="right") - 1
    G = np.full((len(grid), X.shape[1]), np.nan)
    for b_ in np.unique(bins):
        G[b_] = X[bins == b_].mean(0)
    G = pd.DataFrame(G).ffill().rolling(5, center=True, min_periods=1).mean().values
    rel = G / np.nanmean(G, 0)
    ax.imshow(rel.T, aspect="auto", cmap=SEQ, vmin=0.4, vmax=1.6, origin="lower",
              extent=(grid[0], grid[-1] + 0.1, 0, X.shape[1]), interpolation="nearest")
    edges = np.r_[0, s["switches"], len(lab)]
    for e in s["switches"]:
        ax.axvspan(t[e] - TRIM, t[e] + TRIM, color=TRIM_SHADE, alpha=0.25, lw=0)
        ax.axvline(t[e], color=INK, lw=1)
    for e0, e1 in zip(edges[:-1], edges[1:]):
        ax.text((t[e0] + t[e1 - 1]) / 2, X.shape[1] + 1, lab[e0],
                ha="center", va="bottom", fontsize=9, color=INK, clip_on=False)
    ax.set_ylabel("Subcarrier")
    ax.set_title(f"{n}: CSI amplitude vs time (top labels = position; shaded = ±{TRIM:g} s dropped)",
                 loc="left", pad=16)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[-1].set_xlabel("Time (s)")
sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0.4, 1.6))
cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
cb.set_label("Amplitude / subcarrier mean", color=INK2)
cb.outline.set_visible(False)
fig.savefig(HERE / "csi_timeline.png", dpi=160, bbox_inches="tight")
print("saved to", HERE)
