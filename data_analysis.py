"""
Emergency Vehicle Priority System - Data Analysis

Outputs (all saved to thesis_analysis/):
  Figures 01-17  — core analysis: boxplots, CDFs, heatmap, scatter, stats
  Figure  18     — background traffic delay (ON vs OFF), requires background_metrics.csv
  Figure  19     — TL count vs travel time improvement scatter
  Figure  20     — preemption strategy comparison (requires multiple strategies)
  Figure  21     — TL hold duration sensitivity (requires multiple hold durations)
  Figure  22     — demand scenario comparison (requires multiple demand scenarios)
  report_01      — summary stats by (distance, mode)
  report_02      — summary stats by (try, distance, mode)
  report_03      — paired comparison per vehicle per distance
  report_04      — statistical tests (paired t-test + Cohen's d) per distance
  report_05      — preemption event summary by intersection
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
COL_ON  = "#2ecc71"   # Priority ON  — green
COL_OFF = "#e74c3c"   # Priority OFF — red
PALETTE = {"Priority ON": COL_ON, "Priority OFF": COL_OFF}
DIST_CMAP = plt.cm.viridis

OUTPUT_DIR = "thesis_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig_count = 0
def savefig(name, fig=None):
    global fig_count
    fig_count += 1
    path = f"{OUTPUT_DIR}/{fig_count:02d}_{name}.png"
    (fig or plt).savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("=" * 70)
print("EMERGENCY VEHICLE PRIORITY SYSTEM — DATA ANALYSIS")
print("=" * 70)

CSV_FILE      = "emergency_metrics_priority.csv"
PREEMPT_FILE  = "preemption_events.csv"
BG_FILE       = "background_metrics.csv"

if not Path(CSV_FILE).exists():
    raise FileNotFoundError(f"{CSV_FILE} not found. Run V22.CLAUDE.py first.")

df = pd.read_csv(CSV_FILE)
print(f"\nLoaded {len(df)} rows from {CSV_FILE}")
print(f"Columns: {list(df.columns)}")

# Load background metrics (written by V22.CLAUDE.py for every run)
bg_df = None
if Path(BG_FILE).exists():
    bg_df = pd.read_csv(BG_FILE)
    bg_df["priority_enabled"] = bg_df["priority_enabled"].apply(
        lambda x: str(x).strip().lower() == "true")
    print(f"Loaded {len(bg_df)} background vehicle records from {BG_FILE}")
else:
    print(f"  (background metrics file {BG_FILE} not found — Figures 18 will be skipped)")

# Normalise priority_enabled to actual bool (CSV stores it as "True"/"False" strings)
df["priority_enabled"] = df["priority_enabled"].apply(lambda x: str(x).strip().lower() == "true")
df["priority_label"]   = df["priority_enabled"].map({True: "Priority ON", False: "Priority OFF"})

# New columns present in V22.CLAUDE.py output (not in older CSVs)
HAS_DEMAND_SCENARIO     = "demand_scenario"     in df.columns
HAS_PREEMPTION_STRATEGY = "preemption_strategy" in df.columns
HAS_HOLD_DURATION       = "hold_duration"       in df.columns
HAS_TL_COUNT            = "tl_count"            in df.columns

if not HAS_DEMAND_SCENARIO:     df["demand_scenario"]     = "normal"
if not HAS_PREEMPTION_STRATEGY: df["preemption_strategy"] = "smart_phase"
if not HAS_HOLD_DURATION:       df["hold_duration"]       = 10
if not HAS_TL_COUNT:            df["tl_count"]            = np.nan

DEMAND_NAMES  = sorted(df["demand_scenario"].unique())
STRATEGIES    = sorted(df[df["priority_enabled"]]["preemption_strategy"].unique())
HOLD_VALS     = sorted(df[df["priority_enabled"]]["hold_duration"].unique())

# Rename legacy column
if "spillback_pct" in df.columns and "congestion_delta_pct" not in df.columns:
    df.rename(columns={"spillback_pct": "congestion_delta_pct"}, inplace=True)

# Detect CSV format:
#   New (V22.CLAUDE): OFF rows have priority_distance=0.0 (distance-independent baseline).
#   Old (V22):        OFF rows have real distance values — duplicate runs per distance.
HAS_DISTANCE_SWEEP = "priority_distance" in df.columns
if not HAS_DISTANCE_SWEEP:
    print("\nNote: no priority_distance column — assuming single-distance run.")
    df["priority_distance"] = 100.0

off_df = df[~df["priority_enabled"]].copy()
on_df  = df[df["priority_enabled"]].copy()

# Derived columns
df["avg_speed_ms"]  = df["distance"] / df["travel_time"].replace(0, np.nan)
df["stopped_pct"]   = df["stopped_time"] / df["travel_time"].replace(0, np.nan) * 100
off_df["avg_speed_ms"] = off_df["distance"] / off_df["travel_time"].replace(0, np.nan)
off_df["stopped_pct"]  = off_df["stopped_time"] / off_df["travel_time"].replace(0, np.nan) * 100
on_df["avg_speed_ms"]  = on_df["distance"] / on_df["travel_time"].replace(0, np.nan)
on_df["stopped_pct"]   = on_df["stopped_time"] / on_df["travel_time"].replace(0, np.nan) * 100

OLD_FORMAT = HAS_DISTANCE_SWEEP and (off_df["priority_distance"] != 0.0).any()
if OLD_FORMAT:
    print("\nNote: detected old-format CSV (OFF rows have distance values). "
          "Using OFF rows at each distance as the comparison baseline.")
    DISTANCES = sorted(on_df["priority_distance"].unique())
else:
    # New format: OFF rows are a single baseline (priority_distance=0.0)
    DISTANCES = sorted(on_df["priority_distance"].unique())
    print(f"\nNew-format CSV detected. Baseline (OFF) is distance-independent.")
    print(f"ON distances: {[int(d) for d in DISTANCES]} m")

TRIES    = sorted(df["try_number"].unique())
NUM_DIST = len(DISTANCES)
dist_colors = {d: DIST_CMAP(i / max(NUM_DIST - 1, 1)) for i, d in enumerate(DISTANCES)}


def get_off_for_dist(dist):
    """Return the OFF baseline rows for a given distance comparison."""
    if OLD_FORMAT:
        return off_df[off_df["priority_distance"] == dist]
    return off_df  # single baseline, same for all distances


def get_on_for_dist(dist):
    return on_df[on_df["priority_distance"] == dist]

# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nBasic stats:")
print(df[["travel_time", "distance", "stopped_time"]].describe().round(2))

def find_outliers(data, col):
    q1, q3 = data[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data[col] < lo) | (data[col] > hi)], lo, hi

print("\nOutlier counts (IQR method):")
for col in ["travel_time", "distance", "stopped_time"]:
    out, lo, hi = find_outliers(df, col)
    print(f"  {col:20s}: {len(out):3d} outliers  bounds=[{lo:.1f}, {hi:.1f}]")

# ---------------------------------------------------------------------------
# 3. SUMMARY STATS PER (DISTANCE, MODE)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY STATISTICS BY PRIORITY DISTANCE AND MODE")
print("=" * 70)

print("\nBaseline (priority OFF):")
print(off_df[["travel_time", "stopped_time", "congestion_delta_pct"]].describe().round(2))
print("\nPriority ON by distance:")
summary = (
    on_df.groupby("priority_distance")[["travel_time", "stopped_time", "congestion_delta_pct"]]
    .agg(["mean", "median", "std", "min", "max"])
    .round(2)
)
print(summary.to_string())

# ---------------------------------------------------------------------------
# 4. PAIRED STATISTICAL TESTS PER DISTANCE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PAIRED STATISTICAL TESTS (per distance threshold)")
print("=" * 70)

def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

stat_rows = []
for dist in DISTANCES:
    sub_on  = get_on_for_dist(dist)[["try_number", "id", "travel_time"]].rename(
        columns={"travel_time": "tt_on"})
    sub_off = get_off_for_dist(dist)[["try_number", "id", "travel_time"]].rename(
        columns={"travel_time": "tt_off"})
    paired = sub_on.merge(sub_off, on=["try_number", "id"]).dropna()
    if len(paired) < 2:
        continue
    a_on, a_off = paired["tt_on"].values, paired["tt_off"].values
    t_tt, p_tt   = stats.ttest_rel(a_on, a_off)
    _, p_mw      = stats.wilcoxon(a_on, a_off)
    d            = cohens_d(a_on, a_off)
    imp          = (a_off.mean() - a_on.mean()) / a_off.mean() * 100
    sig          = "YES *" if p_tt < 0.05 else "no"

    print(f"\n  Distance {dist}m  (n={len(paired)} pairs)")
    print(f"    Mean ON={a_on.mean():.1f}s  OFF={a_off.mean():.1f}s  "
          f"Improvement={imp:+.1f}%")
    print(f"    Paired t-test: t={t_tt:.3f}  p={p_tt:.4f}  sig={sig}")
    print(f"    Wilcoxon:      p={p_mw:.4f}")
    print(f"    Cohen's d:     {d:.3f}  ({'large' if abs(d)>0.8 else 'medium' if abs(d)>0.5 else 'small'})")

    stat_rows.append({
        "priority_distance_m": dist,
        "n_pairs": len(paired),
        "mean_on_s": round(a_on.mean(), 2),
        "mean_off_s": round(a_off.mean(), 2),
        "improvement_pct": round(imp, 2),
        "paired_t": round(t_tt, 4),
        "p_ttest": round(p_tt, 6),
        "p_wilcoxon": round(p_mw, 6),
        "cohens_d": round(d, 4),
        "significant": sig,
    })

stats_df = pd.DataFrame(stat_rows)

# ---------------------------------------------------------------------------
# 5. FIGURES
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

METRICS_INFO = [
    ("travel_time",        "Travel Time (s)"),
    ("stopped_time",       "Stopped Time (s)"),
    ("congestion_delta_pct", "Congestion Delta (%)"),
]

# ---- Figure 1: Box plots — travel_time by distance and mode ---------------
fig, axes = plt.subplots(1, NUM_DIST, figsize=(4 * NUM_DIST, 5), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Travel Time Distribution: Priority ON vs OFF by Distance Threshold",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES):
    groups = [get_on_for_dist(dist)["travel_time"].values,
              get_off_for_dist(dist)["travel_time"].values]
    bp = ax.boxplot(groups, labels=["ON", "OFF"], patch_artist=True, widths=0.5)
    for patch, col in zip(bp["boxes"], [COL_ON, COL_OFF]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for element in ["medians"]:
        for line in bp[element]: line.set_color("black"); line.set_linewidth(2)
    means = [g.mean() for g in groups]
    ax.plot([1, 2], means, "D", color="navy", markersize=7, zorder=5, label="Mean")
    ax.set_title(f"{int(dist)} m threshold", fontweight="bold")
    ax.set_xlabel("Priority Mode")
    if ax == axes[0]: ax.set_ylabel("Travel Time (s)", fontweight="bold")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("travel_time_boxplot_by_distance")

# ---- Figure 2: Box plots — stopped_time by distance and mode --------------
fig, axes = plt.subplots(1, NUM_DIST, figsize=(4 * NUM_DIST, 5), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Stopped Time Distribution: Priority ON vs OFF by Distance Threshold",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES):
    groups = [get_on_for_dist(dist)["stopped_time"].values,
              get_off_for_dist(dist)["stopped_time"].values]
    bp = ax.boxplot(groups, labels=["ON", "OFF"], patch_artist=True, widths=0.5)
    for patch, col in zip(bp["boxes"], [COL_ON, COL_OFF]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for element in ["medians"]:
        for line in bp[element]: line.set_color("black"); line.set_linewidth(2)
    means = [g.mean() for g in groups]
    ax.plot([1, 2], means, "D", color="navy", markersize=7, zorder=5, label="Mean")
    ax.set_title(f"{int(dist)} m threshold", fontweight="bold")
    ax.set_xlabel("Priority Mode")
    if ax == axes[0]: ax.set_ylabel("Stopped Time (s)", fontweight="bold")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("stopped_time_boxplot_by_distance")

# ---- Figure 3: Mean improvement curve vs distance threshold ---------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Effect of Priority Distance Threshold on Emergency Vehicle Performance",
             fontsize=13, fontweight="bold")

improvements = []
for dist in DISTANCES:
    on  = get_on_for_dist(dist)
    off = get_off_for_dist(dist)
    # Paired improvement per vehicle to get correct CI
    paired = on[["try_number", "id", "travel_time", "stopped_time"]].merge(
        off[["try_number", "id", "travel_time", "stopped_time"]],
        on=["try_number", "id"], suffixes=("_on", "_off")).dropna()
    paired["tt_imp_pct"] = (paired["travel_time_off"] - paired["travel_time_on"]) / paired["travel_time_off"] * 100
    paired["st_imp_pct"] = (paired["stopped_time_off"] - paired["stopped_time_on"]) / paired["stopped_time_off"].replace(0, np.nan) * 100
    n = len(paired)
    tt_imp  = paired["tt_imp_pct"].mean()
    st_imp  = paired["st_imp_pct"].mean()
    tt_ci   = 1.96 * paired["tt_imp_pct"].std() / np.sqrt(n) if n > 1 else 0
    st_ci   = 1.96 * paired["st_imp_pct"].std() / np.sqrt(n) if n > 1 else 0
    improvements.append((dist, tt_imp, st_imp, tt_ci, st_ci))

imp_df = pd.DataFrame(improvements, columns=["dist", "tt_imp", "st_imp", "tt_ci", "st_ci"])

ax = axes[0]
ax.plot(imp_df["dist"], imp_df["tt_imp"], "o-", color="#3498db", linewidth=2.5,
        markersize=9, markerfacecolor="white", markeredgewidth=2.5)
ax.fill_between(imp_df["dist"],
                imp_df["tt_imp"] - imp_df["tt_ci"],
                imp_df["tt_imp"] + imp_df["tt_ci"],
                alpha=0.2, color="#3498db", label="95% CI")
ax.axhline(0, color="grey", linestyle="--", linewidth=1)
ax.fill_between(imp_df["dist"], 0, imp_df["tt_imp"],
                where=imp_df["tt_imp"] > 0, alpha=0.12, color="#2ecc71")
ax.fill_between(imp_df["dist"], 0, imp_df["tt_imp"],
                where=imp_df["tt_imp"] <= 0, alpha=0.12, color="#e74c3c")
ax.set_xlabel("Priority Distance Threshold (m)", fontweight="bold")
ax.set_ylabel("Travel Time Reduction (%)", fontweight="bold")
ax.set_title("Travel Time Improvement vs Threshold")
ax.set_xticks(DISTANCES)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)

ax = axes[1]
ax.plot(imp_df["dist"], imp_df["st_imp"], "s-", color="#e67e22", linewidth=2.5,
        markersize=9, markerfacecolor="white", markeredgewidth=2.5)
ax.fill_between(imp_df["dist"],
                imp_df["st_imp"] - imp_df["st_ci"],
                imp_df["st_imp"] + imp_df["st_ci"],
                alpha=0.2, color="#e67e22", label="95% CI")
ax.axhline(0, color="grey", linestyle="--", linewidth=1)
ax.fill_between(imp_df["dist"], 0, imp_df["st_imp"],
                where=imp_df["st_imp"] > 0, alpha=0.12, color="#2ecc71")
ax.fill_between(imp_df["dist"], 0, imp_df["st_imp"],
                where=imp_df["st_imp"] <= 0, alpha=0.12, color="#e74c3c")
ax.set_xlabel("Priority Distance Threshold (m)", fontweight="bold")
ax.set_ylabel("Stopped Time Reduction (%)", fontweight="bold")
ax.set_title("Stopped Time Improvement vs Threshold")
ax.set_xticks(DISTANCES)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)

plt.tight_layout()
savefig("improvement_vs_distance_threshold")

# ---- Figure 4: Grouped bar chart — mean ± std for all groups --------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Mean Performance: Priority ON vs OFF for Each Distance Threshold",
             fontsize=13, fontweight="bold")

x    = np.arange(NUM_DIST)
w    = 0.35

for ax, (col, ylabel) in zip(axes, [("travel_time", "Mean Travel Time (s)"),
                                     ("stopped_time", "Mean Stopped Time (s)")]):
    means_on  = [get_on_for_dist(d)[col].mean() for d in DISTANCES]
    means_off = [get_off_for_dist(d)[col].mean() for d in DISTANCES]
    stds_on   = [get_on_for_dist(d)[col].std()  for d in DISTANCES]
    stds_off  = [get_off_for_dist(d)[col].std()  for d in DISTANCES]

    ax.bar(x - w/2, means_on,  w, yerr=stds_on,  label="Priority ON",
           color=COL_ON,  alpha=0.85, capsize=5, error_kw={"linewidth": 1.5})
    ax.bar(x + w/2, means_off, w, yerr=stds_off, label="Priority OFF",
           color=COL_OFF, alpha=0.85, capsize=5, error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)} m" for d in DISTANCES])
    ax.set_xlabel("Priority Distance Threshold", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
savefig("grouped_bar_mean_by_distance")

# ---- Figure 5: Violin plots — travel_time distribution full view ----------
fig, axes = plt.subplots(1, NUM_DIST, figsize=(4 * NUM_DIST, 5), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Travel Time Density: Priority ON vs OFF by Distance Threshold",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES):
    sub = pd.concat([get_on_for_dist(dist), get_off_for_dist(dist)]).copy()
    sns.violinplot(data=sub, x="priority_label", y="travel_time", ax=ax,
                   palette=PALETTE, order=["Priority ON", "Priority OFF"],
                   inner="box", linewidth=1.2, cut=0)
    ax.set_title(f"{int(dist)} m threshold", fontweight="bold")
    ax.set_xlabel("")
    if ax == axes[0]:
        ax.set_ylabel("Travel Time (s)", fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=10)
plt.tight_layout()
savefig("violin_travel_time_by_distance")

# ---- Figure 6: CDF — cumulative distribution of travel_time ---------------
fig, axes = plt.subplots(1, NUM_DIST, figsize=(4 * NUM_DIST, 5), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Cumulative Distribution of Travel Time by Distance Threshold",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES):
    for getter, label, color in [(get_on_for_dist, "Priority ON", COL_ON),
                                  (get_off_for_dist, "Priority OFF", COL_OFF)]:
        vals = np.sort(getter(dist)["travel_time"].values)
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, cdf, label=label, color=color, linewidth=2.2)
    ax.set_title(f"{int(dist)} m threshold", fontweight="bold")
    ax.set_xlabel("Travel Time (s)", fontweight="bold")
    if ax == axes[0]:
        ax.set_ylabel("Cumulative Probability", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("cdf_travel_time_by_distance")

# ---- Figure 7: Improvement heatmap — try × distance ----------------------
fig, ax = plt.subplots(figsize=(max(8, NUM_DIST * 1.4), 5))
heatmap_data = np.zeros((len(TRIES), NUM_DIST))
for ci, dist in enumerate(DISTANCES):
    for ri, try_n in enumerate(TRIES):
        on  = get_on_for_dist(dist)[on_df["try_number"] == try_n]["travel_time"].mean()
        off = get_off_for_dist(dist)[off_df["try_number"] == try_n]["travel_time"].mean()
        if off and not np.isnan(off):
            heatmap_data[ri, ci] = (off - on) / off * 100

im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=-30, vmax=60)
ax.set_xticks(range(NUM_DIST))
ax.set_xticklabels([f"{int(d)} m" for d in DISTANCES], fontweight="bold")
ax.set_yticks(range(len(TRIES)))
ax.set_yticklabels([f"Try {t}" for t in TRIES])
ax.set_xlabel("Priority Distance Threshold", fontweight="bold")
ax.set_ylabel("Simulation Try", fontweight="bold")
ax.set_title("Travel Time Improvement (%) — Priority ON vs OFF\nper Try and Distance Threshold",
             fontweight="bold")
for ri in range(len(TRIES)):
    for ci in range(NUM_DIST):
        v = heatmap_data[ri, ci]
        ax.text(ci, ri, f"{v:.1f}%", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="black" if -15 < v < 45 else "white")
plt.colorbar(im, ax=ax, label="Improvement (%)")
plt.tight_layout()
savefig("heatmap_improvement_try_x_distance")

# ---- Figure 8: Per-try trend for best distance ----------------------------
best_row  = stats_df.loc[stats_df["improvement_pct"].idxmax()] if not stats_df.empty else None
best_dist = best_row["priority_distance_m"] if best_row is not None else DISTANCES[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Performance per Try — Best Distance Threshold ({int(best_dist)} m)",
             fontsize=13, fontweight="bold")
sub_best = pd.concat([get_on_for_dist(best_dist), get_off_for_dist(best_dist)])
try_stats = sub_best.groupby(["try_number", "priority_label"])[
    ["travel_time", "stopped_time"]].mean().reset_index()

for ax, (col, ylabel) in zip(axes, [("travel_time", "Mean Travel Time (s)"),
                                     ("stopped_time", "Mean Stopped Time (s)")]):
    for label, color, marker in [("Priority ON", COL_ON, "o"), ("Priority OFF", COL_OFF, "s")]:
        d = try_stats[try_stats["priority_label"] == label]
        ax.plot(d["try_number"], d[col], marker=marker, label=label,
                color=color, linewidth=2, markersize=8)
    ax.set_xlabel("Try Number", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xticks(TRIES)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(f"per_try_trend_best_distance_{int(best_dist)}m")

# ---- Figure 9: Scatter — travel_time vs stopped_time ---------------------
fig, axes = plt.subplots(1, min(NUM_DIST, 4), figsize=(4.5 * min(NUM_DIST, 4), 4), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Travel Time vs Stopped Time by Priority Mode",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES[:4]):
    s_on  = get_on_for_dist(dist)
    s_off = get_off_for_dist(dist)
    for s, label, color in [(s_on, "ON", COL_ON), (s_off, "OFF", COL_OFF)]:
        ax.scatter(s["travel_time"], s["stopped_time"],
                   label=f"Priority {label}", color=color,
                   alpha=0.6, s=60, edgecolors="black", linewidth=0.3)
    corr_on  = s_on["travel_time"].corr(s_on["stopped_time"])
    corr_off = s_off["travel_time"].corr(s_off["stopped_time"])
    ax.text(0.04, 0.97,
            f"r ON ={corr_on:.2f}\nr OFF={corr_off:.2f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.set_title(f"{int(dist)} m", fontweight="bold")
    ax.set_xlabel("Travel Time (s)")
    if ax == axes[0]:
        ax.set_ylabel("Stopped Time (s)", fontweight="bold")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("scatter_travel_vs_stopped")

# ---- Figure 10: Congestion delta comparison -------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Network Congestion Delta: Priority ON vs OFF\n"
             "(Positive = more congestion when EV left vs arrived; Negative = congestion cleared)",
             fontsize=12, fontweight="bold")
frames = []
for dist in DISTANCES:
    for getter in [get_on_for_dist, get_off_for_dist]:
        chunk = getter(dist).copy()
        chunk["distance_label"] = f"{int(dist)} m"
        frames.append(chunk)
plot_data = pd.concat(frames, ignore_index=True)
sns.boxplot(data=plot_data, x="distance_label", y="congestion_delta_pct",
            hue="priority_label", palette=PALETTE,
            order=[f"{int(d)} m" for d in DISTANCES],
            hue_order=["Priority ON", "Priority OFF"], ax=ax)
ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
ax.set_xlabel("Priority Distance Threshold", fontweight="bold")
ax.set_ylabel("Congestion Delta (%)", fontweight="bold")
ax.legend(title="")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
savefig("congestion_delta_comparison")

# ---- Figure 11: Statistical tests summary table ---------------------------
if not stats_df.empty:
    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.6 * len(stats_df)))
    ax.axis("off")
    display_cols = ["priority_distance_m", "n_pairs", "mean_on_s", "mean_off_s",
                    "improvement_pct", "paired_t", "p_ttest", "p_wilcoxon",
                    "cohens_d", "significant"]
    col_labels = ["Dist (m)", "Pairs", "Mean ON (s)", "Mean OFF (s)",
                  "Improvement %", "t-stat", "p (t-test)", "p (Wilcoxon)",
                  "Cohen's d", "Sig?"]
    tbl = ax.table(
        cellText=stats_df[display_cols].round(4).values.tolist(),
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(stats_df) + 1):
        row_data = stats_df.iloc[i - 1]
        bg = "#d5f5e3" if "YES" in str(row_data["significant"]) else "#fadbd8"
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(bg)
    ax.set_title("Paired Statistical Tests: Priority ON vs OFF (Travel Time)",
                 fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()
    savefig("statistical_tests_table")

# ---- Figure 12: Effect size (Cohen's d) chart ----------------------------
if not stats_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bar = [COL_ON if d > 0 else COL_OFF for d in stats_df["cohens_d"]]
    bars = ax.bar([f"{int(d)} m" for d in stats_df["priority_distance_m"]],
                  stats_df["cohens_d"], color=colors_bar, alpha=0.85, edgecolor="black")
    ax.axhline(0,    color="black", linestyle="-",  linewidth=1)
    ax.axhline(0.2,  color="grey",  linestyle="--", linewidth=1, alpha=0.6, label="Small  (0.2)")
    ax.axhline(0.5,  color="grey",  linestyle="-.", linewidth=1, alpha=0.6, label="Medium (0.5)")
    ax.axhline(0.8,  color="grey",  linestyle=":",  linewidth=1, alpha=0.6, label="Large  (0.8)")
    ax.axhline(-0.2, color="grey",  linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(-0.5, color="grey",  linestyle="-.", linewidth=1, alpha=0.6)
    ax.axhline(-0.8, color="grey",  linestyle=":",  linewidth=1, alpha=0.6)
    for bar, val in zip(bars, stats_df["cohens_d"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.02 if val >= 0 else -0.06),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontweight="bold", fontsize=10)
    ax.set_xlabel("Priority Distance Threshold", fontweight="bold")
    ax.set_ylabel("Cohen's d  (ON − OFF, travel time)", fontweight="bold")
    ax.set_title("Effect Size by Distance Threshold\n"
                 "Negative = Priority ON reduces travel time", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savefig("effect_size_cohens_d")

# ---- Figure 13: Preemption events analysis --------------------------------
if Path(PREEMPT_FILE).exists():
    pe = pd.read_csv(PREEMPT_FILE)
    print(f"\nLoaded {len(pe)} preemption events from {PREEMPT_FILE}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Traffic Light Pre-emption Events", fontsize=13, fontweight="bold")

    # Top intersections
    ax = axes[0]
    top_tls = pe["tlid"].value_counts().head(12)
    short_labels = [str(t)[:22] + ("…" if len(str(t)) > 22 else "") for t in top_tls.index]
    ax.barh(range(len(top_tls)), top_tls.values, color=DIST_CMAP(0.6), edgecolor="black")
    ax.set_yticks(range(len(top_tls)))
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Pre-emption Events", fontweight="bold")
    ax.set_title("Top Pre-empted Intersections (all distances)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Events by distance
    ax = axes[1]
    events_by_dist = pe.groupby("priority_distance").size()
    ax.bar([f"{int(d)} m" for d in events_by_dist.index], events_by_dist.values,
           color=[dist_colors[d] for d in events_by_dist.index], edgecolor="black", alpha=0.85)
    ax.set_xlabel("Priority Distance Threshold", fontweight="bold")
    ax.set_ylabel("Total Pre-emption Events", fontweight="bold")
    ax.set_title("Pre-emption Volume by Distance Threshold", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(ax.patches, events_by_dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    savefig("preemption_events_analysis")
else:
    print(f"\n  (Skipping preemption figure — {PREEMPT_FILE} not found)")

# ---- Figure 14: Average speed comparison (normalised for route length) -----
fig, axes = plt.subplots(1, NUM_DIST, figsize=(4 * NUM_DIST, 5), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Average Speed: Priority ON vs OFF by Distance Threshold\n"
             "(normalises for route length differences between vehicles)",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES):
    groups = [get_on_for_dist(dist)["avg_speed_ms"].dropna().values,
              get_off_for_dist(dist)["avg_speed_ms"].dropna().values]
    bp = ax.boxplot(groups, labels=["ON", "OFF"], patch_artist=True, widths=0.5)
    for patch, col in zip(bp["boxes"], [COL_ON, COL_OFF]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for line in bp["medians"]:
        line.set_color("black"); line.set_linewidth(2)
    ax.plot([1, 2], [g.mean() for g in groups], "D", color="navy",
            markersize=7, zorder=5, label="Mean")
    ax.set_title(f"{int(dist)} m threshold", fontweight="bold")
    ax.set_xlabel("Priority Mode")
    if ax == axes[0]:
        ax.set_ylabel("Average Speed (m/s)", fontweight="bold")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("avg_speed_by_distance")

# ---- Figure 15: Per-vehicle improvement breakdown --------------------------
fig, axes = plt.subplots(1, NUM_DIST, figsize=(max(10, 3 * NUM_DIST), 5), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Travel Time Improvement per Vehicle ID\n"
             "(reveals whether improvement is consistent or driven by outliers)",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES):
    on_g  = get_on_for_dist(dist).groupby("id")["travel_time"].mean()
    off_g = get_off_for_dist(dist).groupby("id")["travel_time"].mean()
    merged = pd.DataFrame({"on": on_g, "off": off_g}).dropna()
    merged["imp_pct"] = (merged["off"] - merged["on"]) / merged["off"] * 100
    merged = merged.sort_index()
    short_ids = [v.replace("emergency_", "EV ") for v in merged.index]
    colors = [COL_ON if v >= 0 else COL_OFF for v in merged["imp_pct"]]
    ax.barh(short_ids, merged["imp_pct"], color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"{int(dist)} m", fontweight="bold")
    ax.set_xlabel("Improvement (%)")
    if ax == axes[0]:
        ax.set_ylabel("Vehicle", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
savefig("per_vehicle_improvement_breakdown")

# ---- Figure 16: Congestion vs improvement scatter --------------------------
fig, axes = plt.subplots(1, NUM_DIST, figsize=(4.5 * NUM_DIST, 4), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Does Priority Help More in Congested Conditions?\n"
             "Travel Time Improvement vs Traffic Density at Departure",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES):
    on_sub  = get_on_for_dist(dist)[["try_number", "id", "travel_time"]].rename(
        columns={"travel_time": "tt_on"})
    off_sub = get_off_for_dist(dist)[["try_number", "id", "travel_time",
                                       "traffic_at_start_pct"]].rename(
        columns={"travel_time": "tt_off"})
    merged = on_sub.merge(off_sub, on=["try_number", "id"]).dropna()
    merged["imp_pct"] = (merged["tt_off"] - merged["tt_on"]) / merged["tt_off"] * 100
    ax.scatter(merged["traffic_at_start_pct"], merged["imp_pct"],
               color=dist_colors[dist], alpha=0.6, s=55,
               edgecolors="black", linewidth=0.3)
    if len(merged) > 2:
        m, b, r, p, _ = stats.linregress(merged["traffic_at_start_pct"], merged["imp_pct"])
        xs = np.linspace(merged["traffic_at_start_pct"].min(),
                         merged["traffic_at_start_pct"].max(), 100)
        ax.plot(xs, m * xs + b, color="black", linewidth=1.5, linestyle="--")
        ax.text(0.05, 0.95, f"r={r:.2f}  p={p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"{int(dist)} m", fontweight="bold")
    ax.set_xlabel("Traffic Density at Departure (%)")
    if ax == axes[0]:
        ax.set_ylabel("Travel Time Improvement (%)", fontweight="bold")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("congestion_vs_improvement_scatter")

# ---- Figure 17: Stopped time as % of travel time --------------------------
fig, axes = plt.subplots(1, NUM_DIST, figsize=(4 * NUM_DIST, 5), sharey=True)
if NUM_DIST == 1:
    axes = [axes]
fig.suptitle("Stopped Time as % of Total Travel Time: Priority ON vs OFF\n"
             "(normalised metric — independent of route length)",
             fontsize=13, fontweight="bold")
for ax, dist in zip(axes, DISTANCES):
    groups = [get_on_for_dist(dist)["stopped_pct"].dropna().values,
              get_off_for_dist(dist)["stopped_pct"].dropna().values]
    bp = ax.boxplot(groups, labels=["ON", "OFF"], patch_artist=True, widths=0.5)
    for patch, col in zip(bp["boxes"], [COL_ON, COL_OFF]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for line in bp["medians"]:
        line.set_color("black"); line.set_linewidth(2)
    ax.plot([1, 2], [g.mean() for g in groups], "D", color="navy",
            markersize=7, zorder=5, label="Mean")
    ax.set_title(f"{int(dist)} m threshold", fontweight="bold")
    ax.set_xlabel("Priority Mode")
    if ax == axes[0]:
        ax.set_ylabel("Stopped Time (% of travel time)", fontweight="bold")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("stopped_pct_of_travel_time")

# ---- Figure 18: Background traffic travel time — ON vs OFF -----------------
if bg_df is not None and len(bg_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Background Traffic Impact of Emergency Vehicle Preemption\n"
                 "(does TL preemption delay cross-traffic?)",
                 fontsize=13, fontweight="bold")

    # Left: boxplot of bg travel time by priority mode (all distances)
    ax = axes[0]
    bg_on  = bg_df[bg_df["priority_enabled"]]["travel_time"]
    bg_off = bg_df[~bg_df["priority_enabled"]]["travel_time"]
    bp = ax.boxplot([bg_on.values, bg_off.values], labels=["Priority ON", "Priority OFF"],
                    patch_artist=True, widths=0.5)
    for patch, col in zip(bp["boxes"], [COL_ON, COL_OFF]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for line in bp["medians"]:
        line.set_color("black"); line.set_linewidth(2)
    ax.plot([1, 2], [bg_on.mean(), bg_off.mean()], "D", color="navy",
            markersize=8, zorder=5, label="Mean")
    ax.set_ylabel("Background Vehicle Travel Time (s)", fontweight="bold")
    ax.set_title("All Distances Combined")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    mean_diff = bg_on.mean() - bg_off.mean()
    pct_diff  = mean_diff / bg_off.mean() * 100 if bg_off.mean() > 0 else 0
    ax.text(0.5, 0.97, f"Mean delay: {mean_diff:+.1f}s ({pct_diff:+.1f}%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Right: mean bg travel time by priority_distance
    ax = axes[1]
    bg_by_dist = bg_df.groupby(["priority_distance", "priority_enabled"])["travel_time"].mean().reset_index()
    bg_on_d  = bg_by_dist[bg_by_dist["priority_enabled"]].set_index("priority_distance")
    bg_off_d = bg_by_dist[~bg_by_dist["priority_enabled"]].set_index("priority_distance")
    dist_vals = sorted(bg_by_dist["priority_distance"].unique())
    x_bg = np.arange(len(dist_vals))
    w_bg = 0.35
    ax.bar(x_bg - w_bg/2,
           [bg_on_d.loc[d, "travel_time"] if d in bg_on_d.index else np.nan for d in dist_vals],
           w_bg, label="Priority ON", color=COL_ON, alpha=0.85, edgecolor="black")
    ax.bar(x_bg + w_bg/2,
           [bg_off_d.loc[d, "travel_time"] if d in bg_off_d.index else np.nan for d in dist_vals],
           w_bg, label="Priority OFF", color=COL_OFF, alpha=0.85, edgecolor="black")
    ax.set_xticks(x_bg)
    ax.set_xticklabels([f"{int(d)} m" if d != 0 else "OFF" for d in dist_vals])
    ax.set_xlabel("Priority Distance Threshold", fontweight="bold")
    ax.set_ylabel("Mean Travel Time (s)", fontweight="bold")
    ax.set_title("Background Traffic by Distance Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    savefig("background_traffic_delay")
else:
    print("  (Skipping Figure 18 — no background metrics)")

# ---- Figure 19: TL count vs travel time improvement scatter ----------------
if HAS_TL_COUNT and not on_df["tl_count"].isna().all():
    best_dist_tl = best_dist if not stats_df.empty else DISTANCES[0]
    on_tl = get_on_for_dist(best_dist_tl).copy()
    off_tl = get_off_for_dist(best_dist_tl)[["try_number", "id", "travel_time"]].rename(
        columns={"travel_time": "tt_off"})
    on_tl = on_tl.merge(off_tl, on=["try_number", "id"]).dropna(subset=["tl_count", "travel_time"])
    on_tl["imp_pct"] = (on_tl["tt_off"] - on_tl["travel_time"]) / on_tl["tt_off"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(on_tl["tl_count"], on_tl["imp_pct"],
                    c=on_tl["tl_count"], cmap="viridis", s=70,
                    edgecolors="black", linewidth=0.4, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="TL count")
    if len(on_tl) > 2:
        m, b, r, p, _ = stats.linregress(on_tl["tl_count"], on_tl["imp_pct"])
        xs = np.linspace(on_tl["tl_count"].min(), on_tl["tl_count"].max(), 100)
        ax.plot(xs, m * xs + b, color="black", linewidth=1.5, linestyle="--")
        ax.text(0.05, 0.95, f"r={r:.2f}  p={p:.3f}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of Traffic Lights Crossed", fontweight="bold")
    ax.set_ylabel("Travel Time Improvement (%)", fontweight="bold")
    ax.set_title(f"Do More Traffic Lights = More Benefit?\n"
                 f"(Priority ON, distance={int(best_dist_tl)} m)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("tl_count_vs_improvement")
else:
    print("  (Skipping Figure 19 — tl_count column not available)")

# ---- Figure 20: Strategy comparison bar chart ------------------------------
if len(STRATEGIES) > 1:
    strat_rows = []
    for strat in STRATEGIES:
        s_on = on_df[on_df["preemption_strategy"] == strat]["travel_time"]
        strat_rows.append({
            "strategy": strat,
            "mean_tt": s_on.mean(),
            "std_tt": s_on.std(),
            "mean_stop": on_df[on_df["preemption_strategy"] == strat]["stopped_time"].mean(),
        })
    strat_plot = pd.DataFrame(strat_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Preemption Strategy Comparison\n"
                 "(same distance threshold, identical traffic conditions)",
                 fontsize=13, fontweight="bold")
    off_mean_tt   = off_df["travel_time"].mean()
    off_mean_stop = off_df["stopped_time"].mean()

    for ax, (col, ylabel, ref) in zip(
            axes,
            [("mean_tt",   "Mean Travel Time (s)",  off_mean_tt),
             ("mean_stop", "Mean Stopped Time (s)", off_mean_stop)]):
        bars = ax.bar(strat_plot["strategy"], strat_plot[col],
                      color=DIST_CMAP(np.linspace(0.2, 0.8, len(STRATEGIES))),
                      edgecolor="black", alpha=0.85)
        if col == "mean_tt":
            ax.errorbar(strat_plot["strategy"], strat_plot[col],
                        yerr=strat_plot["std_tt"], fmt="none",
                        capsize=5, color="black", linewidth=1.5)
        ax.axhline(ref, color=COL_OFF, linestyle="--", linewidth=1.5, label="Priority OFF baseline")
        for bar, val in zip(bars, strat_plot[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}s", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xlabel("Preemption Strategy", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    savefig("strategy_comparison")
else:
    print(f"  (Skipping Figure 20 — only one strategy: {STRATEGIES}. "
          "Add more strategies to PREEMPTION_STRATEGIES in V22.CLAUDE.py)")

# ---- Figure 21: Hold duration sensitivity ----------------------------------
if len(HOLD_VALS) > 1:
    hold_rows = []
    for hold in HOLD_VALS:
        h_on = on_df[on_df["hold_duration"] == hold]["travel_time"]
        hold_rows.append({
            "hold_duration": hold,
            "mean_tt":   h_on.mean(),
            "std_tt":    h_on.std(),
            "mean_stop": on_df[on_df["hold_duration"] == hold]["stopped_time"].mean(),
        })
    hold_plot = pd.DataFrame(hold_rows).sort_values("hold_duration")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(hold_plot["hold_duration"], hold_plot["mean_tt"],
                yerr=hold_plot["std_tt"],
                fmt="o-", color="#3498db", linewidth=2.5, markersize=9,
                markerfacecolor="white", markeredgewidth=2.5,
                capsize=5, label="Mean ± SD travel time")
    ax.axhline(off_df["travel_time"].mean(), color=COL_OFF, linestyle="--",
               linewidth=1.5, label="Priority OFF baseline")
    ax.set_xlabel("TL Hold Duration (s)", fontweight="bold")
    ax.set_ylabel("Mean Travel Time (s)", fontweight="bold")
    ax.set_title("Sensitivity of EV Travel Time to TL Green Hold Duration",
                 fontweight="bold")
    ax.set_xticks(HOLD_VALS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("hold_duration_sensitivity")
else:
    print(f"  (Skipping Figure 21 — only one hold duration: {HOLD_VALS}. "
          "Extend HOLD_DURATIONS in V22.CLAUDE.py)")

# ---- Figure 22: Demand scenario comparison ---------------------------------
if len(DEMAND_NAMES) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Demand Scenario Comparison\n"
                 "(does preemption benefit vary with traffic density?)",
                 fontsize=13, fontweight="bold")

    for ax, (col, ylabel) in zip(axes,
            [("travel_time",  "Mean Travel Time (s)"),
             ("stopped_time", "Mean Stopped Time (s)")]):
        for i, dist in enumerate(DISTANCES):
            imp_by_demand = []
            for d_name in DEMAND_NAMES:
                on_chunk  = on_df[(on_df["priority_distance"] == dist) &
                                  (on_df["demand_scenario"] == d_name)][col]
                off_chunk = off_df[off_df["demand_scenario"] == d_name][col]
                imp_by_demand.append({
                    "demand": d_name,
                    "on_mean": on_chunk.mean(),
                    "off_mean": off_chunk.mean(),
                })
            demand_df = pd.DataFrame(imp_by_demand)
            ax.plot(demand_df["demand"], demand_df["on_mean"], "o-",
                    color=dist_colors[dist], linewidth=2, markersize=8,
                    label=f"{int(dist)} m ON")
            ax.plot(demand_df["demand"], demand_df["off_mean"], "s--",
                    color=dist_colors[dist], linewidth=1.5, markersize=7,
                    alpha=0.5, label=f"{int(dist)} m OFF")
        ax.set_xlabel("Demand Scenario", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("demand_scenario_comparison")
else:
    print(f"  (Skipping Figure 22 — only one demand scenario: {DEMAND_NAMES}. "
          "Extend DEMAND_SCENARIOS in V22.CLAUDE.py)")

# ---------------------------------------------------------------------------
# 6. CSV REPORTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GENERATING CSV REPORTS")
print("=" * 70)

# Report 1: summary by (distance, mode)
r1_rows = []
for dist in DISTANCES:
    for getter, label in [(get_on_for_dist, "Priority ON"), (get_off_for_dist, "Priority OFF")]:
        chunk = getter(dist)[["travel_time", "stopped_time", "congestion_delta_pct"]].agg(
            ["count", "mean", "median", "std", "min", "max"]).round(3).T
        chunk.insert(0, "priority_distance_m", dist)
        chunk.insert(1, "mode", label)
        r1_rows.append(chunk)
pd.concat(r1_rows).to_csv(f"{OUTPUT_DIR}/report_01_summary_by_distance_mode.csv")
print(f"  Saved: report_01_summary_by_distance_mode.csv")

# Report 2: summary by (try, distance, mode)
r2_rows = []
for dist in DISTANCES:
    for getter, label in [(get_on_for_dist, "Priority ON"), (get_off_for_dist, "Priority OFF")]:
        chunk = getter(dist).copy()
        chunk["priority_distance_m"] = dist
        chunk["mode"] = label
        r2_rows.append(chunk)
r2 = (
    pd.concat(r2_rows)
    .groupby(["try_number", "priority_distance_m", "mode"])[["travel_time", "stopped_time"]]
    .agg(["count", "mean", "std"])
    .round(3)
)
r2.to_csv(f"{OUTPUT_DIR}/report_02_summary_by_try_distance_mode.csv")
print(f"  Saved: report_02_summary_by_try_distance_mode.csv")

# Report 3: paired per-vehicle comparison (ON vs OFF) per distance
r3_rows = []
for dist in DISTANCES:
    sub_on  = get_on_for_dist(dist)[["try_number", "id", "travel_time", "stopped_time"]].rename(
        columns={"travel_time": "travel_time_on", "stopped_time": "stopped_time_on"})
    sub_off = get_off_for_dist(dist)[["try_number", "id", "travel_time", "stopped_time"]].rename(
        columns={"travel_time": "travel_time_off", "stopped_time": "stopped_time_off"})
    merged = sub_on.merge(sub_off, on=["try_number", "id"]).round(2)
    merged["priority_distance_m"] = dist
    merged["travel_time_delta"] = (merged["travel_time_off"] - merged["travel_time_on"]).round(2)
    r3_rows.append(merged)
if r3_rows:
    pd.concat(r3_rows, ignore_index=True).to_csv(
        f"{OUTPUT_DIR}/report_03_paired_vehicle_comparison.csv", index=False)
    print(f"  Saved: report_03_paired_vehicle_comparison.csv")

# Report 4: statistical tests
if not stats_df.empty:
    stats_df.to_csv(f"{OUTPUT_DIR}/report_04_statistical_tests.csv", index=False)
    print(f"  Saved: report_04_statistical_tests.csv")

# Report 5: preemption summary
if Path(PREEMPT_FILE).exists():
    pe_summary = (
        pd.read_csv(PREEMPT_FILE)
        .groupby(["tlid", "priority_distance"])
        .agg(event_count=("vid", "count"),
             unique_vehicles=("vid", "nunique"),
             mean_dist_m=("dist_m", "mean"),
             mean_time_s=("time_s", "mean"))
        .round(2)
        .sort_values("event_count", ascending=False)
    )
    pe_summary.to_csv(f"{OUTPUT_DIR}/report_05_preemption_by_intersection.csv")
    print(f"  Saved: report_05_preemption_by_intersection.csv")

# ---------------------------------------------------------------------------
# 7. FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Figures generated : {fig_count}")
print(f"  Output directory  : {OUTPUT_DIR}/")
print(f"\nKey findings:")
print(f"  Records analysed  : {len(df)}")
print(f"  Tries             : {len(TRIES)}  ({min(TRIES)}–{max(TRIES)})")
print(f"  Distance thresholds: {[int(d) for d in DISTANCES]} m")
print(f"  Demand scenarios  : {DEMAND_NAMES}")
print(f"  Strategies        : {STRATEGIES}")
print(f"  Hold durations    : {HOLD_VALS} s")

if not stats_df.empty:
    best = stats_df.loc[stats_df["improvement_pct"].idxmax()]
    worst = stats_df.loc[stats_df["improvement_pct"].idxmin()]
    print(f"\n  Best distance  : {int(best['priority_distance_m'])} m "
          f"→ {best['improvement_pct']:+.1f}% travel time improvement "
          f"(p={best['p_ttest']:.4f}, d={best['cohens_d']:.3f})")
    print(f"  Worst distance : {int(worst['priority_distance_m'])} m "
          f"→ {worst['improvement_pct']:+.1f}% travel time improvement")

overall_on  = on_df["travel_time"].mean()
overall_off = off_df["travel_time"].mean()
overall_imp = (overall_off - overall_on) / overall_off * 100
print(f"\n  Overall mean travel time — ON: {overall_on:.1f}s  OFF: {overall_off:.1f}s  "
      f"Δ={overall_imp:+.1f}%")
print("=" * 70)
