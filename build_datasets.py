"""
Cross-check Ferenc et al. (JSVulnerabilityDataSet-1.0.csv) against
Viszkok et al. (pm_uom_all_full.csv), and build two SEPARATE, non-merged
modeling datasets:

  Dataset A - 44-shared-metrics pool, ALL 93 repos, one row per unique
              function. Used for the real generalization test: train on
              the 73 repos Viszkok re-analyzed, hold out the 20 repos
              Viszkok never touched.

  Dataset B - full 78-column pool, the 73 overlapping repos ONLY. A
              self-contained side experiment answering "do the extra
              process/warning metrics help", evaluated with CV inside
              these 73 repos - it never sees the 20 holdout repos.

Run: python build_datasets.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FERENC_PATH = "JSVulnerabilityDataSet-1.0.csv"
VISZKOK_PATH = "pm_uom_all_full.csv"

# Palette (validated categorical/diverging hues - blue = positive, red = negative)
BLUE, RED, GRAY, INK, MUTED, GRID = "#2a78d6", "#e34948", "#f0efec", "#0b0b0b", "#898781", "#e1e0d9"

# Identity/position columns that carry no modeling signal - dropped from
# both datasets. Ferenc/Viszkok don't share all of these names, so we
# only drop whichever of them actually exist on a given frame.
IDENTITY_COLS = [
    "line", "endline", "column", "endcolumn",
    "start_line", "end_line", "start_column", "end_column",
    "name", "path", "longname", "full_repo_path", "hash", "type",
]

# ---------------------------------------------------------------------------
# STEP 1 - repo extraction, function keys, cross-file duplicate detection
# ---------------------------------------------------------------------------
print("=" * 78)
print("STEP 1: repo split + cross-file duplicate functions")
print("=" * 78)

ferenc = pd.read_csv(FERENC_PATH)
viszkok = pd.read_csv(VISZKOK_PATH)


def extract_repo(full_repo_path: str) -> str:
    """https://github.com/owner/repo/blob/<hash>/path -> 'owner/repo'."""
    before_blob = full_repo_path.split("/blob/")[0]
    return before_blob.replace("https://github.com/", "")


ferenc["repo"] = ferenc["full_repo_path"].apply(extract_repo)
viszkok["repo"] = viszkok["full_repo_path"].apply(extract_repo)

# GitHub repo names are case-insensitive - match on a lowercase key, but the
# original-cased "repo" column is what we keep in the final datasets.
ferenc["repo_key"] = ferenc["repo"].str.lower()
viszkok["repo_key"] = viszkok["repo"].str.lower()

ferenc_repos = set(ferenc["repo_key"].unique())
viszkok_repos = set(viszkok["repo_key"].unique())
overlap_repos = ferenc_repos & viszkok_repos
ferenc_only_repos = ferenc_repos - viszkok_repos

print(f"Ferenc repos total:  {len(ferenc_repos)}")
print(f"Viszkok repos total: {len(viszkok_repos)}")
print(f"Repos in BOTH (re-analyzed by Viszkok): {len(overlap_repos)}")
print(f"Repos ONLY in Ferenc (holdout candidates): {len(ferenc_only_repos)}")
print("\nFerenc-only repos (the 20-repo holdout set):")
for r in sorted(ferenc_only_repos):
    print(f"  - {r}")
print("\nRepos present in both (the 73-repo overlap set):")
for r in sorted(overlap_repos):
    print(f"  - {r}")

# Function key = repo + file path + line span. Viszkok's start_line/end_line
# are byte-for-byte identical to its own line/endline columns (verified on
# every row), so keying Ferenc on line/endline and Viszkok on
# start_line/end_line lines the two files up correctly.
ferenc["func_key"] = (
    ferenc["repo_key"] + "|" + ferenc["path"] + "|"
    + ferenc["line"].astype(str) + "|" + ferenc["endline"].astype(str)
)
viszkok["func_key"] = (
    viszkok["repo_key"] + "|" + viszkok["path"] + "|"
    + viszkok["start_line"].astype(str) + "|" + viszkok["end_line"].astype(str)
)

# Ferenc contains exact repeated rows (the same function measured twice with
# identical metrics) - collapse those first so a plain re-run of the same
# measurement isn't mistaken for a second, independent function.
ferenc_exact_dupes = int(ferenc.duplicated().sum())
viszkok_exact_dupes = int(viszkok.duplicated().sum())
ferenc_dedup = ferenc.drop_duplicates().copy()
viszkok_dedup = viszkok.drop_duplicates().copy()
print(f"\nExact repeated rows dropped from Ferenc:  {ferenc_exact_dupes}")
print(f"Exact repeated rows dropped from Viszkok: {viszkok_exact_dupes}")

# A handful of Viszkok repos were analyzed across multiple commits (process
# metrics like churn need commit history), so func_key can still repeat
# inside a single file. Take one representative row per key for the
# cross-file match itself; this does not affect Dataset B, which keeps
# every commit snapshot.
f_by_key = ferenc_dedup.drop_duplicates("func_key").set_index("func_key")
v_by_key = viszkok_dedup.drop_duplicates("func_key").set_index("func_key")

common_keys = f_by_key.index.intersection(v_by_key.index)
print(f"\nFunctions found in BOTH files (same repo+path+line range): {len(common_keys)}")

vuln_f = f_by_key.loc[common_keys, "Vuln"]
vuln_v = v_by_key.loc[common_keys, "Vuln"]
mismatch_mask = (vuln_f.values != vuln_v.values)
n_mismatch = int(mismatch_mask.sum())

print(
    f"Vuln label agreement on matched functions: {(1 - n_mismatch / len(common_keys)):.4%} "
    f"({n_mismatch} mismatches out of {len(common_keys)})"
)

if n_mismatch:
    mismatches = pd.DataFrame({
        "func_key": common_keys.to_numpy()[mismatch_mask],
        "Vuln_ferenc": vuln_f.to_numpy()[mismatch_mask],
        "Vuln_viszkok": vuln_v.to_numpy()[mismatch_mask],
    })
    print("\nMismatched labels - printed, NOT auto-resolved:")
    print(mismatches.to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 2 - Dataset A: 44-shared-metrics pool, all 93 repos
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 2: Dataset A (shared metrics, all 93 repos)")
print("=" * 78)

# Ferenc alone already spans all 93 repos and every function in it also
# carries the 44 shared metric columns, so it is the single source for
# Dataset A - a function re-measured by Viszkok is the SAME function, and
# mixing in Viszkok's copy of it would mean picking between two
# measurements of one thing for no benefit. Using Ferenc's own dedup keeps
# Dataset A snapshot-consistent.
ferenc_dedup["holdout"] = ferenc_dedup["repo_key"].isin(ferenc_only_repos)

dataset_a = ferenc_dedup.drop(
    columns=[c for c in IDENTITY_COLS if c in ferenc_dedup.columns] + ["repo_key", "func_key"]
)

print(f"Dataset A: {len(dataset_a)} unique functions, {dataset_a.shape[1]} columns "
      f"(features + Vuln + repo + holdout)")

for flag, label in [(False, "holdout=False (73 overlap repos, train)"),
                     (True, "holdout=True  (20 Ferenc-only repos, held out)")]:
    sub = dataset_a[dataset_a["holdout"] == flag]
    print(f"  {label}: {len(sub):>6,} functions, {sub['Vuln'].mean() * 100:5.2f}% vulnerable")

# ---------------------------------------------------------------------------
# STEP 3 - Dataset B: full 78-column pool, 73 overlapping repos only
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 3: Dataset B (all 78 columns, 73 overlap repos only)")
print("=" * 78)

dataset_b = viszkok_dedup[viszkok_dedup["repo_key"].isin(overlap_repos)].copy()
dataset_b = dataset_b.drop(
    columns=[c for c in IDENTITY_COLS if c in dataset_b.columns] + ["repo_key", "func_key"]
)

print(f"Dataset B: {len(dataset_b)} rows, {dataset_b.shape[1]} columns (features + Vuln + repo)")
print(f"  repos covered: {dataset_b['repo'].str.lower().nunique()} (no holdout repos included)")
print(f"  class balance: {dataset_b['Vuln'].mean() * 100:.2f}% vulnerable")

dataset_a.to_csv("dataset_A_shared_metrics_93repos.csv", index=False)
dataset_b.to_csv("dataset_B_full_columns_73repos.csv", index=False)
print("\nSaved dataset_A_shared_metrics_93repos.csv and dataset_B_full_columns_73repos.csv")

# ---------------------------------------------------------------------------
# STEP 4 - correlation analysis (run once per dataset)
# ---------------------------------------------------------------------------
diverging_cmap = LinearSegmentedColormap.from_list("diverging_blue_red", ["#184f95", GRAY, "#a82323"])


def correlation_analysis(dataset: pd.DataFrame, tag: str, exclude=("repo", "holdout")):
    print("\n" + "=" * 78)
    print(f"STEP 4: correlation analysis - Dataset {tag}")
    print("=" * 78)

    numeric = dataset.drop(columns=[c for c in exclude if c in dataset.columns])
    numeric = numeric.select_dtypes(include=[np.number])
    corr_matrix = numeric.corr()
    corr_with_vuln = corr_matrix["Vuln"].drop("Vuln")

    # A constant column (zero variance) has an undefined (NaN) correlation -
    # that's not "no relationship", it's "not computable". Report it
    # separately instead of letting it silently sort to the bottom as if it
    # were the weakest real signal.
    zero_variance = corr_with_vuln[corr_with_vuln.isna()].index.tolist()
    if zero_variance:
        print(f"Zero-variance columns (correlation undefined, excluded from ranking): {zero_variance}")
    corr_with_vuln = corr_with_vuln.dropna()
    corr_with_vuln = corr_with_vuln.reindex(corr_with_vuln.abs().sort_values(ascending=False).index)

    print(f"Top 15 features by |correlation| with Vuln:")
    print(corr_with_vuln.head(15).to_string())

    # --- bar chart: top 15, sorted by strength, sign shown by color ---
    top15 = corr_with_vuln.head(15).iloc[::-1]  # largest at top of barh
    colors = [BLUE if v >= 0 else RED for v in top15.values]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.barh(top15.index, top15.values, color=colors, height=0.6, zorder=3)
    ax.axvline(0, color="#c3c2b7", linewidth=1, zorder=2)
    ax.set_xlabel("Correlation with Vuln", color=MUTED)
    ax.set_title(f"Dataset {tag}: top 15 correlations with Vuln", color=INK, loc="left", fontsize=13)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    for bar, val in zip(bars, top15.values):
        offset = 3 if val >= 0 else -3
        ha = "left" if val >= 0 else "right"
        ax.annotate(f"{val:+.2f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    textcoords="offset points", xytext=(offset, 0), va="center", ha=ha,
                    color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"correlation_with_vuln_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation_with_vuln_{tag}.png")

    # --- full heatmap: every numeric feature vs every other (multicollinearity) ---
    n = len(corr_matrix)
    size = max(8, min(0.35 * n + 3, 22))
    fig2, ax2 = plt.subplots(figsize=(size, size), facecolor="#fcfcfb")
    heatmap_cmap = diverging_cmap.copy()
    heatmap_cmap.set_bad(GRID)  # zero-variance columns -> undefined correlation
    im = ax2.imshow(corr_matrix.values, cmap=heatmap_cmap, vmin=-1, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    fontsize = 8 if n <= 40 else 5
    ax2.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=fontsize, color=MUTED)
    ax2.set_yticklabels(corr_matrix.columns, fontsize=fontsize, color=MUTED)
    ax2.set_title(f"Dataset {tag}: full correlation heatmap ({n} numeric features)",
                   color=INK, fontsize=13)
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("Correlation", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    fig2.tight_layout()
    fig2.savefig(f"correlation_heatmap_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved correlation_heatmap_{tag}.png")

    return corr_with_vuln


corr_a = correlation_analysis(dataset_a, "A")
corr_b = correlation_analysis(dataset_b, "B")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"Exact duplicate rows removed: {ferenc_exact_dupes} (Ferenc), {viszkok_exact_dupes} (Viszkok)")
print(f"Cross-file duplicate functions matched: {len(common_keys)} "
      f"({n_mismatch} had disagreeing Vuln labels - see printed list above)")
print(f"Repo split: {len(overlap_repos)} overlap repos (train-eligible) / "
      f"{len(ferenc_only_repos)} Ferenc-only repos (holdout)")
print(f"Dataset A: {len(dataset_a)} functions, {dataset_a['Vuln'].mean()*100:.2f}% vulnerable overall "
      f"({dataset_a['holdout'].sum()} held out)")
print(f"Dataset B: {len(dataset_b)} functions, {dataset_b['Vuln'].mean()*100:.2f}% vulnerable overall")
print(f"Dataset A strongest correlations with Vuln: "
      f"{corr_a.index[0]} ({corr_a.iloc[0]:+.3f}), {corr_a.index[1]} ({corr_a.iloc[1]:+.3f})")
print(f"Dataset A weakest correlation with Vuln: {corr_a.index[-1]} ({corr_a.iloc[-1]:+.3f})")
print(f"Dataset B strongest correlations with Vuln: "
      f"{corr_b.index[0]} ({corr_b.iloc[0]:+.3f}), {corr_b.index[1]} ({corr_b.iloc[1]:+.3f})")
print(f"Dataset B weakest correlation with Vuln: {corr_b.index[-1]} ({corr_b.iloc[-1]:+.3f})")
