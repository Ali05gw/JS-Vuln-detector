# Project Pipeline — JS Vulnerability Detector

> This document maps every ML pipeline step to the exact code that implements it,
> explains what was done in plain English, and tracks what still needs to be done.
> Written to help teammates understand the project and to answer professor questions.

---

## The Standard ML Pipeline and How It Maps Here

```
Raw CSVs → Clean → EDA → Feature Engineering → Normalize → Train → Evaluate → Demo
```

---

## STEP 1 — Data Loading & Cleaning ✅ DONE

### What it means
Load the raw data files, remove rows that are broken/duplicate, fix any type issues.

### What we did
- Loaded both CSV files:
  - `JSVulnerabilityDataSet-1.0.csv` — Ferenc et al., 12,125 rows, 44 static code metrics
  - `pm_uom_all_full.csv` — Viszkok et al., 8,038 rows, 78 columns (44 shared + 34 extra)
- Removed **exact duplicate rows** (same function measured twice with identical numbers)
- Missing values handled with `.fillna(0)` before training — appropriate because these are code count metrics where missing = zero occurrences

### Where in the code — `build_datasets.py`

```python
# Lines 46–47: loading both files
ferenc  = pd.read_csv(FERENC_PATH)
viszkok = pd.read_csv(VISZKOK_PATH)

# Lines 96–99: removing exact duplicate rows
ferenc_exact_dupes  = int(ferenc.duplicated().sum())
viszkok_exact_dupes = int(viszkok.duplicated().sum())
ferenc_dedup  = ferenc.drop_duplicates().copy()
viszkok_dedup = viszkok.drop_duplicates().copy()
```

---

## STEP 2 — EDA (Exploratory Data Analysis) ✅ DONE

### What it means
Understand the data before touching it — check class balance, find duplicates across files, spot problems.

### What we found

| Finding | Detail |
|---|---|
| **Class imbalance** | ~12% vulnerable, ~88% not vulnerable. A model always saying "not vulnerable" scores 88% accuracy — totally useless. This is why we use PR-AUC and MCC instead of accuracy. |
| **Dataset overlap** | All 8,038 Viszkok rows also appear in Ferenc. Viszkok re-analyzed 73 of Ferenc's 93 repos and added extra columns. They are NOT independent datasets. |
| **4,654 cross-file duplicate functions** | Same function (matched by repo + file path + line range) found in both files. Labels agree >99% of the time (~10 mismatches logged). |
| **20 genuinely unseen repos** | Ferenc has 20 repos Viszkok never touched — these became our **holdout test set**. |

### Where in the code — `build_datasets.py`

```python
# Lines 56–67: extract repo names, find overlap and Ferenc-only repos
overlap_repos     = ferenc_repos & viszkok_repos   # 73 repos shared
ferenc_only_repos = ferenc_repos - viszkok_repos   # 20 holdout repos

# Lines 84–91: build function identity key (repo + file path + line range)
# used to find the same function across both files
ferenc["func_key"] = (
    ferenc["repo_key"] + "|" + ferenc["path"] + "|"
    + ferenc["line"].astype(str) + "|" + ferenc["endline"].astype(str)
)

# Lines 111–122: find cross-file duplicates, check label agreement
common_keys   = f_by_key.index.intersection(v_by_key.index)   # 4,654 matches
mismatch_mask = (vuln_f.values != vuln_v.values)               # ~10 disagreements
```

---

## STEP 3 — Feature Engineering ✅ DONE

### What it means
Decide which columns to keep, drop, create, or transform. Bad features make models learn the wrong thing.

---

### Decision 1: Drop identity/position columns

**Columns dropped:** `line`, `endline`, `column`, `endcolumn`, `name`, `path`, `longname`, `full_repo_path`, `hash`, `type`

**Why:** These describe *where* a function sits in a file — not *what the code does*.
When we ran correlation analysis (Step 5), these columns correlated with the vulnerability
label **more strongly than any real complexity metric**. This sounds useful but is actually
a trap — it means the model would learn "functions around line 50 tend to be vulnerable"
rather than "complex functions tend to be vulnerable." This is called a **collection artifact**
(an accident of how the data was collected, not a real pattern). Keeping them would make
results look great on paper but be completely useless on new codebases.

**Where in the code — `build_datasets.py`**

```python
# Lines 33–37: define the list of columns to drop
IDENTITY_COLS = [
    "line", "endline", "column", "endcolumn",
    "start_line", "end_line", "start_column", "end_column",
    "name", "path", "longname", "full_repo_path", "hash", "type",
]

# Lines 148–150: drop them from Dataset A
dataset_a = ferenc_dedup.drop(
    columns=[c for c in IDENTITY_COLS if c in ferenc_dedup.columns] + ["repo_key", "func_key"]
)

# Lines 168–170: drop them from Dataset B
dataset_b = dataset_b.drop(
    columns=[c for c in IDENTITY_COLS if c in dataset_b.columns] + ["repo_key", "func_key"]
)
```

---

### Decision 2: Build Dataset A — 44 shared metrics, all 93 repos

**What it is:** The 44 static code metrics that exist in BOTH datasets, covering all 93 repos.
A `holdout` column marks the 20 Ferenc-only repos — never used in training, only final testing.

**Why this design:** Only these 44 columns exist for ALL 93 repos, so we can train on 73 repos
and test on 20 holdout repos using the same feature set.

**Where in the code — `build_datasets.py` Lines 146–158**

```python
# Mark which rows belong to the holdout set
ferenc_dedup["holdout"] = ferenc_dedup["repo_key"].isin(ferenc_only_repos)

# Drop identity cols → Dataset A
dataset_a = ferenc_dedup.drop(
    columns=[c for c in IDENTITY_COLS if c in ferenc_dedup.columns] + ["repo_key", "func_key"]
)
# Result: 44 feature columns + "Vuln" + "repo" + "holdout"
```

---

### Decision 3: Build Dataset B — 78 columns, 73 repos only

**What it is:** All 78 columns (44 shared + 34 extra Viszkok metrics like code churn, ESLint
warning counts), restricted to the 73 repos Viszkok analyzed. The 20 holdout repos are excluded
because the 34 extra columns simply don't exist for them.

**Why:** To answer — do the extra 34 columns actually improve predictions?
This is our secondary experiment (ablation study).

**Where in the code — `build_datasets.py` Lines 167–170**

```python
# Only keep the 73 overlapping repos
dataset_b = viszkok_dedup[viszkok_dedup["repo_key"].isin(overlap_repos)].copy()
dataset_b = dataset_b.drop(
    columns=[c for c in IDENTITY_COLS if c in dataset_b.columns] + ["repo_key", "func_key"]
)
# Result: 78 feature columns + "Vuln" + "repo"
```

---

## STEP 4 — Normalization ✅ DONE

### What it means
Scale numbers so a feature with values 0–10,000 doesn't dominate one with values 0–1.
No one-hot encoding needed here — all features are already numeric.

### What we did
- **For CNN models:** `MinMaxScaler` scales every feature to [0, 1]
  - Fitted ONLY on training data, then applied to test data (fitting on full data = data leakage — a common mistake)
- **For XGBoost / Random Forest / Bagging:** No scaling needed — tree-based models split on thresholds and are unaffected by feature scale

### Where in the code — `JS_Vulnerability_Pipeline.ipynb`

Search for **`run_stacked_cnn_cv`** in the notebook (Ctrl+F). The scaling is inside that function, applied per fold:

```python
def run_stacked_cnn_cv(X, y, groups, n_splits=5):
    ...
    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)   # fit + transform on TRAIN fold only
    X_te_scaled = scaler.transform(X_te)        # transform only on TEST fold (no re-fitting)
```

Note: the scaling lives *inside* the training function, not as a standalone cell, because it must be fitted fresh for each fold separately — doing it once upfront would be data leakage.

---

## STEP 5 — Correlation Analysis ✅ DONE (PNGs saved in repo)

### What it means
Check how strongly each feature relates to the vulnerability label.
A correlation close to +1 = strong positive relationship. Close to 0 = no relationship.

### How correlation is calculated
**Pearson correlation** — a standard statistical measure returning a value between -1 and +1.
The code computes the full matrix (every feature vs every feature), then extracts just the
`Vuln` column to rank features by predictive strength.

### What the results showed
- **Top predictors:** cyclomatic complexity (`CC`), Halstead difficulty (`HDIFF`), Halstead volume (`HVOL`), lines of code (`LOC`) — complexity-related metrics positively correlate with vulnerability
- **`holdout` and `repo` columns excluded** from the analysis — they are metadata, not features
- **Some features had undefined correlation** (zero variance — same value for every function) — reported separately, not ranked as "weak"

### Where in the code — `build_datasets.py` Lines 186–261

```python
def correlation_analysis(dataset, tag, exclude=("repo", "holdout")):
    numeric        = dataset.select_dtypes(include=[np.number])  # numeric columns only
    corr_matrix    = numeric.corr()                              # full feature-vs-feature matrix
    corr_with_vuln = corr_matrix["Vuln"].drop("Vuln")           # extract just the Vuln column
    # sort by absolute value — strongest relationships (+ or -) come first
    corr_with_vuln = corr_with_vuln.reindex(
        corr_with_vuln.abs().sort_values(ascending=False).index
    )
    # saves bar chart (top 15 features) and full heatmap

corr_a = correlation_analysis(dataset_a, "A")   # → correlation_with_vuln_A.png, correlation_heatmap_A.png
corr_b = correlation_analysis(dataset_b, "B")   # → correlation_with_vuln_B.png, correlation_heatmap_B.png
```

**Output files already in repo:** `correlation_with_vuln_A.png`, `correlation_with_vuln_B.png`,
`correlation_heatmap_A.png`, `correlation_heatmap_B.png`

---

## STEP 6 — Class Imbalance Handling ✅ CODED

### What it means
With 88% "not vulnerable" and 12% "vulnerable," a model that always says "not vulnerable"
gets 88% accuracy but is completely useless. We force the model to learn the rare class.

### Strategies implemented and compared

| Strategy | How it works | Role |
|---|---|---|
| **Class-weighted loss** | Errors on vulnerable functions are penalized more heavily | Primary |
| **Focal loss** | Down-weights easy majority-class examples — vulnerable functions get more gradient signal | Primary for CNNs |
| **SMOTE** | Synthetically creates new minority-class examples by interpolating between real ones | Secondary comparison only |
| **Gaussian noise oversampling** | Adds small random noise to existing vulnerable examples | Secondary comparison only |

### Where in the code — `JS_Vulnerability_Pipeline.ipynb` Cell 12

```python
def get_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}

def focal_loss(gamma=2.0, alpha=0.25):
    # down-weights easy examples so hard (minority) examples get more gradient
    ...
```

---

## STEP 7 — Model Training ✅ CODED — needs to be run

### Classical baselines (fast, no GPU needed)

| Model | Notes |
|---|---|
| XGBoost | Gradient boosted trees, class weights applied |
| Random Forest | Ensemble of decision trees, class weights applied |
| Bagging | Ensemble of base estimators |

### Stacked CNN ensemble (the paper's architecture)

```
Static code metrics (44 numbers per function)
         |
   +-----+-----+
   v     v     v
VGG16  AlexNet LSTM     <- base learners (trained independently)
(1D)   (1D)
   |     |     |
   +-----+-----+
         |   stacked probability outputs
         v
  Logistic Regression   <- meta-learner (learns which base model to trust)
         |
         v
  Final prediction: Vulnerable (1) or Not (0)
```

**"1D-adapted"** — VGG16 and AlexNet were originally for 2D images. Here they are adapted
to work on a 1D sequence of 44 feature values.

**"Stacking"** — Train 3 models, collect their probability outputs, train a 4th model (LR)
on those outputs. The meta-learner learns when to trust each base model.

### Cross-validation strategy: `StratifiedGroupKFold`

Normal k-fold randomly splits rows — wrong here because functions from the same repo could
appear in both train and test (not a fair test). `StratifiedGroupKFold` keeps all functions
from one repository together AND maintains the 12% vulnerable class ratio in each fold.

### Where in the code — `JS_Vulnerability_Pipeline.ipynb`

- **Cell 17:** `run_baseline_cv()` — XGBoost/RF/Bagging through grouped CV
- **Cell 19:** `build_vgg16()`, `build_alexnet()`, `build_lstm()` — CNN architecture definitions
- **Cell 20:** `run_stacked_cnn_cv()` — trains 3 CNNs per fold, stacks outputs, trains LR meta-learner

---

## STEP 8 — Evaluation ✅ CODED — needs to be run

### Metrics and why we use them

| Metric | Why |
|---|---|
| **PR-AUC** | PRIMARY. Not fooled by class imbalance. Measures how well the model finds the rare vulnerable class. |
| **MCC** | Handles imbalance well. 0 = random guessing, +1 = perfect. |
| **F1** | Harmonic mean of precision and recall — standard for imbalanced problems |
| **Precision** | Of everything flagged as vulnerable, how many actually were? |
| **Recall** | Of all actually vulnerable functions, how many did the model catch? |
| **Accuracy** | Reported but NOT primary — misleading on 88/12 splits |

### Two evaluation experiments

**Experiment 1 — In-distribution CV (within 73 repos):**
5-fold grouped CV within the training pool. Answers: how good is the model on familiar data?

**Experiment 2 — Cross-project holdout (headline result):**
Train on all 73 repos → test ONLY on 20 repos the model has never seen.
Answers: does the model generalize to genuinely new codebases?

### Where in the code — `JS_Vulnerability_Pipeline.ipynb`

- **Cell 14:** `evaluate()` — computes all 6 metrics
- **Cell 22:** baseline CV results table
- **Cell 24:** `train_final_and_test_holdout()` — classical models on holdout
- **Cell 25:** `train_final_stacked_cnn()` — stacked CNN on holdout
- **Cell 27:** generalization gap chart — PR-AUC drop from in-distribution to holdout

---

## STEP 9 — Ablation: 44 vs 78 Features ✅ CODED — needs to be run

Do Viszkok's extra 34 columns (code churn, ESLint warnings) help?
XGBoost is run twice on Dataset B (73 repos) — once with 44 features, once with 78 — PR-AUC compared.

**Where:** `JS_Vulnerability_Pipeline.ipynb` Cell 29

---

## Current Status Summary

| Step | Status | File & Lines |
|---|---|---|
| Data loading & cleaning | ✅ Done | `build_datasets.py` L46–99 |
| EDA (overlap, duplicates, class balance) | ✅ Done | `build_datasets.py` L56–131 |
| Feature engineering (drop identity cols) | ✅ Done | `build_datasets.py` L33–37, 148–170 |
| Build Dataset A (44 metrics, 93 repos) | ✅ Done | `build_datasets.py` L146–158 |
| Build Dataset B (78 cols, 73 repos) | ✅ Done | `build_datasets.py` L167–170 |
| Normalization (MinMaxScaler for CNN) | ✅ Done | `JS_Vulnerability_Pipeline.ipynb` Cell 20 |
| Class imbalance strategies | ✅ Done | `JS_Vulnerability_Pipeline.ipynb` Cell 12 |
| Correlation analysis + heatmaps | ✅ Done (PNGs saved) | `build_datasets.py` L186–261 |
| Classical ML baseline CV | ✅ Coded | `JS_Vulnerability_Pipeline.ipynb` Cell 17 — **RUN IT** |
| Stacked CNN ensemble CV | ✅ Coded | `JS_Vulnerability_Pipeline.ipynb` Cell 20 — **RUN IT** |
| Holdout generalization test | ✅ Coded | `JS_Vulnerability_Pipeline.ipynb` Cell 24–25 — **RUN IT** |
| 44 vs 78 feature ablation | ✅ Coded | `JS_Vulnerability_Pipeline.ipynb` Cell 29 — **RUN IT** |
| Results summary / presentation | Not started | Next priority |
| Demo / dashboard | Not started | Final step |

---

## 11-Day Schedule (Days 4–14)

| Days | Task | Priority |
|---|---|---|
| Day 4–5 | Run `build_datasets.py`, confirm CSVs. Run notebook classical ML CV — get real numbers | Critical |
| Day 6–7 | Run stacked CNN CV (slow — 40 epochs x 5 folds x 3 models). Document results while running | Critical |
| Day 8 | Run holdout test. Plot generalization gap. Compare all models | Critical |
| Day 9 | Run 44 vs 78 ablation. Clean up result tables | Important |
| Day 10 | One clean results summary (all key numbers in one place) | Important |
| Day 11–12 | Presentation slides. Know your numbers. Rehearse Q&A | Important |
| Day 13–14 | Buffer / rehearsal / final polish | Polish |

---

## Professor Q&A — Ready-to-Use Answers

### "What did you analyze in the dataset?"

We analyzed two public JavaScript vulnerability datasets at the function level
(Ferenc et al., 12,125 functions; Viszkok et al., 8,038 functions). Key EDA findings:
(1) The datasets are not independent — 73 of Ferenc's 93 repos were re-analyzed by Viszkok,
preventing a naive "train on one, test on the other" design.
(2) Class imbalance is severe (~12% vulnerable) — we use PR-AUC and MCC as primary metrics
because accuracy is misleading on an 88/12 split.
(3) Position/identity features (line numbers, file paths, function names) were spuriously
correlated with the vulnerability label — stronger than any real complexity metric — so we
removed them to prevent the model memorizing collection artifacts.
(4) 4,654 functions appeared in both files with >99% label agreement.

### "What was your feature engineering process?"

Three decisions:
(1) We dropped all identity/position columns because they correlated with the vulnerability
label for non-causal reasons — keeping them would cause the model to learn file-position
patterns, not code quality patterns.
(2) We identified the 44 static code metrics shared between both datasets (cyclomatic
complexity, Halstead measures, lines of code, etc.) as our primary feature set.
(3) We built a secondary 78-column feature set adding Viszkok's 34 process/ESLint metrics
to test whether they add predictive value — this is our ablation study.
No categorical encoding was needed since all features are numeric. MinMaxScaling was applied
to CNN inputs only, fitted on training folds only to prevent data leakage.

### "How does your model work?"

We replicate the stacked CNN architecture from Sheneamer (2024): three 1D-adapted deep
learning base learners (VGG16, AlexNet, LSTM) are trained on the static code metrics.
Their probability outputs are stacked and fed into a Logistic Regression meta-learner
that makes the final prediction. We also run classical ML baselines (XGBoost, Random Forest,
Bagging). Cross-validation uses StratifiedGroupKFold — keeping all functions from one
repository together so no repo leaks across train/test folds, while maintaining the 12%
vulnerable class ratio. Our key contribution over the original paper is a cross-project
generalization test: train on 73 repos, evaluate on 20 genuinely unseen repos that were
never in any training fold.
