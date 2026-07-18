# AI-Based JavaScript Vulnerability Detection: Cross-Dataset Generalization Study

## Overview

This project trains machine learning models to flag potentially vulnerable JavaScript
functions using static code metrics (complexity, size, Halstead measures, etc.), and — unlike
prior published work on this exact task — tests whether a model trained on one codebase's
functions actually holds up on a *different* codebase it has never seen. That cross-project
generalization test is the project's core contribution.

Prior work (Sheneamer, 2024) stacks CNN architectures over the same kind of static metrics
and reports up to ~98% accuracy, but only ever evaluates on random splits *within* a single
dataset. No published result in this space tests a model on genuinely unseen source code.
This project fills that gap using two existing, real, function-level JavaScript vulnerability
datasets — no web scraping or new manual labeling required.

## Datasets

| Dataset | Rows | Vulnerable % | Columns | Source |
|---|---|---|---|---|
| Ferenc et al. (`JSVulnerabilityDataSet-1_0.csv`) | 12,125 | 12.3% | 44 (static/Halstead metrics) | University of Szeged |
| Viszkok et al. (`pm_uom_all_full.csv`) | 8,038 | 11.9% | 78 (44 shared + 34 extra process/ESLint metrics) | Zenodo (DOI 10.5281/zenodo.4590021), CC-BY 4.0 |

**Important discovery made during this project:** every repository in Viszkok's dataset also
appears in Ferenc's dataset. Viszkok's team re-analyzed a 73-repo subset of Ferenc's 93
repos and added extra columns — it is not an independent dataset. Ferenc has 20 repos that
Viszkok never touched; these 20 are the only genuinely unseen codebases available across both
files, and became the held-out test set for the generalization experiment.

## Pipeline

1. **Load and identify repo overlap.** Extract a repo identity from each row's GitHub URL,
   confirm which repos are shared (73) versus Ferenc-only (20, the holdout set).
2. **Deduplicate.** Many rows in both files describe the same function (same repo, file, and
   line range) — 4,654 such duplicates were found. Duplicate label agreement was checked (a
   small number of mismatches, ~10, were found and logged rather than silently resolved).
   Where a function exists in both files, the Viszkok copy is kept (it carries extra columns).
3. **Build two evaluation datasets:**
   - **Dataset A** — the 44 metrics shared by both files, across all 93 repos, with a
     `holdout` flag marking the 20 Ferenc-only repos. This is the main dataset.
   - **Dataset B** — all 78 columns, restricted to the 73 overlapping repos only (since the
     34 extra columns don't exist for the holdout repos). Used only for a secondary
     side-experiment, never for the generalization test.
4. **Drop identity/position columns** (file path, line/column numbers, function name, commit
   hash) from modeling — these describe *where* a function sits in a file, not what the code
   does, and were found to correlate with the label more strongly than any real metric
   (almost certainly a collection artifact, not a real signal).
5. **Correlation analysis** of the remaining features against the vulnerability label.
6. **Repo-grouped cross-validation** (`StratifiedGroupKFold`) for in-distribution evaluation —
   ensures no single project's functions are ever split across train and test within a fold.
7. **Baseline classical models**: XGBoost, Random Forest, Bagging.
8. **Stacked deep learning model**: VGG16/AlexNet/LSTM base learners (1D-adapted) feeding a
   logistic-regression meta-learner, reusing the architecture family from prior work.
9. **Headline test**: train on the full 73-repo pool, evaluate purely on the 20 held-out repos.
10. **Secondary test**: within the 73-repo overlap only, compare 44-metric-only performance
    against the full 78-column version, to see whether Viszkok's extra process/warning
    columns add anything.

## Limitations Encountered and How They Were Addressed

| Limitation | Why it's a problem | Solution implemented |
|---|---|---|
| Class imbalance (~12% vulnerable in both datasets) | A model can score ~88% accuracy by always predicting "not vulnerable" — a useless model that looks good on paper | Class-weighted loss / focal loss as the primary fix; Gaussian noise resampling and SMOTE included as comparisons, not defaults; PR-AUC and MCC reported alongside accuracy/F1 so minority-class failure can't hide |
| Dataset overlap / leakage risk | Nearly all of Viszkok's data comes from repos also in Ferenc's — training on one and testing on the other would test the model on code it may have already seen | Deduplicated to one row per real function; switched the evaluation design to repo-grouped CV plus a genuinely exclusive 20-repo holdout, instead of "dataset A vs dataset B" |
| Position/identity features as spurious predictors | `line`/`column` correlated with the label more than any complexity metric — an artifact of how the data was collected, not a real signal | Dropped from all modeling; kept only as row identifiers |
| One-week time constraint | Not enough time for a full commit-mining + labeling pipeline from scraped GitHub data | Dropped the scraping/labeling plan entirely in favor of the two existing public datasets, restructuring the "unseen data" test around the 20 Ferenc-only repos instead |
| SMOTE's interpolation risk | Several of the 44 metrics are deterministic functions of each other (e.g. Halstead volume is derived from vocabulary and length); interpolating between two real functions can produce metric combinations no real function could have | Used class-weighting/focal loss as the primary imbalance fix; SMOTE and Gaussian noise resampling included only as secondary comparisons, evaluated on the same footing |
| Compute/time limits on deep learning | The original architecture used 1000 training epochs per CNN, impractical to run repeatedly within a week across multiple folds | Reduced to 40 epochs per model per fold — enough to get a usable comparison without making the whole CV sweep infeasible |

## Techniques Implemented

- Repo-level deduplication via a `repo + file path + line range` key
- `StratifiedGroupKFold` cross-validation grouped by repository (not by row)
- Class-weighted loss, focal loss, Gaussian noise oversampling, and SMOTE, compared side by side
- Evaluation via accuracy, precision, recall, F1, PR-AUC, and Matthews Correlation Coefficient (MCC)
- Classical ML baselines (XGBoost, Random Forest, Bagging) alongside a stacked CNN ensemble
- A genuinely held-out, cross-project generalization test (train on 73 repos, test on 20 never-seen repos)
- A secondary ablation on whether process/warning metrics add value on top of static metrics alone

## A Caveat Worth Knowing When Interpreting Results

The 20-repo holdout set has a different vulnerable-class rate (~18%) than the 73-repo training
pool (~10%). Since PR-AUC's baseline (a random guess) is exactly the positive-class rate of
the set being scored, a higher raw PR-AUC on the holdout set doesn't automatically mean better
generalization — it can partly reflect an easier test set. Compare each PR-AUC against its own
set's baseline rate, or rely more on MCC, when judging whether generalization actually held up.

## Future Work

### Extracting real code-level features from the GitHub URLs

Both datasets include a GitHub URL (repo + commit hash) and exact line range for every
function — meaning the actual source code behind every labeled row can be retrieved directly,
without any new scraping or labeling effort. This was identified as a strong next step beyond
the current static-metric-only approach. Proposed features, grouped by what they detect:

- **Dangerous sink calls** — counts of `eval()`, `new Function()`, string-argument
  `setTimeout`/`setInterval`, `innerHTML`/`outerHTML`/`document.write()`,
  `child_process.exec/execSync/spawn`, and non-literal-path `fs.readFile`/`fs.writeFile` calls.
  These are the actual mechanisms behind code execution, XSS, command injection, and path
  traversal — none of which the current structural metrics can see.
- **Tainted input reaching a sink** — a lightweight, same-function check for whether a
  parameter or a common input source (`req.query`, `req.body`, `req.params`) appears
  concatenated into, or passed directly as, an argument to one of the sink calls above.
- **Sanitization presence** — calls to `encodeURIComponent()`, `escape()`,
  `DOMPurify.sanitize()`, `xssFilters.*`, or `validator.*`, treated as a risk-*reducing* signal
  rather than a risk-increasing one.
- **Hardcoded secrets / weak crypto** — regex detection of API-key-shaped strings or long
  base64 blobs assigned to security-sounding variable names, and use of `md5`/`sha1` or
  `Math.random()` in a security-relevant context.
- **Risky imports** — counts of `require()` calls to Node's I/O/process/network modules
  (`fs`, `child_process`, `net`, `http`, `vm`), since functions touching these have inherently
  larger attack surface than pure logic/utility functions.
- **ReDoS indicators** — regex literals in the function body containing nested quantifier
  patterns (e.g. `(a+)+`), a known cause of catastrophic-backtracking denial-of-service.

**How to extract them:** build the raw-file URL from each row's GitHub URL (swap
`github.com` for `raw.githubusercontent.com`, drop the `/blob/` segment), fetch the file,
slice to the row's line range to isolate the exact function, and run keyword/regex checks
over that text — no full AST parsing required for a first pass, though `@babel/parser` or
`esprima` could be added later for more precise detection.

**Why this would help:** the current 44/78 features are all *general code-quality* signals
(size, complexity, duplication) — none of them look at what the code actually does. Adding
these lexical, security-specific features gives the model direct evidence of risky patterns
instead of only an indirect proxy ("complex code tends to be risky"), which should both
improve raw performance and make the model's generalization to unseen codebases more robust,
since these patterns (e.g. `eval()` usage) are meaningful regardless of which project they
appear in.

### Other approaches for further work

- **Source-code embeddings** (e.g. CodeBERT or GraphCodeBERT) fine-tuned on the same labels,
  fused with the static-metric model in a hybrid architecture, to capture semantic patterns
  beyond both structural metrics and simple keyword/regex features.
- **A larger, independently labeled dataset** built via commit-mining (diffing GitHub
  security-fix commits against their parent to label pre-fix functions as vulnerable and
  post-fix functions as fixed), extending well beyond the 93 repos available here, if more
  time is available in a future iteration.
- **AST-based structural features** (node-type histograms, control-flow graph properties)
  as a middle ground between simple static metrics and full semantic embeddings.
- **A live demo tool**: wrap the best-performing trained model together with a static-metric
  extractor (and, longer-term, the lexical security features above) so a real `.js` file can
  be dropped in and scored per-function — turning the trained model into something that could
  realistically sit inside a code review pipeline or CI/CD system.
- **Threshold and deployment tuning**: calibrate the decision threshold to the intended use
  case (e.g. favor recall over precision for a security-triage tool, since a missed
  vulnerability is costlier than a false alarm that a human reviewer quickly dismisses).
