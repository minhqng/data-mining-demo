# BTL11 — Final code audit

**Audit date:** 2026-10-27  
**Auditor role:** independent code reviewer / statistical auditor (read-only on all historical artifacts)  
**Normative contract:** `docs/BTL11_SSOT.md`, `docs/DECISIONS.md`, `docs/METHODOLOGY.md`, `AGENTS.md`  
**Status vocabulary:** `CONFIRMED` (executed and reproduced), `PARTIAL` (partly verified or constrained by
historical schema), `BLOCKED` (human/external gate), `NOT VERIFIED` (not executed in this audit).

---

## 1. Scope and method

### 1.1 What was read

- All non-trivial source modules in `src/btl11/`: `config.py`, `data.py`, `features.py`,
  `preprocess.py`, `models.py`, `evaluation.py`, `experiment.py`, `stability` logic inside
  `experiment.py`, `verification.py`, `profiling.py`, `reporting.py`, `visualization.py`,
  `run_visuals.py`, `robustness.py` (new audit module), `reduction.py`, `eda.py`, `cli.py`,
  `utils.py`, `errors.py`, plus the `__init__`/`__main__` wrappers.
- All 50 tests in `tests/` (`conftest.py`, `test_data.py`, `test_models.py`,
  `test_evaluation.py`, `test_pipeline.py`, `test_profiling.py`, `test_reporting.py`,
  `test_cli.py`, `test_robustness.py`).
- All canonical documents: `BTL11_SSOT.md`, `DECISIONS.md`, `METHODOLOGY.md`, `EXPERIMENTS.md`,
  `REPORT.md`, `DATASET_PROVENANCE.md`, `IMPLEMENTATION_PLAN.md`, `README.md`, `AGENTS.md`,
  `configs/base.yaml`.
- Run artifacts: every `run.json`, `metrics.json`, config snapshot and assignment table for the
  frozen evaluation, holdout, final refit, stability, Ward and all candidate runs; comparison,
  ablation, EDA, PCA, verification, report and post-hoc artifacts; the 12-slide defense deck
  (`deliverables/BTL11_defense.pptx`) read via its OOXML text runs.
- Raw/prepared data identity and audit: `data/raw/MANIFEST.json`,
  `data/processed/audit.json`, `prepare_summary.json`, split files.

### 1.2 What was executed

```text
pytest -q                                      -> 50 passed
ruff check src tests scripts                   -> All checks passed
btl11 verify --run-id <every INDEX run_id>     -> 63/63 PASS, 0 failures
btl11 compare --experiment-set baseline_f9_standard
btl11 report --selected-run <final-refit> ...
btl11 robustness --evaluation-run <frozen> ... -> full LOO + scaler + GMM table
scripts/run_controlled_gmm.py                  -> 8 new GMM fits, all verified
btl11 ward --config configs/base.yaml          -> corrected Ward run, verified
pip-audit -r requirements.lock                 -> no known vulnerabilities
```

Independent recomputations used raw prepared tracks + stored `model.joblib` bundles and sklearn
directly, bypassing the pipeline's metric wrappers. Stored vs recomputed:

| Run | Silhouette | Davies–Bouldin | Calinski–Harabasz | Result |
|---|---:|---:|---:|---|
| Frozen eval (n=3,000) | 0.595621423666 | 0.650471644093 | 3240.61795948 | exact match |
| Holdout (n=3,000) | 0.590221775420 | 0.659256321261 | 2964.07344109 | exact match |
| Final refit verification sample (n=3,000) | 0.584192529339 | 0.673044694957 | 2912.16893586 | exact match |

Additional check: the frozen evaluation model and the final full-data refit produce identical
labels on the final refit's verification IDs (ARI = 1.0), while remaining distinct model
identities (`n_fit` 68,813 vs 89,741; different trained-ID hashes and scaler statistics).

### 1.3 What was **not** verified

- At audit time, a fresh clean-clone install after the audit changes had not been re-run (closed
  later on 2026-10-27; see the closure note at the end).
- Remote GitHub Actions CI was not run (no git remote/push was authorized).
- The defense deck was edited and structurally checked (12 slides, ZIP integrity, text present);
  a 12-page LibreOffice PDF render was produced later on 2026-10-27, while the visual/timing pass
  remains human.
- License/redistribution review remains a human gate; no raw data was published.

---

## 2. Findings

### A-01 — `instrumentalness` dominates the selected RobustScaler K=2 partition

**Severity:** HIGH for scientific interpretation; **not** a code defect.  
**Status:** `CONFIRMED` — analysed, documented, model preserved.  
**Files/artifacts:** `artifacts/robustness/20261027T070249Z/`; frozen run
`20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`;
`src/btl11/robustness.py` (new).

Evidence (development IDs only; fit n=68,813, eval n=3,000):

- `instrumentalness` accounts for **97.29%** of total between-cluster sum of squares and
  **97.29%** of the mean K-Means assignment margin.
- Removing `instrumentalness` from the assignment margin would flip **30.05%** of fit
  assignments; no other feature exceeds 0.13%.
- A K-Means K=2 fit on `instrumentalness` alone reproduces the full eval partition at
  **ARI 0.9892**, with proportions 19.6% / 80.4%. Concretely, **7 of the 3,000 eval tracks**
  differ (contingency [[585, 4], [3, 2408]]; cluster sizes 589/2,411 vs 588/2,412). On the
  fit cohort the same comparison gives ARI 0.9905 with 141/68,813 differing, so 0.9892 is the
  eval-cohort statistic.
- The two 97.29% shares are the same K=2 quantity δ_j²/Σ_jδ_j² computed two ways, not
  independent evidence; they agree to 6.7e-6 (numerical centroid/final-label drift).
- Leave-one-feature-out: dropping `instrumentalness` changes the partition
  (ARI **−0.0498**, proportions 6.5% / 93.5%); dropping any other feature leaves
  **ARI ≥ 0.9938**. Every LOO variant is itself seed-stable (seed ARI mean 1.0).
- The raw feature is strongly zero-inflated/bimodal: zero share **33.34%**, near-zero
  (<1e-4) share **52.12%**, IQR **0.0985**; after RobustScaler its transformed variance is
  **10.82** versus 5.03 for `speechiness` and <1.0 for most others.
- Under StandardScaler the same K=2 model is genuinely multivariate: top between-cluster
  shares are `energy` 29.46%, `acousticness` 26.48%, `loudness` 23.00%;
  `instrumentalness` is only 3.95%, and no single standard-scaled feature reproduces the
  partition (best single-feature ARI: `loudness` 0.4942 vs `instrumentalness` 0.1300).
  The two scaler partitions agree at only **ARI 0.1350**.

**Root cause:** RobustScaler divides by the IQR; `instrumentalness` has a tiny IQR relative to
its between-track spread, so its transformed variance dominates Euclidean distance. This is a
representation-dependent property, not an implementation error. **Resolution:** preserve the
frozen model and holdout evidence; add the quantitative analysis; narrow the report's
interpretation from "balanced multivariate acoustic structure" to "instrumental vs
vocal/non-instrumental segmentation". The model remains defensible for that specific objective.

### A-02 — Ward diagnostic refit its scaler on the 2,000-track sample

**Severity:** MEDIUM (non-controlled comparison overstated agreement).  
**Status:** `CONFIRMED`, `FIXED` for new runs; historical run retained.  
**Files:** `src/btl11/experiment.py` `ward_diagnostic()` (current lines 2387–2540).

The historical Ward run
`20261027T032907Z_f7434a8c_b202fa49_ward_diagnostic` refit `RobustScaler` inside
`ward_diagnostic()` on the bounded sample before Ward. The stored ARI vs the frozen K-Means
was **0.969205**, but the scaler statistics differed from the frozen model's. Re-running Ward
with the frozen evaluation model's fitted transformer on the same 2,000 IDs gives
**ARI 0.883442** (new run
`20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`; silhouette 0.5945 on the
sample representation, DB 0.6142, CH 1938.4). The corrected number is lower but still supports
a strong structural agreement. The old run remains immutable and is clearly marked as
superseded for the controlled claim.

**Fix:** `ward_diagnostic()` now loads the selected candidate's bundle, verifies its checksum,
feature order and algorithm/K, and transforms the sample with that frozen transformer. The
metrics context records `representation = frozen selected-model transformer` and the selected
model SHA.

### A-03 — Verification was range-checking stability ARI and did not fully verify the freeze chain

**Severity:** MEDIUM (assurance gap; no wrong number found).  
**Status:** `CONFIRMED`, `FIXED`.  
**Files:** `src/btl11/verification.py`.

Before the audit, stability verification recomputed ARI values but only checked they were in
`[-1, 1]`; it did not compare them to the stored `seed_ari_mean`, `bootstrap_ari_mean` or
`pairwise_ari_mean`. Holdout/final-refit verification also used the reference run only to fill
`n_init`/`reg_covar`, without checking the freeze → holdout → refit chain, the descriptor hash,
the full-track fit scope, or Ward sample membership.

Added checks include: descriptor presence/hash/equality; holdout `evaluation_model_id` and
one-time flag; holdout ID/eval-ID hashes; final-refit `n_fit`, all-track fit hash and
verification cohort; holdout → evaluation-model identity; evaluation-model dataset/split
identity; stability mean-ARI recomputation from `stability_labels_*.csv`; Ward sample subset
and selected-run reference. The legacy schema fallback (A-12) is explicit and logged.

Result: **63/63 completed runs verify**; key check counts are frozen 61, holdout 70, final
refit 84, stability 68, old Ward 31, corrected Ward 31, baseline GMM full K4 65, PCA90 K6 63,
and all eight exploratory GMM runs 65.

### A-04 — "Best balance" wording for the selected model is false

**Severity:** MEDIUM (scientific wording).  
**Status:** `CONFIRMED`, corrected in `docs/DECISIONS.md` and `docs/REPORT.md`.  
**Files:** `docs/DECISIONS.md` D-05; `docs/REPORT.md` §8.1/§10.

The selected RobustScaler K=2 model has fit-cohort minimum share **0.19546**, versus
**0.24891** for StandardScaler K=2 and **0.30267** for F5 K=2. It is not the most balanced
candidate. Its defensible advantages are in-space silhouette/DB/CH, near-perfect seed/bootstrap
stability, simple K=2 structure, and a clear instrumental/vocal reading. The report's
"best in-space validity, balance" phrase must read "strong in-space validity and adequate
(not maximal) balance", with the dominance caveat.

### A-05 — K-Means vs GMM comparison lacked partition agreement and was framed too narrowly

**Severity:** MEDIUM (interpretation).  
**Status:** `CONFIRMED`, new controlled evidence added.  
**Files:** `docs/REPORT.md` §8.2; `docs/DECISIONS.md` D-06; `src/btl11/robustness.py`.

Existing comparisons used the same fit/eval IDs, F9 representation, seed and K, so the core
protocol was controlled. However, the report reported hard-label silhouette and confidence
without ARI between the algorithms. The audit adds 8 exploratory GMM fits at
`n_init=5` (Standard/Robust × K=2/4 × diag/full) and computes ARI on identical eval IDs.
K-Means and GMM hard partitions largely disagree: ARI ranges **−0.0201 to 0.3063** across the
eight controlled rows. GMM posteriors are confident (mean confidence 0.947–0.997) *and* point
to a different structure; this is not a contradiction. It shows that density fit and hard
segmentation answer different questions. Among the controlled K=2/4 GMM fits, full K=4 has the
lowest BIC in each scaler space (Standard 853,059; Robust 819,184); the original Standard K=2..12
sweep still reaches a lower BIC at K=12 full (678,842), a density-boundary result, not a
segmentation choice. Raw BIC is never compared across scaler spaces.

### A-06 — `min_cluster_share`/`max_cluster_share` ignored empty expected clusters

**Severity:** LOW–MEDIUM (metric guard).  
**Status:** `CONFIRMED`, `FIXED`.  
**File:** `src/btl11/evaluation.py` `score_clustering()` (line 100).

`min_cluster_share` and `max_cluster_share` were computed from observed label counts only. If
`expected_k` contained an empty cluster, the empty cluster disappeared from the minimum
instead of counting as 0.0. The code now uses the expected-k size table (`sizes["share"]`),
which includes zero-count clusters. Regression test:
`tests/test_evaluation.py::test_empty_expected_cluster_counts_as_zero_min_share`.

### A-07 — GMM covariance default was inconsistent between the estimator and metrics

**Severity:** LOW (latent API defect).  
**Status:** `CONFIRMED`, `FIXED`.  
**Files:** `src/btl11/models.py` `fit_clustering_bundle()`; `src/btl11/experiment.py`
`run_candidate()`.

`create_clusterer("gmm", ..., covariance_type=None)` silently defaulted to `diag`, but
`run_candidate()` passed `str(None)` into `parameter_count()`, which would raise if a caller
used the API without an explicit covariance type. `fit_clustering_bundle()` now resolves the
covariance before fitting and stores the resolved value in the bundle; `run_candidate()` uses
`bundle.covariance_type` for information criteria. Regression test:
`tests/test_models.py::test_gmm_default_covariance_is_recorded_and_criteria_are_finite`.
No historical GMM run was affected (all passed an explicit covariance).

### A-08 — Candidate profiles used all tracks (including holdout) as the global reference

**Severity:** LOW–MEDIUM (holdout-discipline nuance).  
**Status:** `CONFIRMED`, `FIXED` for future profiles.  
**File:** `src/btl11/profiling.py` `generate_profiles()` (lines 317–330).

For non-final runs, `generate_profiles()` used `prepared.tracks` (89,741, including holdout)
as the reference for raw-unit z-deviations. That is not a model fit or a label leak, but it
does let holdout feature distributions influence a descriptive scale used during candidate
inspection. It now uses the development split for candidate/holdout scopes and all tracks only
for the final refit. Historical final-refit profiles are unaffected because they used the
all-tracks scope by design; no historical candidate profiles were part of the evidence chain.

### A-09 — Regenerated baseline comparison with cohort-aware BIC ranking

**Severity:** LOW (stale derived artifact).  
**Status:** `CONFIRMED`, `FIXED` as a new derived artifact.  
**Artifacts:** old `artifacts/comparisons/baseline_f9_standard/20261027T032015Z/`; new
`artifacts/comparisons/baseline_f9_standard/20261027T070434Z/`.

The old comparison predated the D-14 cohort-scoped BIC ranking and lacked a `cohort_id`
column, although all 33 rows share one cohort and the BIC values are correct. The regenerated
table contains `cohort_id` and identical BIC values/ranks (max absolute difference 0.0). The
old directory is retained; the report cites the regenerated table for the cohort-aware claim.

### A-10 — Configuration validation gaps

**Severity:** LOW (fail-fast honesty).  
**Status:** `CONFIRMED`, `FIXED` (one enforcement is code-reviewed but not exercised by a
non-converged real fit).  
**File:** `src/btl11/config.py` lines 215–299; `src/btl11/experiment.py` lines 652–657.

- `data.missing_numeric` accepted `impute_dev_only` but the pipeline never implements
  imputation; it silently behaves like `drop_after_audit`. Now non-`drop_after_audit` values
  fail with a clear "not implemented" error.
- `search.silhouette_sample_size` / `silhouette_seed` were validated but unused; changing them
  would silently not affect the scored cohort. They must now equal
  `data.eval_cohort_size` / `data.eval_seed`.
- `gmm.require_convergence` was validated but not enforced. A non-converged GMM candidate now
  fails the run when the flag is true. All historical GMM fits converged, so no historical
  result changes.
- Regression assertions were added to
  `tests/test_data.py::test_config_locks_dataset_and_feature_order`.

### A-11 — `gmm.reg_covar_fallback` is validated but unused

**Severity:** LOW (unexercised configuration hook).  
**Status:** `NOT EXERCISED` / documented open limitation.  
**File:** `src/btl11/config.py` lines 305–309; `src/btl11/models.py` `create_clusterer()`.

The fallback list `[1e-5, 1e-4]` is not consumed anywhere. All 22 baseline GMM fits and all
8 exploratory GMM fits converged at `reg_covar=1e-6`, so no result is affected. A future
numerical-failure path should either implement the fallback with explicit run metadata or
remove the hook; it is not needed for the current evidence.

### A-12 — Historical holdout/final `run.json` lack `frozen_descriptor` fields

**Severity:** LOW (schema drift; historical artifacts are immutable).  
**Status:** `PARTIAL` — verifier handles legacy runs; new runs write the fields.  
**Files:** `src/btl11/experiment.py` (new fields); `src/btl11/verification.py`
`_check_run_chain()` legacy fallback.

The historical holdout/final run documents contain `descriptor` but not
`frozen_descriptor`/`frozen_descriptor_sha256`. The verifier now uses the present
`descriptor` payload when `frozen_descriptor` is absent, records a legacy note, and still
checks descriptor equality against the referenced candidate. New holdout/refit runs write both
fields and their hash. The historical artifacts were not edited.

### A-13 — Minor code-hygiene observations

**Severity:** LOW/INFO. **Status:** documented, no demonstrated impact.

- `stability()` has a dead `np.isnan(base_eval.astype(float))` check that would raise earlier
  on missing IDs; it is redundant with the `reindex` failure.
- `_feature_space_id()` infers the feature set from list length; the primary allowlist
  validation prevents any non-F9/F5 list from reaching it, so this is acceptable but brittle.
- K-Means `converged` is inferred as `n_iter < max_iter`; this is the standard sklearn
  heuristic, not a tolerance-based proof.
- `run_sweep()` uses a timestamp directory with `exist_ok=False`; two sweeps in the same
  second would collide. Not observed in the 2026-10-27 runs.

### A-14 — Report provenance guard added

**Severity:** LOW. **Status:** `FIXED`.  
**File:** `src/btl11/reporting.py` `run_report()` lines 147–174.

`run_report()` already rejected a broken freeze/holdout/refit chain. It now also rejects a
selected final refit, holdout or evaluation model produced from different dataset bytes than
the current prepared snapshot. This prevents silently mixing a report with regenerated data.
Existing report generation still passes.

### A-15 — Security and packaging scan

**Severity:** INFO. **Status:** `CONFIRMED`.

- No credentials, `.env`, private keys or token-like strings in source/config/tests/docs.
  Grep hits for "secret" were track titles inside the non-redistributable raw CSV.
- No absolute local paths in `src/`, `tests/`, `docs/`, `README.md`, `AGENTS.md` or `configs/`.
- Raw CSV is 20 MiB, gitignored (`data/raw/*.csv`), and not published. `artifacts/` is about
  185 MiB (mostly generated run folders/figures; `.joblib` and run directories are gitignored).
- `pip-audit -r requirements.lock`: no known vulnerabilities.
- `joblib` model loading is documented as trusted-local-only; run IDs are path-validated.
- License/redistribution remains `pending_human_verification`; no remote publication occurred.

### A-16 — Test, lint and verification results

**Severity:** INFO. **Status:** `CONFIRMED`.

- `pytest -q`: **50 passed** (synthetic fixtures only; includes the new robustness,
  zero-IQR, empty-cluster, feature-order, GMM-default, tampered-hash and broken-chain tests).
- `ruff check src tests scripts`: **All checks passed**.
- `btl11 verify` on every run in `artifacts/INDEX.csv`: **63/63 PASS, 0 failures**.
- Independent metric recomputation: exact matches for frozen/holdout/final (table in §1.2).
- `btl11 --help`, `btl11 robustness`, `scripts/run_controlled_gmm.py`, `btl11 ward`,
  `btl11 report` and `btl11 compare` all executed successfully on the real snapshot.

### A-17 — Human/external gates

**Severity:** BLOCKED/NOT VERIFIED. **Status:** unchanged from D-01/D-12/D-15.

- License/redistribution: pending human verification (Kaggle ODbL vs HF BSD vs Spotify
  provenance). Raw CSV must remain private.
- No git commit/push/deploy was authorized or performed; remote CI did not run.
- Defense deck: caveat added, 12 slides, ZIP/text integrity checked; a 12-page PDF render exists;
  human rehearsal remains.
- Clean-clone install: closed on 2026-10-27 (see the closure note at the end).

---

## 3. Are the frozen numbers still valid?

**Yes, for what they measure.** The frozen evaluation, holdout and final-refit numbers are
reproduced exactly from raw data + stored bundles. The selected K-Means/RobustScaler/K=2 model
was not refit, retuned, or re-evaluated on holdout. The holdout remains a one-time check.

**What changes is the interpretation.** The K=2 RobustScaler model is overwhelmingly a
one-feature separation driven by `instrumentalness` (97.29% of between-cluster separation;
single-feature ARI 0.9892; removing it changes the partition). It should be presented as an
**instrumental vs vocal/non-instrumental segmentation**, not as evidence of balanced
multivariate acoustic personas. The StandardScaler K=2 partition is more balanced but
different (cross-scaler ARI 0.1350), and its silhouette is lower in its own space; neither
representation is universally superior.

## 4. Residual risks

1. **Construct validity:** Spotify audio features are algorithmic descriptors, not audio;
   `instrumentalness` is itself a model estimate and may be noisy for some genres.
2. **Representation dependence:** scaler choice changes the partition substantially; report
   all scaler-specific metrics with the space ID.
3. **K=2 coarseness:** the final model cannot separate speech-heavy, live or dance subtypes;
   K=6 is richer but less stable and has a ~1.1% cluster.
4. **GMM interpretation:** density components are not semantic clusters; ARI shows the
   controlled GMM hard partitions differ from K-Means.
5. **Ward:** sample-only, no prediction for new tracks; the corrected controlled ARI is 0.883,
   not a generalization metric.
6. **License/publication:** unresolved human gate; no raw data publication.
7. **Deck:** one text caveat added; a PDF render exists, but timing/visual review still needs a
   human pass.

## 5. Post-audit verification closure — 2026-10-27

- Clean-clone install: **CONFIRMED**. Fresh venv from a clean copy (excluding `data/`, `artifacts/`,
  `.venv`); `pip install -e '.[dev]'` succeeded; `pytest -q` 50 passed; `ruff` clean; `btl11 --help` OK.
- Deck render: `deliverables/BTL11_defense.pdf`, 12 pages, LibreOffice 26.8, SHA-256
  `9231daa77b105805e5b7d6d5080a374c442cdcc37926388636c6716cd5e55dcb`; its text matches the PPTX,
  including the feature-dominance caveat. Human visual/timing rehearsal remains.
- Re-verification on the current workspace: `pytest -q` 50 passed, `ruff` clean, `btl11 verify`
  63/63 PASS (0 failures).
- Still open: license/redistribution (human), git remote/CI (not authorized), rehearsal (human).

## 6. What was intentionally not changed

- The frozen model descriptor, evaluation run, holdout run and final refit run.
- `configs/base.yaml` selection/frozen/selected state.
- Any historical `run.json`, `metrics.json`, model, assignment, figure or report table.
- The primary F9/F5 allowlists, split protocol, feature order, cleaning rules or metric
  definitions.
- No alternative dataset, Spotify API, external metadata, synthetic research evidence, P2
  extension or retuning on holdout.
