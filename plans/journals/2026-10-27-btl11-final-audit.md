---
title: BTL11 final scientific audit and robustness hardening
date: 2026-10-27
summary: "Independent audit: feature dominance, corrected Ward, controlled K-Means/GMM, hardened verification, 63 runs, 50 tests"
---

# BTL11 final scientific audit and robustness hardening

## Status

Internally complete and reproducible; public redistribution still **BLOCKED** by license review.
Frozen model, holdout and final refit were **not** retuned or overwritten.

## What was inspected

All modules in `src/btl11/` (including the new `robustness.py`), all 50 tests, all canonical
documents, the raw/prepared data identity, every run artifact in `artifacts/runs/`, comparison,
ablation, EDA, PCA, verification, report and post-hoc artifacts, and the 12-slide defense deck
(OOXML text run inspection). No alternative dataset, Spotify API or synthetic research evidence
was used.

## What was executed

- Independent metric recomputation from raw prepared tracks + stored bundles (frozen, holdout,
  final refit): exact matches.
- `pytest -q` → 50 passed; `ruff check src tests scripts` → clean.
- `btl11 verify` for every ledger run → **63/63 PASS, 0 failures**.
- `btl11 compare` regenerated the baseline table with `cohort_id` (values identical).
- `scripts/run_controlled_gmm.py` → 8 exploratory GMM runs (F9/noPCA, Standard/Robust × K=2/4 ×
  diag/full, n_init=5), all verified.
- `btl11 robustness` → feature dominance, LOO, scaler comparison and controlled GMM table
  (`artifacts/robustness/20261027T070249Z/`).
- `btl11 ward` → corrected frozen-transformer Ward run
  `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`, ARI 0.883442.
- `btl11 report` regenerated `artifacts/final-report-tables/` (63 runs).
- `pip-audit -r requirements.lock` → no known vulnerabilities; secret/absolute-path scan clean.
- Defense deck: one feature-dominance caveat added; 12 slides preserved; SHA-256
  `e880f7eac1222b60c48294b7a05bc48098a1cafaa949a3046531a3ed9b185ba7`.

## New empirical findings

- Under the frozen RobustScaler K=2 model, `instrumentalness` accounts for **97.29%** of
  between-cluster SS and **97.29%** of the mean assignment margin; removing it would flip
  **30.05%** of fit assignments.
- A K-Means K=2 fit on `instrumentalness` alone reproduces the full eval partition at
  **ARI 0.9892** (19.6%/80.4%; one track differs in 3,000 — **corrected below: 7 differ**).
- Leave-one-feature-out: dropping `instrumentalness` changes the partition (ARI **−0.0498**,
  6.5%/93.5%, a speechiness-dominated split); dropping any other feature leaves ARI ≥ 0.9938.
- StandardScaler K=2 is genuinely multivariate (energy 29.46%, acousticness 26.48%, loudness
  23.00% of between SS; instrumentalness 3.95%) and agrees with the robust partition at only
  **ARI 0.1350**.
- Controlled K-Means vs GMM hard-label ARI is **−0.0201…0.3063** while GMM confidence is
  0.947–0.997: GMM is a different confident structure, not merely a weaker K-Means.
- Ward controlled ARI is **0.883442** (frozen transformer), not the historical 0.969205
  (sample-refit scaler).

## Confirmed fixes

- Empty expected clusters now count as 0 in min/max share.
- GMM covariance default is explicit; `gmm.require_convergence=true` is enforced.
- Candidate/holdout profile references use the development split, not holdout.
- Ward uses the frozen selected transformer.
- Verification now checks the freeze → holdout → refit chain, descriptor hashes, stability mean
  ARI, Ward sample membership and a legacy descriptor fallback.
- Config rejects unimplemented imputation and divergent silhouette-cohort settings.
- Report rejects a refit/holdout/evaluation model from different dataset bytes.
- Baseline comparison regenerated with cohort-aware BIC ranking.

## Deliverables

- `docs/FINAL_CODE_AUDIT.md` — source-level findings, severities and resolutions.
- `docs/ROBUSTNESS_ANALYSIS.md` — dominance, LOO, scaler and controlled GMM evidence.
- `docs/FINAL_READINESS.md` — acceptance checklist, gates and verdict.
- Updated `docs/REPORT.md`, `docs/DECISIONS.md` (D-16–D-19), `docs/METHODOLOGY.md`,
  `docs/EXPERIMENTS.md`, `docs/IMPLEMENTATION_PLAN.md`, `README.md`, `AGENTS.md`, `docs/BTL11_SSOT.md`
  changelog.
- New exploratory artifacts under `artifacts/robustness/20261027T070249Z/`,
  `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/`, and the corrected Ward run.

## Remaining gates

- License/redistribution: pending human verification; raw CSV not published.
- No git commit/push/deploy; remote CI not run.
- Defense deck visual render and timing rehearsal; no renderer in this runtime.
- Clean-clone install after the audit changes not re-run.
- `gmm.reg_covar_fallback` unused; all current GMM fits converged.
- Historical holdout/final `run.json` lack `frozen_descriptor`; verifier handles the legacy
  `descriptor` fallback.

## Verdict

K-Means / F9 / RobustScaler / no PCA / K=2 remains defensible **as an instrumental vs
vocal/non-instrumental segmentation**, not as balanced multivariate persona discovery. The frozen
numbers are valid; the report's interpretation was narrowed. No high-severity implementation defect
invalidates the stored results. The holdout was not reopened.

---

## Correction — 2026-10-27 submission-readiness verification

The earlier line "one track differs in 3,000" was wrong: it reported the net cluster-size change
(2,411 → 2,412), not the number of reassigned tracks. Recomputed from the frozen bundle and the
fit/eval splits, the eval contingency is [[585, 4], [3, 2408]] with **7 differing tracks** at ARI
0.9892166883; cluster sizes are 589/2,411 (frozen) vs 588/2,412 (1-D). On the 68,813 fit IDs the
same comparison gives ARI 0.9905 with 141 differing, so 0.9892 is the eval-cohort value. The two
97.29% shares are also the same K=2 δ²/Σδ² quantity computed two ways (0.972878 vs 0.972872), not
independent statistics. Normative docs corrected: `docs/ROBUSTNESS_ANALYSIS.md`, `docs/REPORT.md`,
`docs/DECISIONS.md`, `docs/FINAL_CODE_AUDIT.md`, `docs/FINAL_READINESS.md`, `docs/EXPERIMENTS.md`,
`docs/METHODOLOGY.md`. No model, metric, run or artifact was changed.
