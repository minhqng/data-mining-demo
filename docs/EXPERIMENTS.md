# Experiment ledger — actual commands, runs and evidence

This is the audit trail for the real data run on 2026-10-27. Every number comes from
`artifacts/runs/<run_id>/metrics.json`; the canonical list is `artifacts/INDEX.csv`.
No synthetic fixture result is presented as research evidence.

## 1. Environment and dataset

| Item | Actual |
|---|---|
| Host | Intel Core i7-12700H, 20 logical CPUs, 38.85 GiB RAM |
| OS / Python | Linux, Python 3.13.16 |
| NumPy / pandas / SciPy | 2.5.4 / 3.1.0 / 1.18.1 |
| scikit-learn / matplotlib / seaborn | 1.9.1 / 3.11.2 / 0.13.2 |
| joblib / pyarrow | 1.6.0 / 26.0.0 |
| Raw SHA-256 | `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd` |
| Raw rows / unique IDs / genres | 114,000 / 89,741 / 114 |
| Conflicts / quarantined rows | 0 / 0 |
| Dev / fit / eval / holdout | 71,813 / 68,813 / 3,000 / 17,928 |

No fitting was run on holdout. No full 89,741-pairwise dense silhouette was materialized.

## 2. Command ledger

```bash
btl11 audit --input data/raw/dataset.csv --config configs/base.yaml
btl11 prepare --config configs/base.yaml
btl11 eda --config configs/base.yaml
btl11 pca --config configs/base.yaml

btl11 sweep --algorithm kmeans --config configs/base.yaml
btl11 sweep --algorithm gmm --config configs/base.yaml
btl11 compare --experiment-set baseline_f9_standard --config configs/base.yaml

btl11 ablate --factor scaler --config configs/base.yaml
btl11 ablate --factor feature_set --config configs/base.yaml
btl11 ablate --factor pca --config configs/base.yaml

btl11 stability --model-selector finalists --config configs/base.yaml
btl11 validate-holdout --model-selector frozen_finalists --config configs/base.yaml
btl11 refit --model-selector selected_final --all-tracks --config configs/base.yaml

btl11 visualize --run-id 20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2 \
  --kind pca,tsne,profile,genre --config configs/base.yaml
btl11 ward --config configs/base.yaml
btl11 report --selected-run 20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2 \
  --output artifacts/final-report-tables --config configs/base.yaml

# Post-hoc robustness audit (2026-10-27; development-only, holdout untouched)
.venv/bin/python scripts/run_controlled_gmm.py --config configs/base.yaml
btl11 robustness \
  --evaluation-run 20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42 \
  --config configs/base.yaml
btl11 compare --experiment-set baseline_f9_standard --config configs/base.yaml
btl11 ward --config configs/base.yaml   # corrected frozen-transformer diagnostic

btl11 verify --run-id 20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42 --config configs/base.yaml
btl11 verify --run-id 20261027T032820Z_f7434a8c_b202fa49_holdout_20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_rob --config configs/base.yaml
btl11 verify --run-id 20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2 --config configs/base.yaml
btl11 verify --run-id 20261027T032241Z_ea0570dc_b202fa49_stability --config configs/base.yaml
btl11 verify --run-id 20261027T032907Z_f7434a8c_b202fa49_ward_diagnostic --config configs/base.yaml
btl11 verify --run-id 20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr --config configs/base.yaml
# plus every other completed run_id in artifacts/INDEX.csv (63/63 PASS)
```

`btl11 sweep ... --dry-run` was also executed for both algorithms; it created no run artifacts.

## 3. Run inventory

`artifacts/INDEX.csv` contains **63 completed run folders, 0 failed experiment fits**:

| Run type | Count | Notes |
|---|---:|---|
| candidate | 56 | 48 original baseline/ablation fits + 8 exploratory controlled GMM fits; 9 ablation rows reuse baseline runs |
| final_refit | 3 | presentation/label corrections; latest is authoritative |
| stability | 1 | 60 refits, 0 failures |
| holdout | 1 | one-time frozen model check |
| ward | 2 | historical sample-refit scaler + corrected frozen-transformer diagnostic |

Candidate fit time from `run.json`: original K-Means 2.28–4.21 s; original GMM 1.27–19.38 s
(quick sweep n_init=2); exploratory controlled GMM 2.52–19.29 s (n_init=5). The final K-Means
refit over 89,741 tracks took 2.35 s for the fit stage; profile/plot/report time is not included.
Total wall-clock was not instrumented for the full pipeline.

## 4. Baseline evidence

### 4.1 K-Means F9/Standard/no-PCA

All 11 runs completed and converged. Full table:
`artifacts/comparisons/baseline_f9_standard/20261027T070434Z/comparison.csv` (cohort-aware
regeneration; values identical to the historical `20261027T032015Z` table, max absolute BIC
difference 0.0).

| K | Silhouette ↑ | DB ↓ | CH ↑ | Inertia (fit) | Min share (fit) | Run ID (prefix) |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.257588 | 1.584747 | 800.618 | 484,912.428 | 0.248906 | `20261027T031534Z_2e91fc78_..._k2_s42` |
| 3 | 0.165348 | 1.877351 | 651.662 | 430,566.706 | 0.217343 | `20261027T031536Z_2e91fc78_..._k3_s42` |
| 4 | 0.174162 | 1.687733 | 588.533 | 387,558.891 | 0.077573 | `20261027T031539Z_2e91fc78_..._k4_s42` |
| 5 | 0.177621 | 1.505181 | 563.677 | 349,801.843 | 0.011931 | `20261027T031543Z_2e91fc78_..._k5_s42` |
| 6 | 0.189578 | 1.434144 | 556.749 | 318,000.357 | 0.011364 | `20261027T031546Z_2e91fc78_..._k6_s42` |
| 7 | 0.195150 | 1.377898 | 561.267 | 289,373.459 | 0.011306 | `20261027T031549Z_2e91fc78_..._k7_s42` |
| 8 | 0.174325 | 1.395066 | 529.775 | 273,578.532 | 0.011219 | `20261027T031552Z_2e91fc78_..._k8_s42` |
| 9 | 0.180196 | 1.370084 | 517.052 | 259,282.327 | 0.010143 | `20261027T031557Z_2e91fc78_..._k9_s42` |
| 10 | 0.169580 | 1.424942 | 495.141 | 248,708.117 | 0.010129 | `20261027T031601Z_2e91fc78_..._k10_s42` |
| 11 | 0.173354 | 1.432087 | 479.862 | 239,395.609 | 0.010129 | `20261027T031606Z_2e91fc78_..._k11_s42` |
| 12 | 0.173984 | 1.428320 | 459.789 | 231,866.956 | 0.010129 | `20261027T031610Z_2e91fc78_..._k12_s42` |

K=2 leads silhouette/CH, but was not selected on a single metric. K=5–12 contain a ~1% cluster;
K=4 has a 7.76% minimum. Profiles at K=2/4/6 were inspected before freezing (see §5).

### 4.2 GMM F9/Standard/no-PCA

All 22 runs completed and converged. `bic_rank_within_space` is computed only inside the same
representation. Best BIC is full K=12 (678,842; AIC 672,820; silhouette −0.023407). Full K=11 BIC
699,425, silhouette −0.035952; diag K=12 BIC 767,035, silhouette 0.007661. The BIC boundary was
not used to select K (see `docs/DECISIONS.md` D-04). Full table in the baseline comparison CSV.

Fixed-cohort GMM silhouette is much lower than K-Means at low K (e.g. full K=4: 0.054692;
diag K=4: 0.053909). GMM full K=4 posterior confidence mean 0.9578 (eval); 0.33% of eval tracks
(10/3,000) and 0.30% of dev assignments (214/71,813) are below 0.5; it is
kept as a soft-membership complement.

## 5. Profiles inspected before freeze

Real candidate profiles (raw units) generated with `btl11 profile`:

- K=2 standard: acoustic/low-energy (24.86%) vs energetic (75.14%);
- K=4 standard: vocal/dance, high-energy, acoustic, live/speech; minimum 7.76%;
- K=6 standard: adds instrumental/ambient (7.84%) and speech-heavy (1.13% dev; 1.14% fit); seed ARI 0.9334;
- K=2 robust: instrumental/ambient (19.55%) vs vocal/non-instrumental (80.45%);
- K=6 PCA90: similar to K=6 standard with 7 components; seed ARI 0.9987;
- GMM full K=4: instrumental, dance/speech, energy, mixed; hard silhouette 0.0547.

These profiles informed the shortlist and finalists; they were not used to read holdout labels.

## 6. Ablations (K=2,4,6; K-Means)

Full tables:
`artifacts/ablations/scaler/20261027T032128Z/comparison.csv`,
`artifacts/ablations/feature_set/20261027T032140Z/comparison.csv`,
`artifacts/ablations/pca/20261027T032204Z/comparison.csv`.

| Factor | Actual finding |
|---|---|
| Scaler | Robust K=2: silhouette 0.5956, DB 0.6505, min share 0.1955. Standard K=2: 0.2576/1.5847/0.2489. Unscaled E00: 0.5167/0.6502 in a different geometry — diagnostic only. |
| F5 vs F9 | F5 K=2: 0.2980/1.3700/0.3027; F9 K=2: 0.2576/1.5847/0.2489. F5 is more compact but has fewer acoustic dimensions; F9 primary retained. |
| PCA | PCA90 = 7 components, cumulative variance **0.94823**; PCA95 = 8 components, **0.98444**. PCA90 K=6: 0.2033/1.3613/614.5 vs no-PCA 0.1896/1.4341/556.7. Gains are small; no-PCA retained for simplicity/loadings. |

All 24 ablation rows completed; 15 were new fits, 9 reused baseline runs. Cross-representation
metrics are contextual and are not pooled.

## 7. Stability (five seeds + five bootstrap refits)

Run: `20261027T032241Z_ea0570dc_b202fa49_stability`
(tables: `stability_summary.csv`, `ari_pairs.csv`, `stability_labels_*.csv`).

| Finalist | Seed ARI mean | Bootstrap ARI mean | Pairwise ARI mean |
|---|---:|---:|---:|
| KMeans std K=2 | 1.000000 | 0.991433 | 0.992008 |
| KMeans std K=4 | 0.996929 | 0.980867 | 0.982174 |
| KMeans std K=6 | 0.933423 | 0.916413 | 0.869597 |
| KMeans robust K=2 | 0.999692 | 0.999075 | 0.999177 |
| KMeans PCA90 K=6 | 0.998739 | 0.982706 | 0.985122 |
| GMM full K=4 | 0.978550 | 0.936991 | 0.925512 |

Every refit predicts the same 3,000 reference IDs. 60/60 refits completed; 0 failures.
The K=6-standard instability is real and is not hidden.

## 8. Freeze, holdout and final refit

- Frozen before holdout:
  `20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`
  (KMeans, F9, robust, no PCA, K=2, seed 42).
- One-time holdout:
  `20261027T032820Z_f7434a8c_b202fa49_holdout_20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_rob`
  - Silhouette 0.590222; DB 0.659256; CH 2964.073; n_eval 3,000.
  - Deltas vs dev: silhouette −0.005400; DB +0.008785; CH −276.545.
  - The frozen model was not refit; the holdout was not used for tuning.
- Final refit:
  `20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2`
  - n=89,741; clusters 72,211 (80.47%) and 17,530 (19.53%).
  - Verification sample n=3,000: silhouette 0.584193, DB 0.673045, CH 2912.169
    (in-sample descriptive, not selection evidence).
  - Profiles: cluster 0 vocal/non-instrumental, medium-high energy; cluster 1
    instrumental/ambient. Profile table: `profiles.csv`; examples:
    `tables/representative_tracks.csv`.
  - Post-hoc genre: ARI 0.004067, NMI 0.081685 on 73,442 single-genre tracks;
    16,299 multi-genre tracks omitted from the single-label metric. Full multi-label
    incidence: `tables/genre_cluster_incidence.csv`.
  - PCA2D coordinates n=6,000; t-SNE coordinates n=3,000 (visual only).
- Ward corrected: `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`, n=2,000 sample,
  transformer = **frozen evaluation model** (no scaler refit on the sample), ARI vs the frozen
  K-Means **0.883442**; silhouette 0.5945, DB 0.6142, CH 1938.4 on the sample representation;
  not a generalization metric. The historical run
  `20261027T032907Z_f7434a8c_b202fa49_ward_diagnostic` (ARI 0.969205; silhouette 0.7145) refit the
  scaler on the sample, so it is retained as history but is **superseded for the controlled claim**.

## 9. Verification evidence

`btl11 verify` loads the run model, re-predicts the saved cohort, recomputes silhouette/DB/CH,
checks dataset/processed/split/config hashes, artifact checksums, unique IDs, coverage and GMM
posterior row sums. Since the audit it also checks the freeze → holdout → refit chain, descriptor
hashes, the all-track refit scope, stability mean ARI, Ward sample membership and a documented
legacy `descriptor` fallback for historical schemas. Actual passes:

| Run | Checks |
|---|---:|
| Frozen candidate robust K2 | 61 |
| Holdout | 70 |
| Final refit | 84 |
| Stability | 68 |
| Ward historical | 31 |
| Ward corrected | 31 |
| GMM full K4 (baseline) | 65 |
| PCA90 K6 | 63 |
| Eight exploratory controlled GMM runs | 65 each |
| **All runs in `artifacts/INDEX.csv`** | **63/63 PASS, 0 failures** |

Logs: `artifacts/verification/*.json`. No verification failure remains. These checks do not replace peer review.

## 10. Test and lint evidence

- `.venv/bin/pytest -q`: **50 passed** (synthetic fixtures only; includes full tiny pipeline,
  leakage, conflict, PCA, GMM, ARI, profiling, CLI, verify, report-path, immutable-run,
  failed-run, zero-IQR, empty-cluster, tampered-hash, broken-chain and robustness tests).
- `.venv/bin/ruff check src tests scripts`: **All checks passed**.
- `btl11 --help`, `python -m btl11 --help`, `btl11 sweep --dry-run` smoke tests pass; the new
  `btl11 robustness`, `btl11 ward`, `btl11 compare` and `btl11 report` commands ran on the real data.
- `pip-audit -r requirements.lock`: no known vulnerabilities; secret scan found no credentials/.env.
- Clean-copy install (`/tmp` fresh venv, exact lockfile) passed at the 40-test snapshot. A clean-clone
  install after the audit changes was **re-run on 2026-10-27** and passed: fresh venv from a clean
  copy, `pip install -e '.[dev]'`, 50 tests, ruff clean, CLI OK.
- CI workflow is present but no remote CI run was performed (no push).
- `btl11 report` writes repo-relative figure paths; `REPORT_TABLES.md` contains no local absolute
  paths and now indexes 63 runs.
- Deck artifact record: 12 slides, 637,322 bytes, SHA-256
  `e880f7eac1222b60c48294b7a05bc48098a1cafaa949a3046531a3ed9b185ba7`. The 2026-10-27 audit added
  one feature-dominance caveat to the conclusion slide; slide count/design unchanged. ZIP/SHA/text
  integrity checked. A 12-page LibreOffice PDF render exists (`deliverables/BTL11_defense.pdf`,
  SHA-256 `9231daa77b105805e5b7d6d5080a374c442cdcc37926388636c6716cd5e55dcb`); **human visual/timing
  rehearsal remains**.

## 11. Experiment matrix status

| ID | Planned experiment | Status | Evidence |
|---|---|---|---|
| E00 | Unscaled K-Means diagnostic | DONE | `artifacts/ablations/scaler/.../none_unscaled_*` runs |
| E01 | Standard vs Robust scaler | DONE | scaler ablation table |
| E02 | F5 vs F9 | DONE | feature_set ablation table |
| E03 | K-Means K=2..12 | DONE | 11 baseline runs |
| E04 | K-Means multi-seed + bootstrap | DONE | stability run |
| E05 | GMM diag/full K=2..12, BIC/AIC | DONE | 22 baseline runs |
| E06 | GMM responsibilities/uncertainty | DONE | GMM metrics + profile uncertainty table |
| E07 | no-PCA / PCA90 / PCA95 | DONE | pca ablation + PCA diagnostics |
| E08 | PCA2D visualization | DONE | `artifacts/posthoc/<final>/figures/pca2_scatter.png` + CSV |
| E09 | t-SNE sample | DONE | `artifacts/posthoc/<final>/figures/tsne2d.png` + CSV |
| E10 | Ward sample diagnostic | DONE (corrected) | `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`; historical run retained |
| E11 | Genre post-hoc | DONE | final run genre tables/heatmap |
| E12 | Final profiling | DONE | final run profiles/examples |
| E13 | Feature dominance + leave-one-feature-out robustness | DONE (post-hoc, development-only) | `artifacts/robustness/20261027T070249Z/` |
| E14 | Controlled K-Means vs GMM (F9/noPCA; Standard+Robust; K=2/4) | DONE (exploratory) | 8 GMM runs + `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/` |
| P2 | Tree surrogate/rules/DBSCAN/dashboard | DEFERRED | out of scope after P0/P1 |

## 12. Honest ledger of corrections

- The first final refit lacked raw feature values in representative tracks; fixed, then a new final
  refit was created. The old run remains in `INDEX.csv`.
- A cluster hypothesis label was refined from narrow “high-energy/dance” to
  “vocal/non-instrumental, medium–high energy” after inspecting boundary examples; another new
  final refit was created. The latest run is authoritative.
- No experiment fit failed. A deliberately invalid candidate (unscaled + PCA) was used only in a
  test to verify honest `failed` status; it is not a research result.
- Three final-refit runs are retained as immutable history rather than overwritten.
- **2026-10-27 audit:** Ward was found to refit its scaler on the 2,000-track sample. The historical
  run is retained; a corrected frozen-transformer run gives ARI 0.883442 (not 0.969205) and is now
  cited. The report/decision text was corrected.
- **2026-10-27 audit:** the baseline comparison was regenerated with `cohort_id`
  (`20261027T070434Z`); BIC values and ranks are identical to the historical table.
- **2026-10-27 audit:** feature-dominance/LOO and controlled GMM analyses were added. The selected
  model is unchanged and the holdout was not reopened; the interpretation was narrowed to
  instrumental vs vocal/non-instrumental segmentation.
- **2026-10-27 audit:** verification was strengthened; all 63 runs pass. The test suite is 50 tests.
  The defense deck received one caveat on its conclusion slide (12 slides preserved).

## 13. Post-hoc robustness and controlled GMM (2026-10-27)

Commands:

```bash
.venv/bin/python scripts/run_controlled_gmm.py --config configs/base.yaml
btl11 robustness \
  --evaluation-run 20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42 \
  --config configs/base.yaml
btl11 ward --config configs/base.yaml
```

Artifacts:

- `artifacts/robustness/20261027T070249Z/` — `robustness.json`, scaler parameters, variance
  decomposition, assignment contributions, single-feature partitions, LOO table/profiles/seed
  stability, scaler-space table, controlled GMM table, two figures.
- `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/` — 8-run comparison with
  cohort-aware BIC ranks.
- Corrected Ward: `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`.

Headline results (development-only; holdout untouched):

- `instrumentalness` = 97.29% of between-cluster SS and 97.29% of assignment margin (the same
  K=2 δ²/Σδ² quantity computed two ways); single-feature K=2 ARI 0.9892 with 7/3,000 eval
  tracks differing (contingency [[585, 4], [3, 2408]]); LOO instrumentalness ARI −0.0498; all
  other LOO ARI ≥ 0.9938.
- Robust vs StandardScaler K=2 partitions agree at ARI 0.1350; Standard is multivariate
  (energy/acousticness/loudness) while Robust is effectively univariate.
- Controlled K-Means vs GMM hard-label ARI −0.0201…0.3063; GMM confidence 0.947–0.997. Among the
  controlled K=2/4 fits, BIC ranks full K=4 first within each scaler space; the Standard K=2..12
  baseline still reaches a lower BIC at K=12 full (678,842). BIC is not comparable across spaces.
- Corrected Ward ARI 0.883442 (sample-only).

Full methods and tables: `docs/ROBUSTNESS_ANALYSIS.md`; findings:
`docs/FINAL_CODE_AUDIT.md`; readiness: `docs/FINAL_READINESS.md`.
