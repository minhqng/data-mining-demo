# BTL11 — Robustness analysis: feature dominance and controlled comparisons

**Analysis date:** 2026-10-27  
**Evidence class:** **post-hoc exploratory**, development-only; **not** a new confirmatory
holdout evaluation. The holdout was not touched.  
**Frozen model (unchanged):**
`20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`  
**Primary artifact:**
`artifacts/robustness/20261027T070249Z/` (`robustness.json`, CSVs, figures)  
**Controlled GMM runs:**
`20261027T065819Z`–`20261027T065917Z` (8 runs, experiment set
`exploratory_controlled_gmm`)  
**Corrected Ward run:**
`20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`

All numbers below come from those artifacts. The frozen model, evaluation run, holdout run,
final refit and historical artifacts were not modified. Every exploratory fit uses the same
development fit IDs (68,813) and fixed eval cohort (3,000) as the frozen protocol.

---

## 1. Question

The final model reports development silhouette 0.5956 and holdout silhouette 0.5902 with
cluster proportions about 80.47% / 19.53% and a very strong `instrumentalness` contrast. The
audit question is whether this apparent clustering quality is substantially driven by one
feature (especially `instrumentalness`) rather than balanced multivariate acoustic structure,
and whether that would indicate a scaling artifact, a legitimate acoustic separation, or an
implementation defect.

The analysis therefore:

1. extracts the exact `RobustScaler.center_` and `scale_`;
2. profiles raw distributions and transformed variances;
3. decomposes within/between-cluster dispersion and K-Means assignment margins by feature;
4. fits single-feature K-Means and leave-one-feature-out (LOO) variants on fit IDs and
   compares assignments on identical eval IDs with ARI;
5. contrasts RobustScaler with the controlled StandardScaler K=2 model;
6. compares K-Means and GMM on identical IDs / feature space / scaler / K / seed.

---

## 2. Exact scaling parameters and raw distributions

Fitted on the 68,813 fit IDs. `center` = median and `scale` = IQR for RobustScaler.
`near_zero` = `|x| < 1e-4`.

| Feature | Raw median | Q1 | Q3 | IQR | zero share | near-zero share | center | scale | transformed var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acousticness | 0.188 | 0.0172 | 0.627 | 0.6098 | 0.00042 | 0.0453 | 0.188 | 0.6098 | 0.3084 |
| danceability | 0.576 | 0.449 | 0.692 | 0.2430 | 0.00183 | 0.00183 | 0.576 | 0.2430 | 0.5289 |
| energy | 0.676 | 0.456 | 0.853 | 0.3970 | 0.00001 | 0.00078 | 0.676 | 0.3970 | 0.4182 |
| valence | 0.456 | 0.249 | 0.681 | 0.4320 | 0.00205 | 0.00350 | 0.456 | 0.4320 | 0.3695 |
| tempo | 122.008 | 99.223 | 140.062 | 40.839 | 0.00183 | 0.00183 | 122.008 | 40.839 | 0.5448 |
| loudness | −7.191 | −10.343 | −5.108 | 5.235 | 0 | 0 | −7.191 | 5.235 | 0.9905 |
| speechiness | 0.0489 | 0.0359 | 0.0863 | 0.0504 | 0.00183 | 0.00183 | 0.0489 | 0.0504 | **5.0254** |
| instrumentalness | **0.0000582** | **0** | **0.0985** | **0.0985** | **0.33335** | **0.52117** | 0.0000582 | 0.0985 | **10.8156** |
| liveness | 0.132 | 0.0981 | 0.279 | 0.1809 | 0.00003 | 0.00003 | 0.132 | 0.1809 | 1.1619 |

Two features are transformed into much larger ranges than the rest because their IQR is tiny
relative to their between-track spread: `instrumentalness` (var 10.82) and `speechiness`
(var 5.03). `instrumentalness` is strongly zero-inflated: one third of tracks are exactly zero
and more than half are below 1e-4. This is a property of the raw feature, not a pipeline
artefact.

Artifacts: `scaler_parameters.csv`, `robustness.json` → `scaler_parameters`.

---

## 3. Variance decomposition and assignment attribution

For the frozen K-Means fit on the fit cohort (n = 68,813), with transformed matrix `Z`, assigned
labels `a(i)`, centroids `c_k`, and per-feature sums of squares:

- total SS per feature = between-cluster SS + within-cluster SS (ANOVA identity);
- `between_share` (η²) = between SS_j / total SS_j;
- `share_of_total_between` = between SS_j / Σ_j between SS_j;
- `share_of_mean_margin` = mean_i(m_ij) / mean_i(m_i), where
  m_ij = (z_ij − c_other(i),j)² − (z_ij − c_a(i),j)² is feature j's contribution to the
  squared-distance margin between the assigned centroid and the nearest alternative centroid,
  and m_i = Σ_j m_ij;
- `flip_share_if_feature_removed` = fraction of fit points with m_i − m_ij < 0, i.e. whose
  assignment would reverse if that feature's contribution were removed from the margin.

For K=2 the nearest alternative is always the other centroid, so
between SS_j = (n₀n₁/n)·δ_j² and mean_i(m_ij) = δ_j², where δ_j is the centroid difference on
feature j. Both shares therefore equal δ_j²/Σ_jδ_j²: the two 97.29% figures below are two
computational routes to the same K=2 quantity, **not independent corroboration**.

| Feature | between share (η²) | share of total between SS | share of mean margin | flip share if removed |
|---|---:|---:|---:|---:|
| acousticness | 0.00754 | 0.00022 | 0.00023 | 0.00007 |
| danceability | 0.03130 | 0.00160 | 0.00160 | 0.00036 |
| energy | 0.02801 | 0.00113 | 0.00113 | 0.00029 |
| valence | 0.09558 | 0.00341 | 0.00341 | 0.00036 |
| tempo | 0.00267 | 0.00014 | 0.00014 | 0.00004 |
| loudness | 0.16309 | 0.01561 | 0.01561 | 0.00132 |
| speechiness | 0.00873 | 0.00424 | 0.00424 | 0.00070 |
| **instrumentalness** | **0.93111** | **0.97288** | **0.97287** | **0.30048** |
| liveness | 0.00685 | 0.00077 | 0.00077 | 0.00019 |

`instrumentalness` alone carries **97.29%** (0.972878) of the total between-cluster separation
and **97.29%** (0.972872) of the assignment margin; removing it would flip **30.05%** of fit
assignments. The 6.7e-6 difference between the two shares is numerical: the stored centroids
differ from the final-label group means by at most 8.6e-4 (a Lloyd final-label artifact), so the
identity in §3 holds to that precision.
`speechiness` is the largest *within-cluster* dispersion contributor (50.77% of total within
SS) but contributes only 0.42% of the margin; it adds within-cluster spread, not between-cluster
separation.

Artifacts: `variance_decomposition.csv`, `assignment_contributions.csv`.

---

## 4. Single-feature K-Means

A K=2 K-Means fit on one feature at a time (fit IDs; eval predictions; ARI vs the frozen full
model on the same 3,000 eval IDs):

| Feature | ARI vs full | 1-D silhouette | eval shares (cluster 0 / 1) |
|---|---:|---:|---:|
| acousticness | 0.0131 | 0.7377 | 34.3% / 65.7% |
| danceability | 0.0202 | 0.5596 | 42.2% / 57.8% |
| energy | 0.0289 | 0.6065 | 39.3% / 60.7% |
| valence | 0.0277 | 0.6079 | 47.7% / 52.3% |
| tempo | −0.0006 | 0.5449 | 48.4% / 51.6% |
| loudness | 0.2208 | 0.6634 | 16.7% / 83.3% |
| speechiness | −0.0501 | 0.8456 | 93.7% / 6.3% |
| **instrumentalness** | **0.9892** | 0.9146 | **19.6% / 80.4%** |
| liveness | −0.0356 | 0.7686 | 89.7% / 10.3% |

The one-dimensional silhouette values are **not** comparable to the 9-D full-model silhouette;
they are shown only to describe each 1-D split. The comparable quantity is ARI on identical
IDs. `instrumentalness` alone reproduces the full partition almost exactly (ARI 0.9892;
**7 of 3,000** eval tracks differ), while every other single feature fails (next best:
`loudness`, ARI 0.2208).

Exact eval contingency (rows = frozen 9-D partition, columns = 1-D instrumentalness partition):

| | 1-D cluster 0 | 1-D cluster 1 | total |
|---|---:|---:|---:|
| **frozen cluster 0** | 585 | 4 | 589 |
| **frozen cluster 1** | 3 | 2408 | 2411 |
| **total** | 588 | 2412 | 3000 |

The cluster sizes move from 589/2,411 to 588/2,412 — a **net** change of one — but **seven**
tracks are reassigned: 4 leave the vocal cluster and 3 leave the instrumental cluster. The
"one track" wording in the 2026-10-27 audit confused the net size delta with the symmetric
difference. On the 68,813 fit IDs the same comparison gives ARI 0.9905 with 141 differing
assignments, so the reported 0.9892 is unambiguously the eval-cohort value, not a fit-cohort
value. The fitted 1-D boundary corresponds to raw `instrumentalness` ≈ 0.413.

Artifact: `single_feature_partitions.csv`.

---

## 5. Leave-one-feature-out sensitivity

Each variant refits `RobustScaler` + K-Means K=2 (seed 42, `n_init=20`) on the fit IDs with one
feature removed and predicts the same 3,000 eval IDs. The "none" row is the frozen reference.
Silhouette/DB/CH are in the variant's own space and are contextual; ARI vs the full model is
the comparable partition-change measure.

| Removed feature | ARI vs full | silhouette (own space) | min / max eval share | seed-ARI mean |
|---|---:|---:|---:|---:|
| none (frozen reference) | 1.0000 | 0.5956 | 19.63% / 80.37% | 0.9997 |
| acousticness | 0.9985 | 0.6042 | 19.67% / 80.33% | 1.0000 |
| danceability | 1.0000 | 0.6106 | 19.63% / 80.37% | 1.0000 |
| energy | 0.9985 | 0.6062 | 19.67% / 80.33% | 1.0000 |
| valence | 1.0000 | 0.6064 | 19.63% / 80.37% | 1.0000 |
| tempo | 1.0000 | 0.6126 | 19.63% / 80.37% | 1.0000 |
| loudness | 0.9938 | 0.6125 | 19.57% / 80.43% | 1.0000 |
| speechiness | 0.9985 | 0.6655 | 19.67% / 80.33% | 1.0000 |
| **instrumentalness** | **−0.0498** | 0.6010 | **6.50% / 93.50%** | 1.0000 |
| liveness | 0.9985 | 0.6260 | 19.67% / 80.33% | 1.0000 |

Interpretation:

- The partition is **not sensitive to any feature except `instrumentalness`**. Dropping any
  other feature leaves ARI ≥ 0.9938.
- Dropping `instrumentalness` does not yield a noisy version of the same partition; it yields a
  **different** stable partition (ARI −0.0498). The replacement split is a speechiness split:
  its two eval clusters have speechiness medians 0.0465 vs 0.3430, while `instrumentalness`
  medians are both ~0. That is exactly what the IQR scaling explanation predicts: after the
  dominant low-IQR feature is removed, the next low-IQR feature (`speechiness`) takes over.
- Every LOO variant is itself perfectly seed-stable (seed ARI mean 1.0). The instability is in
  the *choice of feature*, not in K-Means optimization.

Raw-unit medians for the frozen and instrumentalness-out variants (eval cohort):

| Variant / cluster | n | acoustic | dance | energy | instrumental | speech | valence | tempo | loudness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen C0 | 589 | 0.096 | 0.524 | 0.588 | **0.854** | 0.0477 | 0.237 | 123.98 | −10.31 |
| Frozen C1 | 2,411 | 0.177 | 0.585 | 0.685 | **~0** | 0.0489 | 0.507 | 122.06 | −6.68 |
| LOO-instrumentalness C0 | 2,805 | 0.162 | 0.569 | 0.670 | 0.0001 | 0.0465 | 0.452 | 122.99 | −7.18 |
| LOO-instrumentalness C1 | 195 | 0.226 | 0.683 | 0.692 | 0 | **0.343** | 0.531 | 116.50 | −7.05 |

Artifacts: `leave_one_feature_out.csv`, `loo_profiles.csv`, `loo_seed_stability.csv`,
`figures/loo_ari.png`.

---

## 6. Controlled scaler comparison

The same K-Means K=2 / F9 / no-PCA / seed-42 / fit-eval protocol was applied under
StandardScaler (`20261027T031534Z_..._standard_nopca_k2_s42`) and RobustScaler (the frozen
model). Cross-scaler partition **ARI = 0.1350** on identical eval IDs: the two scalers find
substantially different structures.

| Feature | Robust between share | Standard between share | Standard share of total between | Standard single-feature ARI |
|---|---:|---:|---:|---:|
| acousticness | 0.0075 | 0.5172 | 26.48% | 0.4459 |
| danceability | 0.0313 | 0.1127 | 5.77% | 0.0934 |
| energy | 0.0280 | 0.5755 | **29.46%** | 0.4474 |
| valence | 0.0956 | 0.1298 | 6.64% | 0.0644 |
| tempo | 0.0027 | 0.0595 | 3.05% | 0.0316 |
| loudness | 0.1631 | 0.4493 | **23.00%** | **0.4942** |
| speechiness | 0.0087 | 0.0218 | 1.11% | −0.0446 |
| instrumentalness | **0.9311** | 0.0771 | **3.95%** | 0.1300 |
| liveness | 0.0069 | 0.0104 | 0.53% | −0.0269 |

Under StandardScaler, `energy` (29.46%), `acousticness` (26.48%) and `loudness` (23.00%) share
the between-cluster separation, and the best single-feature reproduction is only ARI 0.4942
(`loudness`); `instrumentalness` contributes 3.95%. This is a genuinely multivariate
partition, but its in-space silhouette is lower (0.2576 vs 0.5956 for RobustScaler).

Neither scaler is universally better: they answer different segmentation questions. The raw
silhouette values **must not** be ranked as if the two distance geometries were a common axis.
Artifacts: `scaler_space_feature_attribution.csv`, `figures/feature_dominance.png`.

---

## 7. Controlled K-Means vs GMM comparison

Eight exploratory GMM fits were run with the same fit/eval IDs, F9 feature order, no PCA,
seed 42 and K as the corresponding K-Means models; GMM `n_init=5`, K-Means `n_init=20`.
Covariance `diag`/`full` is recorded. ARI is computed between the hard labels of the two
algorithms on the same eval IDs. BIC ranks only within one scaler space/cohort; raw BIC across
Standard/Robust is not comparable.

| Space | K | Cov | K-Means sil. | GMM sil. | ARI KM vs GMM | GMM confidence | GMM norm. entropy | GMM BIC | BIC rank in space | GMM p |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 2 | diag | 0.5956 | 0.2388 | 0.3021 | 0.9949 | 0.0188 | 1,147,820 | 4 | 37 |
| Robust | 2 | full | 0.5956 | 0.2401 | 0.3063 | 0.9965 | 0.0134 | 969,983 | 2 | 109 |
| Robust | 4 | diag | 0.4884 | 0.0398 | 0.1835 | 0.9462 | 0.0971 | 986,830 | 3 | 75 |
| Robust | 4 | full | 0.4884 | 0.0674 | 0.2538 | 0.9628 | 0.0775 | 819,184 | 1 | 219 |
| Standard | 2 | diag | 0.2576 | 0.1136 | 0.0704 | 0.9935 | 0.0250 | 1,203,240 | 3 | 37 |
| Standard | 2 | full | 0.2576 | 0.0961 | −0.0201 | 0.9510 | 0.1864 | 1,397,164 | 4 | 109 |
| Standard | 4 | diag | 0.1742 | 0.0579 | 0.1227 | 0.9489 | 0.0948 | 1,006,070 | 2 | 75 |
| Standard | 4 | full | 0.1742 | 0.0547 | 0.1125 | 0.9577 | 0.0803 | 853,059 | 1 | 219 |

All 8 GMM fits converged and all 8 runs verify with 65 checks each.

Reading:

- **Hard partitions disagree**: ARI is between −0.0201 and 0.3063. At Robust K=2, about 70% of
  the assignment structure differs even though both models see the same 9-D space and K.
- **GMM posteriors are confident**: mean max posterior 0.947–0.997. High confidence does not
  mean the GMM recovered the K-Means partition; it means each model is confident about its own
  (different) components.
- **Hard-label separation is lower for GMM**: within each row, GMM hard-label silhouette is
  below K-Means. This is expected when a density model uses components to fit skew/outliers
  rather than to maximize centroid separation.
- **BIC prefers full K=4 among the controlled K=2/4 fits** (Standard 853,059; Robust 819,184),
  but BIC is a penalized likelihood criterion, not a segmentation criterion. The original
  Standard K=2..12 sweep reaches a lower BIC at K=12 full (678,842), which is a density-boundary
  result, not evidence for a better segmentation. The Robust K=2 full BIC (969,983) is worse than
  Robust K=4 full, while the K=2 hard-label separation is much better. This is the density-fit vs
  hard-segmentation distinction, not a contradiction.

Artifacts: `controlled_model_comparison.csv`,
`artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/comparison.csv`.
Run IDs: see `artifacts/INDEX.csv`, experiment set `exploratory_controlled_gmm`.

One small note: the baseline Standard full K=4 fit at `n_init=2` has BIC 853,058 while the
controlled `n_init=5` fit has 853,059; the one-unit difference is EM-optimization noise and does
not change any conclusion.

---

## 8. Ward correction

The historical Ward run
`20261027T032907Z_f7434a8c_b202fa49_ward_diagnostic` refit `RobustScaler` on the 2,000-track
sample. That made the ARI vs the frozen K-Means a comparison of *both* a different clustering
algorithm and a different scaler fit. Re-running Ward on the same 2,000 IDs with the frozen
evaluation model's transformer gives:

| Ward representation | ARI vs frozen K-Means | Silhouette (sample) | DB | CH |
|---|---:|---:|---:|---:|
| Historical sample-refit scaler | 0.969205 | 0.7145 | 0.4273 | 4797.0 |
| **Frozen transformer (corrected)** | **0.883442** | 0.5945 | 0.6142 | 1938.4 |

The corrected ARI is lower but still high: the broad two-way structure is not unique to
K-Means. It is a sample-only diagnostic; it is not a generalization metric, and its silhouette
is not comparable to the fixed eval cohort.

Run: `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`.

---

## 9. Verdict: legitimate separation, scaling-amplified, not an implementation defect

**Legitimate acoustic separation.** Raw `instrumentalness` is strongly bimodal. In the frozen
eval cohort, 94.7% of the instrumental cluster has `instrumentalness ≥ 0.5`, while 0% of the
other cluster does. The partition is stable across seeds and bootstrap and has a clear
raw-unit interpretation. The pipeline computed the model and metrics correctly; the stored
numbers reproduce exactly.

**Scaling-amplified dominance.** RobustScaler divides by the IQR. `instrumentalness` has
IQR 0.0985 and a large between-track spread, so its transformed variance is ~10.8 versus <1
for most features. The result is a nearly univariate partition. Under StandardScaler the same
K=2 model is multivariate (energy/acousticness/loudness) and the two partitions agree at only
ARI 0.1350. Removing `instrumentalness` does not produce noise: it produces a different stable
split dominated by the next low-IQR feature, `speechiness`.

**Not a defect.** There is no leakage, no incorrect distance, no wrong metric and no
reproducibility failure. The effect is a data/representation property that the report must
state. The selected model remains defensible for **instrumental vs vocal/non-instrumental
segmentation**, which is a useful and stable acoustic distinction. It is **not** defensible as
evidence of balanced nine-feature multivariate personas, and the high silhouette should not be
read as "the data contain two balanced acoustic clusters" without the dominance caveat.

**Unresolved limitation.** K=2 is coarse; K=6 is richer but less stable (bootstrap ARI 0.9164)
and contains a ~1.1% cluster. A future study could pre-register a multivariate feature space
(e.g., drop/limit `instrumentalness`, or use StandardScaler) and evaluate that segmentation on
a fresh holdout; this audit did not reopen the existing holdout.

---

## 10. Reproduce

```bash
# Frozen model and historical evidence are read-only inputs.
btl11 robustness \
  --evaluation-run 20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42 \
  --config configs/base.yaml

# Controlled GMM exploration (8 new candidate runs; separate experiment set).
.venv/bin/python scripts/run_controlled_gmm.py --config configs/base.yaml

# Corrected Ward representation.
btl11 ward --config configs/base.yaml
```

Outputs: `artifacts/robustness/<timestamp>/robustness.json`, the CSV tables and
`figures/feature_dominance.png`, `figures/loo_ari.png`. Every run is verified with
`btl11 verify`.
