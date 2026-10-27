# Methodology — scientific protocol and safeguards

This document explains why the pipeline is statistically defensible. It does not replace the
normative SSOT (`docs/BTL11_SSOT.md`) or the evidence ledger (`docs/EXPERIMENTS.md`).

## 1. Research questions

1. Do tracks form meaningful acoustic clusters when only the nine provided audio features are used?
2. How do hard K-Means partitions differ from probabilistic GMM components?
3. Which number of clusters is defensible when validity, balance, stability, interpretability and
   parsimony are considered together?
4. Can the resulting clusters be described as acoustic personas without claiming they are genres?

`track_genre` is never a target, never a feature and never a tuning signal. It is opened only after
the model is frozen.

## 2. Unit of analysis and cleaning

The primary unit is one unique `track_id`, not one CSV row. The same Spotify track can appear under
multiple genres. Keeping all rows would up-weight multi-genre tracks in centroid estimation and
inflate density. The protocol therefore:

1. drops the Kaggle export index column;
2. validates `track_id`, the nine audio features and `track_genre`;
3. checks repeated IDs for audio-vector agreement at `atol=1e-8, rtol=1e-6`;
4. keeps one deterministic row per ID (sorted by `track_id`, genre, source row) and stores all
   unique `(track_id, genre)` pairs in a separate bridge;
5. fails closed on conflicting vectors instead of `keep='first'` or averaging;
6. records exact duplicates, multi-genre counts, missingness and range/outlier flags in
   `data/processed/audit.json`.

Outliers are reported, not automatically removed or winsorized. `tempo == 0` and `loudness > 0` are
quality flags, not automatic exclusions. No imputation is used in the primary path; missing feature
rows would be quarantined after audit.

## 3. Splits and evaluation cohorts

Let \(U\) be the set of unique tracks. A stable hash of `split_seed:track_id` assigns roughly 80% to
development \(D\) and 20% to holdout \(H\), with \(D \cap H=\varnothing\). A further stable hash
selects a fixed cohort \(E \subset D\) of 3,000 tracks for model comparison. The fit set is
\(F=D\setminus E\), with 68,813 tracks in the real run. Thus every candidate is fitted without the
3,000 IDs used to compare it. The same \(E\) is used for all candidates in a controlled comparison.

The holdout evaluation cohort is drawn from \(H\) with a separate deterministic hash. \(H\) is used
once, after the winner is frozen; no selection feedback is allowed. The final full-data refit uses
\(U\) and has a distinct model identity.

## 4. Representations

- **F9 primary:** acousticness, danceability, energy, valence, tempo, loudness, speechiness,
  instrumentalness, liveness.
- **F5 required ablation:** acousticness, danceability, energy, valence, tempo.
- **StandardScaler primary:** \(z_{ij}=(x_{ij}-\mu_j)/\sigma_j\).
- **RobustScaler ablation/final candidate:**
  \(z^{r}_{ij}=(x_{ij}-\mathrm{median}(x_j))/(Q_{0.75}(x_j)-Q_{0.25}(x_j))\).
- **PCA:** `PCA(variance=0.90/0.95)`, `whiten=False`, fitted after scaling and only on \(F\).
  The actual component count is data-determined. PCA95 is not 95% cluster accuracy; PCA2D is a
  display projection, not the training representation by default.
- **t-SNE:** visualization only. It is not used to choose K, and silhouette is never computed on
  its coordinates.

## 5. Clustering models

### K-Means

For centroids \(\mu_k\):

\[
J=\sum_{k=1}^{K}\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert_2^2 .
\]

The implementation uses scikit-learn `KMeans(init='k-means++', n_init=20, max_iter=500,
tol=1e-4, random_state=seed, algorithm='lloyd')`. `n_init=20` reduces initialization dependence;
`k-means++` is the careful seeding strategy. The elbow is exploratory only.

### GMM

\[
p(x)=\sum_{k=1}^{K}\pi_k\mathcal N(x\mid\mu_k,\Sigma_k),\qquad
\gamma_{ik}=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}.
\]

`GaussianMixture` is fitted by EM (`n_init=2` quick sweep, `n_init=5` finalists), covariance
`diag`/`full`, bounded `reg_covar` only for numerical issues. Hard labels are
`argmax(predict_proba)`; confidence is \(\max_k\gamma_{ik}\); entropy is
\(-\sum_k\gamma_{ik}\log\gamma_{ik}\), normalized by \(\log K\). These are mixture-component
posteriors conditional on the fitted model, **not genre probabilities**.

\[
\mathrm{BIC}=-2\log\hat L+p\ln n,\qquad \mathrm{AIC}=-2\log\hat L+2p.
\]

For dimension \(d\) and \(K\) components:
\(p_{\text{diag}}=K(2d+1)-1\),
\(p_{\text{full}}=(K-1)+Kd+Kd(d+1)/2\).
For \(d=9,K=12\) these are 227 and 659 free parameters. BIC is compared only within the same
representation, fit cohort and likelihood base measure. A lower BIC indicates a better penalized
Gaussian density fit, not automatically better semantic clusters.

## 6. Validity metrics and guards

On the fixed evaluation cohort:

- Silhouette \(s(i)=(b(i)-a(i))/\max(a(i),b(i))\) — maximize.
- Davies–Bouldin — minimize.
- Calinski–Harabasz — maximize.
- Cluster sizes/share, empty clusters, minimum share, convergence.
- K-Means inertia/SSE; GMM BIC/AIC, log-likelihood and posterior uncertainty.

Silhouette is defined only when \(2 \le K_{\text{observed}} \le n-1\). Otherwise the pipeline stores
`value=null` and an explicit reason, never 0. The full pairwise silhouette on 89,741 tracks is not
computed; the fixed cohort bounds the workload. DB/CH use the same cohort for comparability. Metrics
are contextual to the representation; raw values across F5/F9, Standard/Robust, PCA/no-PCA are not a
single common axis.

## 7. Model selection

Selection is multi-criterion and explicitly excludes invalid/collapsed fits:

1. convergence and non-empty components;
2. cluster balance and absence of unexplained tiny clusters;
3. silhouette/DB/CH in the same controlled comparison;
4. seed and bootstrap stability on identical reference IDs;
5. BIC within a GMM representation, without crossing spaces;
6. interpretability of raw-unit profiles and representative tracks;
7. parsimony.

No composite “grade score” is invented. The final decision is recorded in `docs/DECISIONS.md` and
`configs/base.yaml`. On this dataset, RobustScaler/K=2 was selected because its in-space validity,
near-perfect stability, interpretability and simplicity dominated the alternatives; its balance is
adequate (19.55% minimum fit share) but not maximal (Standard K2 24.89%; F5 K2 30.27%). K=6-standard
was richer but less stable and had a 1.14% fit cluster (782/68,813). The holdout check was performed
only after this freeze.

The 2026-10-27 audit adds an important qualification: the selected K=2 partition is overwhelmingly
driven by `instrumentalness` (97.29% of between-cluster SS; single-feature ARI 0.9892 with
7/3,000 eval tracks differing), so it is
described as an **instrumental vs vocal/non-instrumental segmentation**, not as balanced
multivariate persona discovery. See `docs/ROBUSTNESS_ANALYSIS.md` and D-16 in `docs/DECISIONS.md`.

## 8. Stability

Seed stability refits the full transformer + estimator with five different `random_state` values and
predicts the **same** evaluation IDs. Bootstrap stability resamples fit rows with replacement and
again predicts the same reference IDs. Pairwise agreement is measured by Adjusted Rand Index, which
is invariant to cluster-label renaming. ARI is never computed between labels from different row sets.
Both seed and sampling uncertainty are reported separately.

## 9. Holdout and final refit

The frozen development model is applied to a 3,000-track holdout cohort without refitting. This
checks out-of-sample assignment/geometry sanity, not classification accuracy, since no ground-truth
labels exist. After that, the selected configuration is refitted on all 89,741 unique tracks to
produce assignments, raw-unit profiles and visuals. The full refit has a distinct run ID and does
not inherit holdout metrics.

The 2026-10-27 robustness/LOO and controlled-GMM analyses use **development IDs only**; the holdout
was not reopened. They are post-hoc exploratory evidence, not a second confirmatory evaluation.

## 10. Cluster interpretation and genre post-hoc

Profiles are computed in **raw feature units** with mean, median, quartiles, IQR, global deviations
and representative local tracks (near-centroid/high-posterior and boundary examples). Candidate and
holdout profiles use the development split as the global reference; only the final full refit uses
all tracks. Cluster names are acoustic hypotheses, not verified genres; the pipeline does not listen
to audio.

Genre validation uses a multi-label incidence heatmap with the number of unique cluster tracks as
denominator; a row can sum above 100% because a track can carry multiple genres. A single-genre-only
cohort is declared for ARI/NMI, with omitted multi-genre tracks reported. Genre agreement is
diagnostic only and is never used to tune the model.

## 11. Reproducibility and limitations

Each run stores a config snapshot, dataset/processed/split hashes, seed, feature order, scaler/PCA
components, algorithm/K/covariance, fit/eval counts, metrics, warnings, convergence, model and
assignment checksums. `btl11 verify` recomputes hashes, ID coverage, predictions and saved validity
metrics, and checks the freeze → holdout → refit chain. The audit verified **63/63 completed runs**;
the synthetic test suite has **50 tests**. Determinism is claimed within this environment; different
BLAS/platform builds can yield small numerical differences. Spotify audio features are algorithmic
descriptors, not raw audio. The dataset's license/provenance remains a human gate. See
`docs/REPORT.md` for the full threats-to-validity discussion.

## 12. Post-hoc robustness audit (2026-10-27)

`btl11 robustness` writes `artifacts/robustness/<timestamp>/` and never mutates a run. It:

1. extracts the frozen RobustScaler center/scale and raw/transformed feature distributions;
2. decomposes between-cluster and within-cluster SS and K-Means assignment margins by feature;
3. fits single-feature K-Means and leave-one-feature-out variants on fit IDs, predicting the same
   eval IDs;
4. compares RobustScaler and StandardScaler K=2 partitions on identical IDs with ARI;
5. joins the exploratory controlled K-Means/GMM runs into a table with hard-label ARI, posterior
   confidence/entropy and within-space BIC.

Key result: `instrumentalness` dominates the robust K=2 partition; the model is defensible for an
instrumental/vocal segmentation, not for balanced multivariate structure. Ward is corrected to use
the frozen transformer (controlled ARI 0.8834 rather than the historical 0.9692 sample-refit
value). Full methods, tables and caveats: `docs/ROBUSTNESS_ANALYSIS.md`.
