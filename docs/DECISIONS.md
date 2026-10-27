# Decision record — BTL11

The normative SSOT remains `docs/BTL11_SSOT.md`. This file records material decisions, their
evidence and any unresolved gates. No numeric claim here overrides run artifacts.

## A. Locked decisions inherited from the SSOT

| ID | Decision | Status |
|---|---|---|
| ADR-01 | Exact Kaggle Spotify Tracks Dataset only | Locked; file present and hashed |
| ADR-02 | Unsupervised acoustic segmentation; genre post-hoc only | Locked; tested |
| ADR-03 | F9 primary, F5 ablation | Locked; F5 tested |
| ADR-04 | StandardScaler primary, RobustScaler ablation | Locked, but data evidence supports selecting RobustScaler for the final model (see D-05) |
| ADR-05 | K-Means + GMM mandatory; PCA mandatory; t-SNE visual only | Locked |
| ADR-06 | Staged K search, not a full Cartesian grid | Locked |
| ADR-07 | Ward on a bounded sample as conditional diagnostic | Executed on n=2,000 |
| ADR-08 | Tree surrogate / association rules are P2 extensions only | Deferred |
| ADR-09 | No genre/metadata in primary X | Locked; leakage tests |
| ADR-10 | Every metric records space, cohort, sampling and seed | Implemented |
| ADR-11 | K-Means final + GMM complementary, or vice versa | Implemented; final selection below |
| ADR-12 | CLI + artifacts + report before any dashboard | Implemented; no dashboard |

## B. New decisions from actual execution

### D-01 — License/provenance remains pending; local analysis allowed, redistribution blocked

Observed on 2026-10-27: Kaggle page metadata lists **ODbL 1.0**, Hugging Face mirror lists **BSD**,
and the source states the data was collected via the **Spotify Web API**. The discrepancy and
Spotify Platform/Content restrictions are unresolved. The project owner supplied the snapshot and
authorized local university analysis; this is not a redistribution license.

Consequences:

- `MANIFEST.json`: `license_review_status = pending_human_verification`,
  `redistribution_allowed = false`, `local_analysis_authorized_by_owner = true`,
  `license_metadata_observed.discrepancy = true`;
- raw CSV is gitignored and was never published;
- public release/deployment remains a **human gate** (see `docs/DATASET_PROVENANCE.md`).

### D-02 — Python 3.13 runtime, while code targets 3.11+

Only Python 3.13.16 was available. The package uses 3.11-compatible syntax and declares
`requires-python >=3.11,<3.14`; CI targets Python 3.12. Exact tested versions are in
`requirements.lock`. This is a runtime deviation, not a research-contract change.

### D-03 — Fixed selection cohort is excluded from fitting

Dev/holdout are hash-split at the ID level. A deterministic 3,000-ID cohort inside dev is reserved
for model comparison; fitting uses the remaining 68,813 dev IDs. The holdout cohort is 3,000 IDs
inside the 17,928-track holdout. This prevents in-sample evaluation optimism and ID leakage.
The observed split sizes are recorded in `data/processed/prepare_summary.json`.

### D-04 — GMM BIC boundary at K=12 was not used to select K

On F9/Standard/no-PCA, GMM full-covariance BIC decreased monotonically to the K=12 search boundary
(BIC 678,842; rank 1 within this representation), while hold-one-out-free silhouette became
negative at higher K (full K=12: −0.0234; full K=11: −0.0360; diag K=11: −0.00009). Diag K=12
BIC 767,035. This is evidence that a better penalized density fit does not imply better semantic
segmentation; extending K was not justified. BIC was never compared across feature spaces.

### D-05 — Final model: K-Means, F9, RobustScaler, no PCA, K=2, seed 42

Frozen **before** the one-time holdout check. Evidence from the fixed dev/eval cohort:

| Candidate | Silhouette | DB ↓ | CH | Min share (fit) | Seed ARI | Bootstrap ARI | Rationale |
|---|---:|---:|---:|---:|---:|---:|---|
| KMeans std K=2 | 0.2576 | 1.5847 | 800.6 | 0.2489 | 1.0000 | 0.9914 | Simple, balanced, high CH |
| KMeans std K=4 | 0.1742 | 1.6877 | 588.5 | 0.0776 | 0.9969 | 0.9809 | Interpretable, stable |
| KMeans std K=6 | 0.1896 | 1.4341 | 556.7 | 0.0114 | 0.9334 | 0.9164 | Rich profiles, but 1.14% fit tiny cluster (782/68,813; 814/71,813 dev) and lower stability |
| KMeans robust K=2 | **0.5956** | **0.6505** | **3240.6** | 0.1955 | **0.99969** | **0.99907** | Best in-space validity, adequate (not maximal) balance, near-perfect stability, simple, interpretable |
| KMeans PCA90 K=6 | 0.2033 | 1.3613 | 614.5 | 0.0114 | 0.9987 | 0.9827 | Marginal gain, extra complexity/tiny cluster |
| GMM full K=4 | 0.0547 | 2.5828 | 233.4 | 0.1376 | 0.9786 | 0.9370 | Soft-membership complement; weaker hard separation |

Cross-space silhouette/DB/CH are contextual; the decision does not rest on raw scale-free ranking.
RobustScaler receives this evidence-based exception under ADR-04 because it combines the best
in-space separation with the most stable assignments and a clear acoustic interpretation. Its
minimum fit share is 19.55%, lower than Standard K2 (24.89%) and F5 K2 (30.27%), so "balance" is
adequate but **not** the best among candidates. The 2026-10-27 audit additionally shows that the
RobustScaler K=2 partition is overwhelmingly driven by `instrumentalness` (see D-16); the model is
retained as an instrumental vs vocal/non-instrumental segmentation, not as a balanced multivariate
persona model.

Holdout (frozen, n=3,000): silhouette **0.5902**, DB **0.6593**, CH **2964.07**; deltas vs dev:
−0.0054, +0.0088, −276.5. This supports out-of-sample structural sanity. Holdout was not used to
change the selection.

### D-06 — GMM full K=4 kept as probabilistic complement, not final

GMM full K=4 has BIC 853,058, AIC 851,057, silhouette 0.05469, mean posterior confidence 0.9578
(eval cohort); 0.33% of eval tracks (10/3,000) and 0.30% of dev assignments (214/71,813) are below 0.5 confidence. It is useful for soft membership
and borderline analysis, but its hard-label geometry is much weaker than the selected K-Means. It
is not presented as a genre classifier.

The 2026-10-27 controlled comparison adds ARI between K-Means and GMM hard labels on identical
IDs: across Standard/Robust × K=2/4 × diag/full the ARI is **−0.0201…0.3063**, while GMM
confidence is high (0.947–0.997). GMM is therefore a different segmentation, not merely a noisier
K-Means; it optimizes density/likelihood, whereas K-Means optimizes hard SSE. See D-18 and
`docs/ROBUSTNESS_ANALYSIS.md`.

### D-07 — K=6 standard is an exploratory richer segmentation, not the final

K=6 has interpretable extra personas (instrumental/ambient, speech-heavy) and better silhouette than
K=4 standard, but its seed/bootstrap/pairwise ARI are lower (0.9334/0.9164/0.8696) and its smallest
cluster is 1.14% on the fit cohort (782/68,813; 814/71,813 dev assignments). The project reports it as a trade-off, not as suppressed evidence.

### D-08 — PCA90 not selected despite small metric gains

PCA90 retains 7 components with cumulative variance 0.94823 (PCA95: 8 components, 0.98444). It
slightly improves K-Means validity at K=2/4/6, but the primary no-PCA representation is simpler,
keeps all F9 loadings directly interpretable, and avoids adding a representation whose BIC would not
be comparable across spaces. PCA diagnostics are retained and reported.

### D-09 — E00 unscaled is diagnostic only

Unscaled K-Means yields large apparent silhouette/CH (e.g., K=2 silhouette 0.5167, CH 4883.7) in a
different distance geometry dominated by tempo/loudness. Those numbers are **not** comparable with
scaled spaces and are never used to recommend the unscaled model.

### D-10 — Two-stage lifecycle and distinct final model identity

The evaluation model (dev fit) and final full-data refit are separate run IDs and model hashes.
The full refit does not inherit holdout metrics; its verification sample metrics are descriptive.
Evidence:

- frozen evaluation: `20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`
- holdout: `20261027T032820Z_f7434a8c_b202fa49_holdout_20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_rob`
- final refit: `20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2`

### D-11 — No P2 extensions

Decision-tree surrogate, association rules, DBSCAN, dashboard and autoencoder were deferred. P0/P1
are satisfied; adding P2 would not answer the core research questions and would increase scope risk.
The `extensions` flags in `configs/base.yaml` remain `false`.

### D-12 — No git commit, push, deployment or public data release

The user did not authorize remote/git publication. The repository is a working tree; run manifests
record `git.available=false`. No raw CSV or credentials were committed or uploaded.

### D-13 — Report corrections retained as honest history

Three final-refit runs exist because a profile-display defect (missing raw feature values in
representative tracks) and a hypothesis-label refinement were corrected. All runs remain in the
immutable index. The report cites the latest final refit
`20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2`. Earlier final-refit runs are retained,
not silently deleted.

### D-14 — Governance hardening after independent review

An independent read-only review and evidence audit found four provenance/verification gaps. They
were repaired and covered by regression tests:

- one-time holdout is now scoped to the **descriptor + dataset + split**, so creating a newer
  candidate with the same descriptor cannot reset it;
- refit requires `selected == frozen` and a completed holdout, and it uses the exact run that the
  holdout validated as its evaluation model; report rejects a broken chain;
- a prior human license decision is preserved only when the raw SHA-256 is unchanged; a new
  snapshot resets to pending review rather than inheriting an old approval/rejection;
- `verify` additionally checks scaler/PCA/covariance/feature-space/n_fit/seed/estimator parameters,
  per-cluster counts, null-metric reasons, artifact-path containment, and recomputes inertia or
  GMM BIC/AIC; verification logs are written under `artifacts/verification/`;
- BIC ranking is grouped by representation **and cohort id**; run reuse compares model-relevant
  config snapshots (kmeans/gmm/pca_whiten);
- PCA2D/t-SNE/profile-derived views write to `artifacts/posthoc/<run_id>/`, leaving completed run
  artifacts unchanged.

The test suite grew to **42 synthetic tests** at that review (50 after the post-audit work in
D-19), including repeat-holdout, freeze/refit, hash-bound license, tampered-run verify,
mixed-cohort BIC, GMM-final-refit and report-path regressions. Residual non-blocking limitation:
the one-time scope is per (descriptor, dataset, split), not a global single-use guarantee across
intentionally changed splits.

### D-15 — Course brief and four instructor decks cross-checked; defense deck generated

On 2026-10-27 the original brief (`DATA MINING PROJECTS.docx`, section III, project 3) and the four
instructor decks (`Data.pptx`, `Cluster Analysis.pptx`, `Classification.pptx`, `Pattern Mining.pptx`)
were located and checked directly rather than relying on the SSOT summary. The brief confirms the
required task: normalize/reduce audio features, apply K-Means and GMM, project with PCA/t-SNE, and
profile clusters on the Kaggle Spotify Tracks Dataset.

The traceability matrix in `docs/REPORT.md` §11 was corrected against the actual slide text:

- *Cluster Analysis* slides 58–61 present generative/likelihood clustering concepts, not a full
  GMM/EM derivation; GMM/EM/BIC is required by the assignment and supported by standard
  references, not claimed as deck content;
- *Classification* slides 52–62 cover metrics, over/underfitting, holdout, cross-validation and ROC,
  not leakage; leakage control is this project's own protocol;
- decision-tree slides use information gain/gain ratio, not a Gini emphasis.

A 12-slide defense deck was generated from the same artifacts (`deliverables/BTL11_defense.pptx`):
problem/questions, data/provenance, EDA, pipeline, K-Means vs GMM, K selection, GMM BIC, ablations,
stability + holdout, profiles/visuals, case study + genre sanity, conclusions/limitations. The
2026-10-27 audit added one feature-dominance caveat to the conclusion slide (D-16), using the
robustness artifact; slide count and design are unchanged. The course-slide cross-check gate is
closed; license/redistribution and remote publication remain human gates.

### D-16 — Feature dominance quantified; final model interpretation narrowed

The 2026-10-27 robustness audit (`docs/ROBUSTNESS_ANALYSIS.md`,
`artifacts/robustness/20261027T070249Z/`) quantifies what the frozen model is actually separating.
Under the frozen RobustScaler, `instrumentalness` accounts for **97.29%** of total between-cluster
sum of squares (0.972878) and **97.29%** of the mean K-Means assignment margin (0.972872). At K=2
these are analytically the same δ²/Σδ² quantity (both shares equal δ_j²/Σ_jδ_j²), so the two
figures are two computational routes, not independent corroboration. A K=2 partition on
`instrumentalness` alone reproduces the full partition at **ARI 0.9892** on the 3,000 fixed eval
IDs: **7 tracks differ**, with contingency [[585, 4], [3, 2408]] and cluster sizes 589/2,411 vs
588/2,412 (the net size change is one; the symmetric difference is seven). On the 68,813 fit IDs
the same comparison gives ARI 0.9905 with 141 differing tracks, so the reported 0.9892 is the
eval-cohort value. Removing `instrumentalness` changes the partition (ARI **−0.0498**; a
speechiness-dominated 6.5%/93.5% split), while removing any other feature leaves ARI ≥ 0.9938. Under StandardScaler the same K=2 model is multivariate
(energy 29.46%, acousticness 26.48%, loudness 23.00% of between SS; instrumentalness 3.95%), and
the two scaler partitions agree at only ARI **0.1350**.

Decision: keep the frozen evaluation model, holdout result and final refit unchanged; do not
retune on the new analysis. Describe the model as an **instrumental vs vocal/non-instrumental
segmentation**, not as balanced nine-feature multivariate persona discovery. The audit does not
show an implementation defect; it shows a representation-dependent single-feature dominance that
must be stated honestly.

### D-17 — Ward controlled representation corrected

The historical Ward run `20261027T032907Z_f7434a8c_b202fa49_ward_diagnostic` refit `RobustScaler`
on the 2,000-track sample before Ward, so its ARI 0.969205 vs K-Means mixed an algorithm change
with a scaler change. The corrected run
`20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr` uses the frozen evaluation
model's fitted transformer and obtains **ARI 0.883442** on the same IDs. `ward_diagnostic()` now
loads and hash-checks the frozen bundle, transforms the sample with it, records the representation
in metrics, and keeps the run sample-only (no `.predict`). The historical run is retained and
marked superseded for the controlled claim; the report uses 0.883442.

### D-18 — Controlled K-Means vs GMM comparison added

Eight exploratory GMM runs (`experiment_set=exploratory_controlled_gmm`) cover F9/no-PCA,
Standard/Robust × K=2/4 × diag/full with the same fit/eval IDs, feature order, seed and K as the
corresponding K-Means runs; GMM uses `n_init=5`. All 8 converge and verify. On identical eval IDs,
K-Means vs GMM hard-label ARI is **−0.0201…0.3063**, while GMM confidence is 0.947–0.997. GMM
therefore yields a different, confident soft/density structure, not merely a weaker copy of
K-Means. Among the controlled K=2/4 fits, full K=4 has the lowest BIC in each scaler space
(Standard 853,059; Robust 819,184), but the original Standard K=2..12 sweep reaches a lower BIC at
K=12 full (678,842), a density-boundary result rather than a segmentation choice. Raw BIC is never
compared across scaler spaces. K-Means remains the segmentation model; GMM remains the
probabilistic complement.

### D-19 — Post-audit code, configuration and verification hardening

Implemented after the independent audit (all covered by `pytest`; 50 tests total):

- `score_clustering` now counts an empty expected cluster as share 0.0 in min/max share;
- GMM covariance default is resolved consistently in the bundle and information criteria;
- `gmm.require_convergence=true` now fails a non-converged GMM candidate;
- candidate/holdout profile references use the development split, not the holdout;
- `ward_diagnostic` uses the frozen selected model's transformer;
- `verify` now checks the freeze → holdout → refit chain, descriptor hashes, full-track refit
  scope, stability mean ARI, Ward sample membership, and the legacy `descriptor` field for
  historical runs;
- `config` rejects unimplemented `impute_dev_only` and requires silhouette cohort settings to
  match the fixed eval cohort;
- `report` rejects a selected refit/holdout/evaluation model from different dataset bytes;
- new `btl11 robustness` command + `scripts/run_controlled_gmm.py`;
- baseline comparison regenerated with `cohort_id` (`20261027T070434Z`; values identical to the
  historical table);
- 63/63 runs verify; report tables regenerated; defense deck caveat added.

Residual non-blocking items: `gmm.reg_covar_fallback` remains unused (all GMM fits converged);
historical holdout/final `run.json` lack `frozen_descriptor` fields (verifier handles the `descriptor`
fallback); remote CI did not run. The clean-clone install was re-run and passed on 2026-10-27.

## C. Open gates

| Gate | Owner action |
|---|---|
| License/redistribution | Verify Kaggle ODbL vs HF BSD and Spotify terms; record decision |
| Remote/public release | Human decision; no push/deploy was performed |
| Course rules / rubric | Four decks and the brief were cross-checked (D-15); official grading rubric was not provided, so no score claim is made |
| Defense rehearsal | Rehearse the rendered `deliverables/BTL11_defense.pdf`; check the added caveat fits visually and adjust timing to the class limit |
