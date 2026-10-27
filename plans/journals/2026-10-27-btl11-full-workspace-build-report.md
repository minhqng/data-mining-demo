---
title: BTL11 full workspace build report
date: 2026-10-27
summary: "Full build report: real Spotify experiments, 54 completed runs, frozen K-Means K=2, verification and deliverables"
---

# BTL11 full workspace build report

# BTL11 full workspace build report

## Status
CODE_COMPLETE / REAL_EXPERIMENTS_EXECUTED / PUBLIC_REDISTRIBUTION_BLOCKED.

## Data
- Exact Kaggle Spotify Tracks Dataset, raw SHA-256 b202fa49...177bd, 114,000 rows.
- 89,741 unique track_ids, 114 genres, 16,299 multi-genre tracks, 113,550 genre bridge rows.
- 16,641 duplicate track IDs, 894 exact duplicate rows, 0 conflicting IDs, 0 quarantined rows.
- Splits: dev 71,813 / fit 68,813 / eval 3,000 / holdout 17,928.
- License pending human verification; redistribution_allowed=false.

## Model and evidence
- Frozen evaluation model: K-Means, F9, RobustScaler, no PCA, K=2, seed 42.
  Run 20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42.
- Development metrics: silhouette 0.5956, DB 0.6505, CH 3240.62.
- One-time holdout: silhouette 0.5902, DB 0.6593, CH 2964.07 (deltas small).
- Stability: seed ARI 0.99969, bootstrap ARI 0.99907, pairwise 0.99918.
- Final refit: 89,741 tracks; cluster 0 72,211 (80.47%, vocal/non-instrumental),
  cluster 1 17,530 (19.53%, instrumental/ambient).
- Genre post-hoc only: ARI 0.00407, NMI 0.08169 on 73,442 single-genre tracks.
- Ward diagnostic: ARI 0.9692 vs selected model on the same 2,000 sample; sample-only.

## Verification
- artifacts/INDEX.csv: 54 completed runs, 0 failed (48 candidate, 3 final_refit, 1 stability,
  1 holdout, 1 ward).
- 7 verification logs, 389 checks passed; key runs verified 60/56/66/56/25 checks.
- pytest 42 passed; ruff clean; CLI smoke passed.
- Defense deck: deliverables/BTL11_defense.pptx, 12 slides, 637,227 bytes,
  SHA-256 1dfa3038015dc8079137114c13cacf404da1d6df4daab1a941cff02a96cbadc8;
  geometry audit clean; visual rehearsal pending.

## Human gates
- License/redistribution review before public release of the raw CSV.
- Git remote/push/deploy not performed.
- Official grading rubric not provided; no score claim.
- Defense timing/rehearsal remains a human task.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
