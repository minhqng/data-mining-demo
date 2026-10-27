# AGENTS.md — BTL11

## Canonical context

1. `docs/BTL11_SSOT.md` is the normative specification. Read it before changing dataset, features,
   splits, metrics, artifact schemas or protocols.
2. `docs/DECISIONS.md` records the locked decisions and the evidence-based final selection.
3. `docs/REPORT.md` contains only claims traceable to `artifacts/runs/<run_id>/` or stated theory.
4. `docs/FINAL_CODE_AUDIT.md`, `docs/ROBUSTNESS_ANALYSIS.md` and `docs/FINAL_READINESS.md` record
   the 2026-10-27 audit, feature-dominance evidence and remaining gates. Read them before changing
   model-selection wording.

## Hard constraints

- Sole research dataset: exact Kaggle Spotify Tracks Dataset
  (`maharshipandya/-spotify-tracks-dataset`) at `data/raw/dataset.csv`.
  Never substitute GTZAN/FMA/mirrors/demo data/Spotify API.
- Unit of analysis is unique `track_id`. Repeated IDs with identical audio features collapse to
  one track; all genres go to `track_genres.parquet`. Conflicting vectors fail closed.
- `track_genre`, IDs and metadata never enter X or a fitted transformer.
- Fit scaler/PCA/clusterers only on `fit` IDs (development minus fixed eval cohort). Holdout is
  used once, after freezing; final full refit is a distinct model identity.
- Never invent K, metrics, profiles or run IDs. Invalid metrics are `null` with a reason.
- Never compare raw BIC across different feature/scaler/PCA spaces.
- Do not describe the selected RobustScaler K=2 model as balanced multivariate persona discovery:
  `instrumentalness` accounts for 97.29% of between-cluster separation and a single-feature K=2
  reproduces the partition at ARI 0.9892. Describe it as instrumental vs vocal/non-instrumental
  segmentation (D-16).
- Do not publish `data/raw/dataset.csv`; license status is pending human verification.
- Do not commit/push, deploy, or change access rights without explicit human authorization.

## Commands

```bash
pytest -q
ruff check src tests scripts
btl11 --help
btl11 audit --input data/raw/dataset.csv --config configs/base.yaml
btl11 prepare --config configs/base.yaml
btl11 sweep --algorithm kmeans --config configs/base.yaml --dry-run
btl11 verify --run-id <RUN_ID> --config configs/base.yaml
btl11 robustness --evaluation-run <FROZEN_RUN_ID> --config configs/base.yaml
.venv/bin/python scripts/run_controlled_gmm.py --config configs/base.yaml
```

Full staged commands are in `README.md` and `docs/EXPERIMENTS.md`.

## Current state

- 50 synthetic tests pass; lint clean; CI workflow is fixture-only.
- 63 completed run folders; 0 failed experiment fits; 63/63 verify PASS.
- Frozen evaluation model: `20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`.
- One-time holdout: `20261027T032820Z_f7434a8c_b202fa49_holdout_20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_rob`.
- Final refit: `20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2`.
- Corrected Ward (frozen transformer): `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`
  (ARI 0.883442); historical sample-refit Ward ARI 0.969205 is retained but superseded.
- Robustness/dominance: `artifacts/robustness/20261027T070249Z/`.
- Controlled GMM comparison: `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/`.
- Selected configuration is already in `configs/base.yaml`; do not retune on holdout.
- Course brief and four instructor decks were cross-checked (D-15); defense deck is
  `deliverables/BTL11_defense.pptx` (12 slides, one dominance caveat added in the audit); a
  12-page LibreOffice render is `deliverables/BTL11_defense.pdf`.
- Clean-clone install re-verified 2026-10-27: fresh venv from a clean copy, `pip install -e '.[dev]'`,
  50 tests pass, ruff clean, CLI OK.

## Known gates

- Redistribution/publication of raw data is blocked pending human license review (Kaggle ODbL
  vs Hugging Face BSD discrepancy; Spotify Web API provenance).
- No git commit/remote was created in this runtime; remote CI did not run.
- The rendered defense deck still needs a human visual/timing rehearsal.
- `gmm.reg_covar_fallback` is validated but unused; all current GMM fits converged.

Report changes with run IDs and exact artifact paths. Do not call the repository verified without
actually running the checks.
