---
title: BTL11 defense deck rebuild
date: 2026-10-27
summary: Rebuilt missing 12-slide defense deck from report tables; verified artifacts and recorded hash
---

# BTL11 defense deck rebuild

# BTL11 defense deck rebuild

## Context
`deliverables/BTL11_defense.pptx` was documented as complete but was missing from the workspace.

## Change
- Rebuilt the 12-slide Vietnamese defense deck from `artifacts/final-report-tables/REPORT_TABLES.md`
  and the key artifact figures (EDA correlation, K-Means metrics, GMM BIC/AIC, stability, final profiles).
- No new empirical numbers; all values trace to existing run artifacts.
- Clarified that K-Means uses Robust scaling while the GMM baseline uses Standard scaling, so
  silhouette is not compared directly across scaler spaces.
- Recorded the deck size/hash in `docs/EXPERIMENTS.md` and `docs/IMPLEMENTATION_PLAN.md`.

## Evidence
- Deck: 12 slides, 637,227 bytes, SHA-256
  `1dfa3038015dc8079137114c13cacf404da1d6df4daab1a941cff02a96cbadc8`.
- Geometry audit: 12/12 pages clean; PPTX zip integrity OK.
- `pytest -q`: 42 passed; `ruff check src tests`: clean.
- Key run verification: 60, 56, 66, 56, 25 checks passed for frozen/holdout/final/stability/Ward.
- `artifacts/INDEX.csv`: 54 completed runs, 0 failed.

## Remaining human gates
- License/redistribution review before any public release of the raw CSV.
- Git remote/push/deploy not performed.
- Visual rendering and defense timing rehearsal remain human tasks (no Office renderer in this runtime).

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
