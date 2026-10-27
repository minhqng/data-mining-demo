---
title: BTL11 completion verification pass
date: 2026-10-27
summary: "Closed remaining technical gates: clean-clone install, 63/63 verify, deck PDF render; docs reconciled"
---

# BTL11 completion verification pass

## Context

Continuation of the 2026-10-27 final audit and submission-readiness correction. The two remaining
technical gates listed as `NOT VERIFIED` were the post-audit clean-clone install and a rendered
defense deck; a full re-verification of the current workspace was also due.

## What was executed

- `.venv/bin/pytest -q` → **50 passed**; `.venv/bin/ruff check src tests scripts` → clean.
- `btl11 verify` for every run folder → **63/63 PASS, 0 failures**; logs refreshed in
  `artifacts/verification/`.
- Clean-clone install in `/tmp/btl11_cleanclone` (copy excluding `data/`, `artifacts/`,
  `deliverables/`, `.venv`, caches and egg-info): fresh venv, `pip install -e '.[dev]'` → success,
  `pytest -q` → **50 passed**, `ruff` clean, `btl11 --help` OK.
- Deck artifacts verified:
  - `deliverables/BTL11_defense.pptx`: 12 slides, 637,322 bytes, SHA-256
    `e880f7eac1222b60c48294b7a05bc48098a1cafaa949a3046531a3ed9b185ba7`.
  - `deliverables/BTL11_defense.pdf`: 12 pages, LibreOffice 26.8 render, SHA-256
    `9231daa77b105805e5b7d6d5080a374c442cdcc37926388636c6716cd5e55dcb`; extracted text matches
    the PPTX, including the feature-dominance caveat on the conclusion slide.

## Documentation reconciled

`AGENTS.md`, `README.md`, `docs/FINAL_READINESS.md`, `docs/FINAL_CODE_AUDIT.md` (new §5 closure
note; former §5 renumbered to §6), `docs/IMPLEMENTATION_PLAN.md`, `docs/EXPERIMENTS.md`,
`docs/DECISIONS.md`, `docs/REPORT.md` and the `docs/BTL11_SSOT.md` changelog now record the
clean-clone pass and the PDF render, and no longer list them as open technical gates.

## Not changed

No dataset, feature allowlist, split, metric definition, model, frozen run, holdout result or
run artifact was changed. The holdout was not reopened.

## Remaining human gates

- License/redistribution review (Kaggle ODbL vs HF BSD vs Spotify provenance); raw CSV private.
- Git commit/push/deploy decision; remote CI has not run.
- Visual/timing rehearsal of the rendered defense deck.
- Official course rubric was not provided.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
