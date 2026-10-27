# BTL11 — Final readiness assessment

**Assessment date:** 2026-10-27  
**Completion pass:** 2026-10-27 — clean-clone install and the 12-page PDF deck render were verified.  
**Repository status:** internally complete and scientifically reproducible; **public
redistribution is blocked by the unresolved license gate**.  
**Evidence sources:** `docs/FINAL_CODE_AUDIT.md`, `docs/ROBUSTNESS_ANALYSIS.md`,
`docs/REPORT.md`, `docs/DECISIONS.md`, `artifacts/INDEX.csv`,
`artifacts/verification/`, `artifacts/robustness/20261027T070249Z/`.

**Status key:** `CONFIRMED` = executed and reproduced; `PARTIAL` = constrained by historical
schema or sample-only evidence; `BLOCKED` = human/external gate; `NOT VERIFIED` = not executed.

---

## 1. Verdict

| Question | Answer | Status |
|---|---|---|
| Are the frozen evaluation, holdout and final-refit numbers reproducible? | Yes — exact recomputation from raw data + stored bundles | `CONFIRMED` |
| Is K-Means / F9 / RobustScaler / no PCA / K=2 still defensible? | **Yes, narrowly**: as an instrumental vs vocal/non-instrumental segmentation. It is not defensible as balanced nine-feature multivariate persona discovery. | `CONFIRMED` with narrowed claim |
| Does the holdout need to be re-opened? | No. It was used once and remains untouched; the new work is post-hoc development-only. | `CONFIRMED` |
| Are there unfixed high-severity implementation defects? | No demonstrated defect invalidates the stored numbers. The high-impact issue is interpretation (feature dominance), not a numeric/code error. | `CONFIRMED` |
| Can the repository be called public-release ready? | No — license/redistribution remains pending human verification. | `BLOCKED` |
| Is the defense deck consistent? | Yes for numbers; a dominance caveat was added to slide 13. The deck renders to a 12-page PDF (`deliverables/BTL11_defense.pdf`); human visual/timing rehearsal remains. | `PARTIAL` |

---

## 2. Acceptance checklist

### Data and provenance

| Item | Result | Evidence |
|---|---|---|
| Exact Kaggle Spotify Tracks Dataset | `CONFIRMED` | raw SHA-256 `b202fa49…177bd`, 114,000 rows, 89,741 unique IDs |
| Unit = unique `track_id` | `CONFIRMED` | one row per ID; 113,550 `(track_id, genre)` bridge rows |
| Multi-genre handling | `CONFIRMED` | 16,299 multi-genre tracks; incidence heatmap; single-label ARI/NMI cohort |
| Conflicting audio vectors | `CONFIRMED` | 0 conflicts; fail-closed code + test |
| No dataset substitution | `CONFIRMED` | GTZAN/FMA/mirrors/Spotify API not used; fixtures are tests-only |
| License gate | `BLOCKED` | `pending_human_verification`; `redistribution_allowed=false` |

### Splits, leakage and preprocessing

| Item | Result | Evidence |
|---|---|---|
| ID-disjoint dev/holdout, fit/eval | `CONFIRMED` | 71,813 / 17,928; 68,813 / 3,000; assertions + tests |
| Fit-only scaler/PCA | `CONFIRMED` | `preprocess.py`; adversarial shift test |
| No genre/metadata in X | `CONFIRMED` | exact F5/F9 allowlist; feature-order tests; leak test |
| Fixed eval cohort | `CONFIRMED` | 3,000 IDs; all candidates scored on the same cohort |
| Final refit distinct identity | `CONFIRMED` | different n_fit/hash; final refit does not inherit holdout |

### Models and metrics

| Item | Result | Evidence |
|---|---|---|
| K-Means K=2..12, GMM diag/full K=2..12 | `CONFIRMED` | 33 baseline runs, 0 failures |
| Silhouette/DB/CH guarded | `CONFIRMED` | null + reason on invalid cluster counts |
| BIC within representation/cohort only | `CONFIRMED` | cohort-aware ranking; regenerated comparison |
| Inertia / GMM likelihood / parameter count | `CONFIRMED` | independent recomputation; d=9,K=12 227/659 |
| Stability ARI on identical IDs | `CONFIRMED` | 5 seeds + 5 bootstrap; verification recomputes means |
| Holdout one-time | `CONFIRMED` | guard by descriptor + dataset + split; no retune |
| Ward diagnostic | `PARTIAL` | sample-only; corrected frozen-transformer ARI 0.8834, not generalization |
| Controlled K-Means vs GMM | `CONFIRMED` | 8 exploratory rows, same IDs/features/scaler/K/seed; ARI computed |
| Feature dominance / LOO | `CONFIRMED` | `artifacts/robustness/20261027T070249Z/` |

### Engineering and evidence integrity

| Item | Result | Evidence |
|---|---|---|
| Tests | `CONFIRMED` | `pytest -q` → 50 passed (synthetic only) |
| Lint | `CONFIRMED` | `ruff check src tests scripts` → all passed |
| Artifact verification | `CONFIRMED` | 63/63 runs PASS, 0 failures |
| Independent metric recomputation | `CONFIRMED` | frozen/holdout/final exact matches |
| Tamper resistance | `CONFIRMED` | artifact hash + broken-chain tests; run dirs immutable |
| Security scan | `CONFIRMED` | no secrets/absolute paths; raw gitignored; `pip-audit` clean |
| Remote CI | `NOT VERIFIED` | no git remote/push authorized |
| Clean-clone install after audit | `CONFIRMED` | fresh venv from a clean copy, 2026-10-27: `pip install -e '.[dev]'`, 50 passed, ruff clean, CLI OK |
| Documentation consistency | `CONFIRMED` | audit/robustness/readiness + report/decisions updated |
| Defense deck | `PARTIAL` | 12 slides, caveat added, ZIP/text checked; 12-page LibreOffice PDF render exists; human timing/visual pass remains |

---

## 3. Exactly what the audit changed

Code:

- `evaluation.py`: empty-cluster min/max share now counts zero.
- `models.py` / `experiment.py`: GMM covariance default is explicit and consistent;
  `gmm.require_convergence=true` is enforced.
- `profiling.py`: candidate/holdout profiles use the development split as the reference;
  only the final refit uses all tracks.
- `experiment.py`: Ward uses the frozen selected model's transformer, not a sample-refit
  scaler.
- `verification.py`: freeze → holdout → refit chain, descriptor hashes, stability mean ARI,
  Ward sample membership, and a documented legacy schema fallback.
- `robustness.py` + `btl11 robustness`: feature attribution, LOO, scaler comparison and
  controlled GMM table.
- `config.py`: reject unimplemented imputation; require silhouette cohort settings to match
  the fixed eval cohort.
- `reporting.py`: reject a final refit/holdout/evaluation model from different dataset bytes.
- `scripts/run_controlled_gmm.py`: reproducible 8-run exploratory GMM comparison.

Artifacts:

- New runs: 8 controlled GMM candidates + 1 corrected Ward run; all verified.
- New robustness directory: `artifacts/robustness/20261027T070249Z/`.
- New cohort-aware baseline comparison:
  `artifacts/comparisons/baseline_f9_standard/20261027T070434Z/`.
- New controlled-GMM comparison:
  `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/`.
- Regenerated report tables and verification logs.
- Defense deck: one conclusion-slide caveat added (12 slides preserved). New SHA-256:
  `e880f7eac1222b60c48294b7a05bc48098a1cafaa949a3046531a3ed9b185ba7`
  (637,322 bytes).

No historical run, model, metric, holdout result or final-refit artifact was overwritten.

---

## 4. Scientific limitations that remain

1. **Single-feature dominance:** the selected K=2 RobustScaler model is effectively an
   `instrumentalness` threshold. 97.29% of between-cluster SS and 97.29% of the assignment
   margin come from that feature (at K=2 these are the same δ²/Σδ² quantity, not independent
   evidence); a one-feature K=2 partition reproduces it at ARI 0.9892, with **7 of 3,000 eval
   tracks differing** (contingency [[585, 4], [3, 2408]]); removing it changes the partition
   (ARI −0.0498). The model is valid for instrumental/vocal segmentation, not for balanced
   multivariate persona claims.
2. **Representation dependence:** StandardScaler produces a different, more multivariate
   partition (energy/acousticness/loudness), and the two partitions agree at only ARI 0.1350.
   Silhouette/DB/CH are not comparable across scaler spaces.
3. **K=2 coarseness:** speech-heavy, live, dance and other subtypes are merged in the final
   model. K=6 is richer but less stable (bootstrap ARI 0.9164) and has a ~1.1% cluster.
4. **GMM is not a substitute segmentation:** controlled GMM hard partitions disagree with
   K-Means (ARI −0.0201 to 0.3063) even though GMM posteriors are confident. BIC ranks density
   fit within a space; it does not select semantic clusters.
5. **Ward is sample-only:** corrected ARI 0.8834 on 2,000 fit IDs; no `.predict()` for new
   tracks; not a generalization metric.
6. **Exploratory robustness is post-hoc:** it uses development IDs only and is not a new
   confirmatory evaluation. No claim should present it as an untouched holdout result.
7. **Spotify descriptors are algorithmic:** `instrumentalness` and other features are not raw
   audio; no listening study was performed.
8. **Genre is not a target:** ARI 0.0041 / NMI 0.0817 on 73,442 single-genre tracks; this is a
   diagnostic, not a failure of the clustering.

---

## 5. Human-only gates

| Gate | Required action | Current status |
|---|---|---|
| License/redistribution | Verify Kaggle ODbL vs HF BSD vs Spotify terms and institutional policy; record decision in `DATASET_PROVENANCE.md` and `MANIFEST.json` | `BLOCKED` |
| Public release / git | Decide whether to commit/push/deploy; no remote action was performed | `BLOCKED` (not authorized) |
| Defense rehearsal | Review the rendered 12-page PDF (`deliverables/BTL11_defense.pdf`) and rehearse timing; confirm the added caveat fits visually | `NOT VERIFIED` |
| Course rubric | No official rubric was provided; no score/ROI claim is made | `BLOCKED` (external) |

The clean-clone install gate is closed: on 2026-10-27 a fresh venv was created from a clean copy
(excluding `data/`, `artifacts/` and `.venv`), `pip install -e '.[dev]'` succeeded, `pytest -q`
reported 50 passed, `ruff` was clean and `btl11 --help` worked. Remote CI remains `NOT VERIFIED`.
The deck PDF render is `deliverables/BTL11_defense.pdf` (12 pages, LibreOffice 26.8, SHA-256
`9231daa7…5dcb`); its text matches the PPTX, including the dominance caveat.

---

## 6. Submission recommendation

**Internally ready for academic submission and defense, with a narrowed claim.** The pipeline,
artifacts and verification chain are complete and reproducible. The report must state:

> The selected K-Means / RobustScaler / K=2 model discovers a stable instrumental vs
> vocal/non-instrumental split driven overwhelmingly by `instrumentalness`; it is not a
> balanced nine-feature multivariate segmentation. StandardScaler yields a different, more
> multivariate partition. The holdout numbers are valid for the frozen model but were not
> re-used for the new analyses.

**Not ready for public data/code redistribution** until the license gate is cleared by a
responsible human.
