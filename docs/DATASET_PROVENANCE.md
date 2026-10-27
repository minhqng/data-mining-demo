# Dataset provenance and compliance record

**Recorded:** 2026-10-27
**Project:** BTL11 — Music Genre & Audio Track Segmentation
**Status:** local academic analysis executed; **public redistribution is NOT cleared**.

## 1. Exact identity

| Field | Value |
|---|---|
| Dataset | Kaggle — Spotify Tracks Dataset |
| Kaggle id | `maharshipandya/-spotify-tracks-dataset` |
| Canonical page | <https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset> |
| Local raw file | `data/raw/dataset.csv` |
| Raw SHA-256 | `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd` |
| Raw bytes | 20,118,244 |
| File mtime (filesystem metadata) | 2022-10-22 (not proof of download time) |
| Authoritative download timestamp | **not recorded** by the supplier |
| Manifest | `data/raw/MANIFEST.json` |
| Audit artifact | `data/processed/audit.json` |

The file was supplied by the project owner at the repository root and moved to `data/raw/dataset.csv`
without changing its bytes. No Kaggle CLI/API download, no Hugging Face mirror download, no Spotify
Web API call and no scraping was performed by the coding agent.

The dataset is the **only** empirical data source. GTZAN, FMA, Million Song Dataset, other Spotify
variants, Hugging Face mirrors, scraped tracks, demo CSVs and Spotify Web API are prohibited
substitutes. Tiny synthetic fixtures in `tests/` are test-only and are not research evidence.

## 2. Actual audit counts (computed from the bytes above)

| Quantity | Actual value |
|---|---:|
| Raw rows | 114,000 |
| Columns after dropping Kaggle export index (`Unnamed: 0`) | 20 |
| Unique `track_id` | 89,741 |
| Duplicate-ID rows | 40,900 |
| IDs appearing more than once | 16,641 |
| Max rows per ID | 9 |
| Tracks with >1 genre | 16,299 |
| Unique genres | 114 |
| Genre bridge rows `(track_id, genre)` | 113,550 |
| Exact duplicate rows after ignoring export index | 894 (450 redundant rows beyond first) |
| Conflicting audio vectors for the same ID | **0** |
| Missing/nonfinite audio feature cells | 0 |
| Invalid/blank `track_id` | 0 |
| Rows quarantined in prepare | 0 |
| `tempo == 0` flags | 157 rows = 157 unique tracks |
| `loudness > 0` flags | 90 rows = 68 unique tracks |
| Unit-interval feature out-of-range cells | 0 |

The Kaggle description/mirror mentions “125 genres”, but the actual artifact contains **114**
`track_genre` values. The report uses the computed value, never the web description.

The repeated IDs are multi-genre appearances of the same audio track: after deduplication there are
89,741 unique tracks. All repeated audio vectors agree exactly within the configured
`atol=1e-8, rtol=1e-6` tolerance, so no conflict resolution/averaging was needed. Genre incidence
is retained in `data/processed/track_genres.parquet`; no track is double-weighted by its genre count.

## 3. Splits and unit of analysis

- Primary unit: one unique `track_id`.
- `dev` + `holdout`: 71,813 + 17,928 = 89,741 unique tracks.
- Fixed selection cohort: 3,000 IDs inside dev; **excluded from fitting**.
- `fit`: 68,813 development IDs.
- Holdout evaluation cohort: 3,000 IDs inside holdout, used once after freeze.
- Splits are deterministic stable hashes of IDs, not random row splits.

## 4. License evidence and unresolved discrepancy

Observed on 2026-10-27:

1. The Kaggle dataset page metadata displays **ODbL 1.0** (Open Data Commons Open Database
   License 1.0), “Database: Open Database, Contents: © Original Authors”.
2. The Hugging Face mirror card displays **BSD** and says the data was collected and cleaned via
   **Spotify's Web API**.
3. Spotify Developer Policy/Terms impose restrictions on Spotify Platform/Content, including
   ML/AI use; this repository does not claim those terms are automatically satisfied by an
   independently supplied Kaggle snapshot.

The two public license assertions conflict, and the Spotify API provenance creates an additional
risk. Therefore:

```text
license_review_status          = pending_human_verification
redistribution_allowed         = false
local_analysis_authorized_by_owner = true
```

Local analysis was authorized by the project owner for the assigned university coursework. It is
not a redistribution license and not legal advice.

## 5. Human action required before public release

1. Verify the current Kaggle license field and the dataset's provenance in writing.
2. Determine whether Spotify Developer Terms/Policy restrict the intended ML/AI and publication use.
3. Check institutional policy for course submission and public repositories.
4. Record the decision here and in `data/raw/MANIFEST.json`; if the use is not permitted, switch to
   an approved alternative **only after instructor approval** (do not silently substitute data).
5. Keep `data/raw/dataset.csv` out of git. It is gitignored in this repository. Do not publish
   credentials.

Until those steps are completed, the repository status remains:

```text
CODE_COMPLETE / REAL_EXPERIMENTS_EXECUTED / PUBLIC_REDISTRIBUTION_BLOCKED
```

## 6. Sources checked

- [Kaggle dataset page](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- [Hugging Face mirror card](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- [Spotify Developer Policy](https://developer.spotify.com/policy)
- [Spotify developer update 2026-02-06](https://developer.spotify.com/blog/2026-02-06-update-on-developer-access-and-platform-security)
- Local `docs/BTL11_SSOT.md` sections on provenance and compliance

The observed license metadata is recorded in `MANIFEST.json` under
`license_metadata_observed`, with `discrepancy: true`.
