# BTL11 — Machine-generated report tables

- Generated: `2026-10-27T07:08:46Z`
- Selected final refit run: `20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2`
- Evaluation model run: `20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`
- Holdout run: `20261027T032820Z_f7434a8c_b202fa49_holdout_20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_rob`
- Dataset SHA-256: `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd`
- License status: `pending_human_verification` (redistribution allowed: `False`)

> Every empirical value below comes from the run artifacts in this repository. Sections without a completed run are marked NOT RUN.

## 1. Dataset

| dataset_sha256 | source | kaggle_dataset_id | raw_bytes | raw_rows | unique_tracks | n_genres | tracks_with_multiple_genres | genre_bridge_rows | n_exact_duplicate_rows | n_duplicate_track_ids | n_conflicting_track_ids | n_quarantined_rows | license_review_status | redistribution_allowed | local_analysis_authorized | kaggle_page_license | huggingface_mirror_license | license_metadata_discrepancy | dev_tracks | fit_tracks | eval_tracks | holdout_tracks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd | Kaggle — Spotify Tracks Dataset | maharshipandya/-spotify-tracks-dataset | 20118244 | 114000 | 89741 | 114 | 16299 | 113550 | 894 | 16641 | 0 | 0 | pending_human_verification | False | True | ODbL 1.0 (Open Data Commons Open Database License 1.0), copyright original authors | BSD | True | 71813 | 68813 | 3000 | 17928 |

## 2. Baseline K-Means (F9/Standard/No-PCA unless shown)

| run_id | k | feature_space_id | n_fit | n_eval | silhouette | davies_bouldin | calinski_harabasz | inertia_fit | inertia_eval | min_cluster_share_fit | empty_clusters_fit | converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261027T031534Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k2_s42 | 2 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.257588 | 1.58475 | 800.618 | 484912 | 20983 | 0.248906 | 0 | True |
| 20261027T031536Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k3_s42 | 3 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.165348 | 1.87735 | 651.662 | 430567 | 18533.7 | 0.217343 | 0 | True |
| 20261027T031539Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k4_s42 | 4 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.174162 | 1.68773 | 588.533 | 387559 | 16737.1 | 0.0775726 | 0 | True |
| 20261027T031543Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k5_s42 | 5 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.177621 | 1.50518 | 563.677 | 349802 | 15187.7 | 0.0119309 | 0 | True |
| 20261027T031546Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k6_s42 | 6 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.189578 | 1.43414 | 556.749 | 318000 | 13801.1 | 0.0113641 | 0 | True |
| 20261027T031549Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k7_s42 | 7 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.19515 | 1.3779 | 561.267 | 289373 | 12533.2 | 0.011306 | 0 | True |
| 20261027T031552Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k8_s42 | 8 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.174325 | 1.39507 | 529.775 | 273579 | 11893.7 | 0.0112188 | 0 | True |
| 20261027T031557Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k9_s42 | 9 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.180196 | 1.37008 | 517.052 | 259282 | 11178.8 | 0.0101434 | 0 | True |
| 20261027T031601Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k10_s42 | 10 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.16958 | 1.42494 | 495.141 | 248708 | 10698 | 0.0101289 | 0 | True |
| 20261027T031606Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k11_s42 | 11 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.173354 | 1.43209 | 479.862 | 239396 | 10227.8 | 0.0101289 | 0 | True |
| 20261027T031610Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k12_s42 | 12 | f9\|standard\|none\|whiten=False | 68813 | 3000 | 0.173984 | 1.42832 | 459.789 | 231867 | 9907.34 | 0.0101289 | 0 | True |

## 3. Baseline GMM (BIC/AIC only comparable within one representation)

| run_id | k | covariance_type | feature_space_id | cohort_id | bic | bic_rank_within_space | aic | silhouette | davies_bouldin | calinski_harabasz | converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261027T031628Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k2_s42 | 2 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 1.20326e+06 | 21 | 1.20293e+06 | 0.113641 | 3.05703 | 278.041 | True |
| 20261027T031629Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k2_s42 | 2 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 1.39716e+06 | 22 | 1.39617e+06 | 0.0960907 | 3.16287 | 248.823 | True |
| 20261027T031632Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k3_s42 | 3 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 1.10742e+06 | 20 | 1.10691e+06 | 0.0748147 | 2.41216 | 318.463 | True |
| 20261027T031634Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k3_s42 | 3 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 955989 | 17 | 954490 | 0.0647382 | 2.4742 | 252.818 | True |
| 20261027T031641Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k4_s42 | 4 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 1.0065e+06 | 19 | 1.00581e+06 | 0.0539093 | 2.58131 | 275.172 | True |
| 20261027T031644Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k4_s42 | 4 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 853058 | 14 | 851057 | 0.0546923 | 2.58276 | 233.356 | True |
| 20261027T031655Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k5_s42 | 5 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 957095 | 18 | 956236 | 0.0555053 | 2.61519 | 260.487 | True |
| 20261027T031658Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k5_s42 | 5 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 806946 | 11 | 804442 | 0.0391795 | 3.14004 | 195.818 | True |
| 20261027T031708Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k6_s42 | 6 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 895482 | 16 | 894450 | 0.0494145 | 2.55444 | 287.2 | True |
| 20261027T031714Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k6_s42 | 6 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 769862 | 8 | 766855 | 0.000404938 | 3.11954 | 188.852 | True |
| 20261027T031730Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k7_s42 | 7 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 872916 | 15 | 871710 | 0.0233904 | 2.50727 | 264.396 | True |
| 20261027T031737Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k7_s42 | 7 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 750570 | 6 | 747060 | -0.0293694 | 3.44875 | 164.274 | True |
| 20261027T031757Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k8_s42 | 8 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 837365 | 13 | 835985 | 0.0268178 | 2.80103 | 242.039 | True |
| 20261027T031802Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k8_s42 | 8 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 737685 | 5 | 733673 | -0.011815 | 4.54642 | 169.448 | True |
| 20261027T031823Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k9_s42 | 9 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 819359 | 12 | 817805 | 0.0300893 | 2.79308 | 233.886 | True |
| 20261027T031830Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k9_s42 | 9 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 712351 | 4 | 707836 | -0.00612039 | 4.28948 | 182.114 | True |
| 20261027T031850Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k10_s42 | 10 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 806601 | 10 | 804874 | 0.0129712 | 2.83226 | 208.357 | True |
| 20261027T031856Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k10_s42 | 10 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 701085 | 3 | 696068 | -0.0147444 | 3.64363 | 175.388 | True |
| 20261027T031913Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k11_s42 | 11 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 786099 | 9 | 784198 | -9.21441e-05 | 3.23769 | 198.594 | True |
| 20261027T031920Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k11_s42 | 11 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 699425 | 2 | 693905 | -0.0359523 | 4.21337 | 142.732 | True |
| 20261027T031941Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-diag_k12_s42 | 12 | diag | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 767035 | 7 | 764960 | 0.007661 | 4.03228 | 190.74 | True |
| 20261027T031949Z_2e91fc78_b202fa49_gmm_f9_standard_nopca_cov-full_k12_s42 | 12 | full | f9\|standard\|none\|whiten=False | 1781cbd281f62c1d | 678842 | 1 | 672820 | -0.0234069 | 3.59622 | 150.397 | True |

## 4. Ablations

### scaler

| variant | algorithm | k | covariance_type | feature_space_id | silhouette | davies_bouldin | calinski_harabasz | bic | min_cluster_share_fit | converged | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none_unscaled_E00_diagnostic | kmeans | 2 |  | f9\|none\|none\|whiten=False | 0.516732 | 0.65015 | 4883.66 |  | 0.493032 | True | 20261027T032118Z_07fbb11b_b202fa49_kmeans_f9_none_nopca_k2_s42 |
| robust | kmeans | 2 |  | f9\|robust\|none\|whiten=False | 0.595621 | 0.650472 | 3240.62 |  | 0.195457 | True | 20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42 |
| standard | kmeans | 2 |  | f9\|standard\|none\|whiten=False | 0.257588 | 1.58475 | 800.618 |  | 0.248906 | True | 20261027T031534Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k2_s42 |
| none_unscaled_E00_diagnostic | kmeans | 4 |  | f9\|none\|none\|whiten=False | 0.472741 | 0.635263 | 6512.92 |  | 0.13666 | True | 20261027T032121Z_07fbb11b_b202fa49_kmeans_f9_none_nopca_k4_s42 |
| robust | kmeans | 4 |  | f9\|robust\|none\|whiten=False | 0.488386 | 0.729702 | 2631.16 |  | 0.0102161 | True | 20261027T032112Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k4_s42 |
| standard | kmeans | 4 |  | f9\|standard\|none\|whiten=False | 0.174162 | 1.68773 | 588.533 |  | 0.0775726 | True | 20261027T031539Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k4_s42 |
| none_unscaled_E00_diagnostic | kmeans | 6 |  | f9\|none\|none\|whiten=False | 0.431443 | 0.715019 | 6615.57 |  | 0.108802 | True | 20261027T032124Z_07fbb11b_b202fa49_kmeans_f9_none_nopca_k6_s42 |
| robust | kmeans | 6 |  | f9\|robust\|none\|whiten=False | 0.337379 | 1.07461 | 2128.53 |  | 0.0101434 | True | 20261027T032115Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k6_s42 |
| standard | kmeans | 6 |  | f9\|standard\|none\|whiten=False | 0.189578 | 1.43414 | 556.749 |  | 0.0113641 | True | 20261027T031546Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k6_s42 |

### feature_set

| variant | algorithm | k | covariance_type | feature_space_id | silhouette | davies_bouldin | calinski_harabasz | bic | min_cluster_share_fit | converged | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f5 | kmeans | 2 |  | f5\|standard\|none\|whiten=False | 0.297997 | 1.37 | 1226.4 |  | 0.302675 | True | 20261027T032131Z_07fbb11b_b202fa49_kmeans_f5_standard_nopca_k2_s42 |
| f9 | kmeans | 2 |  | f9\|standard\|none\|whiten=False | 0.257588 | 1.58475 | 800.618 |  | 0.248906 | True | 20261027T031534Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k2_s42 |
| f5 | kmeans | 4 |  | f5\|standard\|none\|whiten=False | 0.225805 | 1.40695 | 982.068 |  | 0.181056 | True | 20261027T032134Z_07fbb11b_b202fa49_kmeans_f5_standard_nopca_k4_s42 |
| f9 | kmeans | 4 |  | f9\|standard\|none\|whiten=False | 0.174162 | 1.68773 | 588.533 |  | 0.0775726 | True | 20261027T031539Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k4_s42 |
| f5 | kmeans | 6 |  | f5\|standard\|none\|whiten=False | 0.198115 | 1.41905 | 858.31 |  | 0.118233 | True | 20261027T032137Z_07fbb11b_b202fa49_kmeans_f5_standard_nopca_k6_s42 |
| f9 | kmeans | 6 |  | f9\|standard\|none\|whiten=False | 0.189578 | 1.43414 | 556.749 |  | 0.0113641 | True | 20261027T031546Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k6_s42 |

### pca

| variant | algorithm | k | covariance_type | feature_space_id | silhouette | davies_bouldin | calinski_harabasz | bic | min_cluster_share_fit | converged | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_pca | kmeans | 2 |  | f9\|standard\|none\|whiten=False | 0.257588 | 1.58475 | 800.618 |  | 0.248906 | True | 20261027T031534Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k2_s42 |
| pca90 | kmeans | 2 |  | f9\|standard\|pca90\|whiten=False | 0.267346 | 1.51562 | 856.964 |  | 0.248456 | True | 20261027T032143Z_07fbb11b_b202fa49_kmeans_f9_standard_pca90_k2_s42 |
| pca95 | kmeans | 2 |  | f9\|standard\|pca95\|whiten=False | 0.260212 | 1.56828 | 816.013 |  | 0.249648 | True | 20261027T032153Z_07fbb11b_b202fa49_kmeans_f9_standard_pca95_k2_s42 |
| no_pca | kmeans | 4 |  | f9\|standard\|none\|whiten=False | 0.174162 | 1.68773 | 588.533 |  | 0.0775726 | True | 20261027T031539Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k4_s42 |
| pca90 | kmeans | 4 |  | f9\|standard\|pca90\|whiten=False | 0.188447 | 1.61496 | 641.943 |  | 0.07734 | True | 20261027T032146Z_07fbb11b_b202fa49_kmeans_f9_standard_pca90_k4_s42 |
| pca95 | kmeans | 4 |  | f9\|standard\|pca95\|whiten=False | 0.1777 | 1.66852 | 602.675 |  | 0.0776888 | True | 20261027T032156Z_07fbb11b_b202fa49_kmeans_f9_standard_pca95_k4_s42 |
| no_pca | kmeans | 6 |  | f9\|standard\|none\|whiten=False | 0.189578 | 1.43414 | 556.749 |  | 0.0113641 | True | 20261027T031546Z_2e91fc78_b202fa49_kmeans_f9_standard_nopca_k6_s42 |
| pca90 | kmeans | 6 |  | f9\|standard\|pca90\|whiten=False | 0.203291 | 1.36126 | 614.504 |  | 0.0114368 | True | 20261027T032150Z_07fbb11b_b202fa49_kmeans_f9_standard_pca90_k6_s42 |
| pca95 | kmeans | 6 |  | f9\|standard\|pca95\|whiten=False | 0.193779 | 1.41415 | 572.935 |  | 0.0113641 | True | 20261027T032200Z_07fbb11b_b202fa49_kmeans_f9_standard_pca95_k6_s42 |

> Cross-representation metrics are contextual; only BIC ranks inside the same feature-space identifier.
> E00 `none_unscaled` is a scale diagnostic, never the recommended model.

## 5. Stability (same fixed reference IDs; ARI permutation invariant)

| finalist | algorithm | k | covariance_type | n_seed_comparisons | seed_ari_mean | seed_ari_std | bootstrap_n | bootstrap_ari_mean | bootstrap_ari_std | pairwise_ari_mean | n_reference_ids |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kmeans-k2-f9-standard-none-n/a-s42 | kmeans | 2 |  | 5 | 1 | 0 | 5 | 0.991433 | 0.00692538 | 0.992008 | 3000 |
| kmeans-k4-f9-standard-none-n/a-s42 | kmeans | 4 |  | 5 | 0.996929 | 0.00237073 | 5 | 0.980867 | 0.0050681 | 0.982174 | 3000 |
| kmeans-k6-f9-standard-none-n/a-s42 | kmeans | 6 |  | 5 | 0.933423 | 0.132662 | 5 | 0.916413 | 0.131029 | 0.869597 | 3000 |
| kmeans-k2-f9-robust-none-n/a-s42 | kmeans | 2 |  | 5 | 0.999692 | 0.000616998 | 5 | 0.999075 | 0.000755665 | 0.999177 | 3000 |
| kmeans-k6-f9-standard-0.90-n/a-s42 | kmeans | 6 |  | 5 | 0.998739 | 0.00076836 | 5 | 0.982706 | 0.00598262 | 0.985122 | 3000 |
| gmm-k4-f9-standard-none-full-s42 | gmm | 4 | full | 5 | 0.97855 | 0.02034 | 5 | 0.936991 | 0.0975521 | 0.925512 | 3000 |

## 6. One-time holdout check (frozen evaluation model)

| metric | value | direction | space | n_eval | reason |
| --- | --- | --- | --- | --- | --- |
| calinski_harabasz | 2964.07 | maximize | f9\|robust\|none\|whiten=False | 3000 |  |
| calinski_harabasz_delta_vs_dev | -276.545 | context | delta: holdout minus development (same representation) | 3000 |  |
| davies_bouldin | 0.659256 | minimize | f9\|robust\|none\|whiten=False | 3000 |  |
| davies_bouldin_delta_vs_dev | 0.00878468 | context | delta: holdout minus development (same representation) | 3000 |  |
| silhouette | 0.590222 | maximize | f9\|robust\|none\|whiten=False | 3000 |  |
| silhouette_delta_vs_dev | -0.00539965 | context | delta: holdout fixed cohort minus development fixed cohort (same representation, different tracks) | 3000 |  |

## 7. Final refit profiles (raw units; acoustic hypotheses, not genres)

| cluster | n | share | acousticness_median | danceability_median | energy_median | instrumentalness_median | speechiness_median | valence_median | tempo_median | interpretation_hypothesis | top_positive_features | top_negative_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 72211 | 0.80466 | 0.192 | 0.584 | 0.688 | 5.33e-06 | 0.0492 | 0.505 | 122.013 | Vocal/non-instrumental, năng lượng trung bình–cao (giả thuyết âm học, không phải genre) | loudness (+0.20), valence (+0.15), danceability (+0.09) | instrumentalness (-0.48), acousticness (-0.04), tempo (+0.02) |
| 1 | 17530 | 0.19534 | 0.149 | 0.52 | 0.586 | 0.858 | 0.0482 | 0.229 | 122.013 | Instrumental/ambient (giả thuyết âm học, không phải genre) | instrumentalness (+1.96), acousticness (+0.17), tempo (-0.10) | loudness (-0.82), valence (-0.63), danceability (-0.36) |

## 8. Post-hoc multi-label genre evidence (top 10 per cluster)

| cluster | genre | n_tracks_with_genre | cluster_n | share_of_cluster_tracks |
| --- | --- | --- | --- | --- |
| 0 | pagode | 1000 | 72211 | 0.0138483 |
| 0 | reggae | 1000 | 72211 | 0.0138483 |
| 0 | reggaeton | 1000 | 72211 | 0.0138483 |
| 0 | sertanejo | 1000 | 72211 | 0.0138483 |
| 0 | country | 998 | 72211 | 0.0138206 |
| 0 | forro | 998 | 72211 | 0.0138206 |
| 0 | mandopop | 998 | 72211 | 0.0138206 |
| 0 | r-n-b | 997 | 72211 | 0.0138068 |
| 0 | comedy | 996 | 72211 | 0.0137929 |
| 0 | gospel | 996 | 72211 | 0.0137929 |
| 1 | study | 911 | 17530 | 0.0519681 |
| 1 | minimal-techno | 882 | 17530 | 0.0503137 |
| 1 | sleep | 816 | 17530 | 0.0465488 |
| 1 | new-age | 808 | 17530 | 0.0460924 |
| 1 | detroit-techno | 801 | 17530 | 0.0456931 |
| 1 | idm | 778 | 17530 | 0.0443811 |
| 1 | ambient | 765 | 17530 | 0.0436395 |
| 1 | iranian | 680 | 17530 | 0.0387906 |
| 1 | techno | 641 | 17530 | 0.0365659 |
| 1 | classical | 638 | 17530 | 0.0363948 |

> Denominator = unique tracks per cluster. Shares can sum >100% because tracks can carry multiple genres. Genre was never a training/tuning input.

{
  "adjusted_rand_vs_genre": {
    "reason": null,
    "value": 0.004067215673730017
  },
  "caveat": "Genre taxonomy does not define acoustic personas. Low/high agreement is diagnostic, not a target and not proof of cluster quality.",
  "cluster_n": 2,
  "cohort_rule": "Single-genre-only tracks (exactly one unique genre in track_genres). The cohort is declared before computing ARI/NMI; no labels were used for tuning.",
  "genre_n_single_cohort": 112,
  "n_assigned": 89741,
  "n_omitted_not_single_label": 16299,
  "n_single_genre_eval": 73442,
  "normalized_mutual_info_vs_genre": {
    "reason": null,
    "value": 0.08168542875027116
  },
  "run_id": "20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2",
  "schema": "btl11.genre_posthoc.v1",
  "scope": "all_unique_tracks"
}

## 9. Final refit metrics (in-sample descriptive verification sample)

| metric | value | direction | space | n_eval | reason |
| --- | --- | --- | --- | --- | --- |
| calinski_harabasz | 2912.17 | maximize | f9\|robust\|none\|whiten=False | 3000 |  |
| davies_bouldin | 0.673045 | minimize | f9\|robust\|none\|whiten=False | 3000 |  |
| silhouette | 0.584193 | maximize | f9\|robust\|none\|whiten=False | 3000 |  |
| silhouette_descriptive | 0.584193 | context | f9\|robust\|none\|whiten=False | 3000 |  |

## 10. Figures

| figure | scope | run_id | caption |
| --- | --- | --- | --- |
| artifacts/runs/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/figures/cluster_sizes_all.png | final_refit | 20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2 | cluster sizes all |
| artifacts/runs/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/figures/genre_cluster_heatmap.png | final_refit | 20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2 | genre cluster heatmap |
| artifacts/runs/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/figures/profile_heatmap.png | final_refit | 20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2 | profile heatmap |
| artifacts/posthoc/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/figures/pca2_scatter.png | posthoc_visual | 20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2 | pca2 scatter |
| artifacts/posthoc/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/figures/tsne2d.png | posthoc_visual | 20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2 | tsne2d |
| artifacts/comparisons/baseline_f9_standard/20261027T070434Z/elbow.png | baseline_comparison | multiple | elbow |
| artifacts/comparisons/baseline_f9_standard/20261027T070434Z/gmm_bic_aic.png | baseline_comparison | multiple | gmm bic aic |
| artifacts/comparisons/baseline_f9_standard/20261027T070434Z/kmeans_metrics.png | baseline_comparison | multiple | kmeans metrics |
| artifacts/ablations/scaler/20261027T032128Z/metrics_by_variant.png | ablation_scaler | multiple | metrics by variant |
| artifacts/ablations/feature_set/20261027T032140Z/metrics_by_variant.png | ablation_feature_set | multiple | metrics by variant |
| artifacts/ablations/pca/20261027T032204Z/metrics_by_variant.png | ablation_pca | multiple | metrics by variant |
| artifacts/runs/20261027T032241Z_ea0570dc_b202fa49_stability/figures/stability.png | stability | 20261027T032241Z_ea0570dc_b202fa49_stability | stability |
