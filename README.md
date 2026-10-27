# BTL11 — Music Genre & Audio Track Segmentation

Repository nghiên cứu phân cụm không giám sát cho đề tài **Music Genre & Audio Track Segmentation**
(môn Data Mining). Pipeline chạy trên **đúng snapshot Kaggle Spotify Tracks Dataset** được chỉ định,
với unit of analysis là **một `track_id` duy nhất**, đánh giá trên cohort cố định, kiểm tra độ ổn định,
holdout một lần, refit toàn bộ và sinh báo cáo từ artifact thực tế.

**Trạng thái hiện tại (2026-10-27):**

| Hạng mục | Trạng thái |
|---|---|
| Code + package + CLI + tests + CI workflow | **PASS** (50 tests pass, `ruff` sạch) |
| Dataset thật + audit + dedup + splits | **PASS** |
| K-Means / GMM sweeps, ablations, stability | **PASS** (0 fit lỗi) |
| One-time holdout của frozen model | **PASS** |
| Final refit + profiles + PCA/t-SNE + genre post-hoc | **PASS** |
| Ward + controlled K-Means/GMM + feature-dominance/LOO robustness | **PASS** (audit 2026-10-27) |
| Verify artifact độc lập | **PASS** (63/63 run) |
| Bộ slide bảo vệ 12 slide | **PASS** — caveat feature-dominance đã thêm; PDF render 12 trang đã có; rehearsal còn lại |
| Public redistribution raw Spotify snapshot | **BLOCKED — cần con người xác minh license** |
| Git remote / deploy / push | **KHÔNG thực hiện** (không được yêu cầu) |

Báo cáo đầy đủ: [`docs/REPORT.md`](docs/REPORT.md). Audit khoa học:
[`docs/FINAL_CODE_AUDIT.md`](docs/FINAL_CODE_AUDIT.md),
[`docs/ROBUSTNESS_ANALYSIS.md`](docs/ROBUSTNESS_ANALYSIS.md),
[`docs/FINAL_READINESS.md`](docs/FINAL_READINESS.md). Bảng máy sinh:
[`artifacts/final-report-tables/REPORT_TABLES.md`](artifacts/final-report-tables/REPORT_TABLES.md).

## 1. Dataset bắt buộc

- **Tên:** Kaggle — Spotify Tracks Dataset
- **Kaggle dataset id:** `maharshipandya/-spotify-tracks-dataset`
- **Trang chuẩn:** <https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset>
- **File dùng cho nghiên cứu:** `data/raw/dataset.csv` (do người dùng cung cấp, không tải từ mirror)
- **SHA-256:** `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd`
- **Bytes:** 20,118,244
- **Rows / unique tracks / genres thực đo:** 114,000 / 89,741 / 114

Không được thay thế hoặc trộn với GTZAN, FMA, Million Song Dataset, biến thể Spotify khác,
Hugging Face mirror, CSV demo, dữ liệu scrape hoặc Spotify Web API. Fixture tổng hợp rất nhỏ
trong `tests/` chỉ dùng để kiểm thử code, **không phải dữ liệu nghiên cứu**.

### License và quyền phân phối

Audit ngày 2026-10-27 ghi nhận metadata công khai **không đồng nhất**:

- Trang Kaggle hiển thị **ODbL 1.0** (Open Data Commons Open Database License);
- Dataset card Hugging Face mirror hiển thị **BSD**;
- Trang nguồn cho biết dữ liệu được thu thập qua **Spotify Web API**.

Vì vậy trạng thái trong `data/raw/MANIFEST.json` là `pending_human_verification`,
`redistribution_allowed: false`. Phân tích học thuật cục bộ được chủ dự án cho phép, nhưng
**không được public raw CSV, không deploy thương mại, không mặc định Spotify Platform/Content
được dùng cho ML/AI**. Hành động cần làm trước khi public: xác minh license/điều khoản với
Kaggle, nhà trường và Spotify; ghi kết luận vào `docs/DATASET_PROVENANCE.md`. Đây không phải
tư vấn pháp lý.

## 2. Cài đặt

Yêu cầu: Python 3.11–3.13. Repo đã được chạy thực tế trên Python 3.13.16; CI dùng 3.12.
Không cần GPU, database, Docker, DVC, Spark hay Spotify API.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
btl11 --help
```

Dependency chính xác của lần chạy hiện tại nằm trong `requirements.lock`. Có thể tái lập môi trường
bằng `pip install -r requirements.lock && pip install -e . --no-deps`.

## 3. Quickstart tái lập

Đặt đúng file CSV tại `data/raw/dataset.csv`, rồi chạy từ repo root:

```bash
btl11 audit --input data/raw/dataset.csv --config configs/base.yaml
btl11 prepare --config configs/base.yaml
btl11 eda --config configs/base.yaml
btl11 pca --config configs/base.yaml

btl11 sweep --algorithm kmeans --config configs/base.yaml --dry-run
btl11 sweep --algorithm kmeans --config configs/base.yaml
btl11 sweep --algorithm gmm --config configs/base.yaml
btl11 compare --experiment-set baseline_f9_standard --config configs/base.yaml

btl11 ablate --factor scaler --config configs/base.yaml
btl11 ablate --factor feature_set --config configs/base.yaml
btl11 ablate --factor pca --config configs/base.yaml

btl11 stability --model-selector finalists --config configs/base.yaml
btl11 validate-holdout --model-selector frozen_finalists --config configs/base.yaml
btl11 refit --model-selector selected_final --all-tracks --config configs/base.yaml

btl11 visualize --run-id <FINAL_REFIT_RUN_ID> --kind pca,tsne,profile,genre --config configs/base.yaml
btl11 ward --config configs/base.yaml
btl11 report --selected-run <FINAL_REFIT_RUN_ID> \
  --output artifacts/final-report-tables --config configs/base.yaml
btl11 verify --run-id <FINAL_REFIT_RUN_ID> --config configs/base.yaml

# Post-hoc robustness audit (development-only; holdout không được mở lại)
.venv/bin/python scripts/run_controlled_gmm.py --config configs/base.yaml
btl11 robustness \
  --evaluation-run 20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42 \
  --config configs/base.yaml
```

`configs/base.yaml` đã chứa shortlist, finalists và **frozen/selected** được chốt từ development
evidence. Nếu chạy lại từ artifact trống, các run cần thiết sẽ được tạo theo đúng thứ tự trên;
các ablation tái dùng baseline run tương ứng. **Holdout là one-time**: trong workspace hiện tại,
chạy lại `validate-holdout` sẽ báo lỗi cho đến khi dùng artifact directory mới. Không dùng
`--allow-repeat` để che giấu việc tái sử dụng holdout.

### Artifact của lần chạy hiện tại

- Evaluation model (frozen): `20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`
- Holdout: `20261027T032820Z_f7434a8c_b202fa49_holdout_20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_rob`
- Final refit: `20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2`
- Stability: `20261027T032241Z_ea0570dc_b202fa49_stability`
- Ward corrected (frozen transformer): `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`
- Ward historical (sample-refit scaler, superseded for controlled ARI): `20261027T032907Z_f7434a8c_b202fa49_ward_diagnostic`
- Robustness/dominance: `artifacts/robustness/20261027T070249Z/`
- Controlled GMM comparison: `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/`
- Run index: `artifacts/INDEX.csv` (63 completed runs; 0 failed experiment fits)
- Report tables: `artifacts/final-report-tables/`

## 4. Kết quả chính (chỉ từ artifact thật)

| Hạng mục | Kết quả |
|---|---|
| Final model | K-Means, F9, **RobustScaler**, no PCA, **K=2**, seed 42 |
| Development eval cohort (n=3,000) | Silhouette **0.5956**, DB **0.6505**, CH **3240.62** |
| Holdout một lần (n=3,000) | Silhouette **0.5902**, DB **0.6593**, CH **2964.07** |
| Stability (5 seeds + 5 bootstrap) | seed ARI mean **0.99969**, bootstrap ARI mean **0.99907** |
| Final refit | 89,741 tracks; cụm 0 = 72,211 (80.47%), cụm 1 = 17,530 (19.53%) |
| Profile | Cụm 1: instrumental/ambient; cụm 0: vocal/non-instrumental, năng lượng trung bình–cao |
| **Feature dominance** | `instrumentalness` = **97.29%** between-SS và **97.29%** assignment margin; single-feature K=2 ARI **0.9892**; LOO instrumentalness ARI **−0.0498**, mọi LOO khác ≥0.9938 |
| Genre post-hoc | ARI 0.00407, NMI 0.08169 trên 73,442 single-genre tracks; **không** dùng để chọn model |
| GMM comparator | full K=4 (Standard): BIC 853,058; silhouette 0.0547; hard partition khác K-Means (ARI −0.0201…0.3063 trong controlled 8-run table) |
| Ward diagnostic (n=2,000) | ARI vs selected K-Means **0.8834** với frozen transformer (run corrected); historical 0.9692 dùng sample-refit scaler; sample-only, không phải generalization |

GMM BIC giảm tới biên K=12 nhưng silhouette giảm/âm ở K cao; vì vậy BIC không được dùng
một mình để chọn K. K=6 standard có profile phong phú hơn nhưng stability thấp hơn và có cụm
khoảng 1,1%; K=2 robust được chọn vì validity trong không gian của nó, stability gần 1, cân bằng
đủ (không tối đa), tính diễn giải và parsimony.

**Caveat bắt buộc:** trong không gian RobustScaler, K=2 chủ yếu là một ngưỡng `instrumentalness`;
mô hình là **instrumental vs vocal/non-instrumental segmentation**, không phải balanced multivariate
persona discovery. StandardScaler cho partition đa biến hơn (energy/acousticness/loudness) nhưng
khác biệt (ARI 0.135) và silhouette thấp hơn trong space của nó. Xem
`docs/ROBUSTNESS_ANALYSIS.md` và `docs/DECISIONS.md` D-16.

## 5. Cấu trúc repo

```text
configs/base.yaml                 # config đã chốt shortlist/finalist/selected
data/raw/dataset.csv              # KHÔNG commit; do người dùng cung cấp
data/raw/MANIFEST.json            # provenance + license status
data/processed/                   # tracks.parquet, track_genres.parquet, audit.json
data/splits/                      # dev/holdout/eval/fit ID lists, hash-stable
src/btl11/                        # package: data, features, preprocess, models, robustness, ...
tests/                            # 50 test synthetic-only
scripts/run_controlled_gmm.py     # exploratory controlled GMM comparison runner
artifacts/runs/                   # immutable run artifacts (gitignored)
artifacts/INDEX.csv               # run ledger thực tế (63 completed)
artifacts/final-report-tables/    # bảng báo cáo máy sinh
artifacts/posthoc/                # PCA2D/t-SNE/derived views (không sửa run gốc)
artifacts/robustness/             # feature-dominance/LOO/controlled-GMM evidence
artifacts/comparisons/            # baseline + exploratory controlled comparison
artifacts/verification/           # log verify độc lập theo run
docs/                             # SSOT, report, methodology, provenance, decisions, audit
.github/workflows/ci.yml          # CI fixture-only, không có dataset thật
```

## 6. Kiểm thử và chất lượng

```bash
pytest -q
ruff check src tests scripts
```

Kết quả lần chạy hiện tại: **50 passed**, ruff sạch. Test không dùng dữ liệu thật và không thể
thay thế real-data experiments. Một end-to-end test synthetic chạy toàn bộ pipeline tới report/verify;
các test bổ sung cover zero-IQR, empty-cluster share, feature-order reorder, tampered hash, broken
freeze chain và robustness analysis.

Các guard khoa học quan trọng:

- `track_genre`/metadata không bao giờ vào feature matrix (có test leakage riêng);
- fit scaler/PCA/K-Means/GMM chỉ trên `fit` = dev trừ fixed eval cohort; holdout không tham gia fit;
- silhouette/DB/CH chỉ trên cohort ID cố định, null kèm lý do khi không xác định;
- BIC chỉ xếp hạng trong cùng representation + cohort (`feature_space_id` + `cohort_id`);
- stability ARI chỉ so các assignment trên **cùng reference IDs**, bất biến với đổi tên cluster;
- mọi run có config snapshot, hash dataset/split, checksum model/assignments; verify tính lại metric
  và kiểm tra freeze → holdout → refit chain; **63/63 run PASS**;
- feature dominance/LOO được đo riêng và không dùng để retune model hay mở lại holdout;
- không pickle/load model ngoài run artifact local do chính project tạo.

## 7. Tài nguyên và thời gian

Máy chạy thực tế: Intel Core i7-12700H, 20 logical CPUs, 38.85 GiB RAM, Linux, Python 3.13.16,
scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4. Không dùng GPU.

Per-fit time được ghi trong từng `run.json`/metrics; tổng wall-clock của toàn bộ lệnh CLI không được
instrument riêng nên không tuyên bố. Trong **56 candidate fits**: 48 fit gốc có K-Means 2.28–4.21
giây/fit và GMM 1.27–19.38 giây/fit (diag/full, n_init=2); 8 exploratory controlled GMM có
2.52–19.29 giây/fit (n_init=5). Stability chạy 60 refits, 0 lỗi; final refit K-Means trên
89,741 tracks mất khoảng 2.35 giây fit (không tính profile/plot). Full robustness analysis
(`btl11 robustness`, 10 LOO variants × 5 seed refits + attribution + controlled table) mất khoảng
3 phút 11 giây trên máy này; đây là wall-clock quan sát, không phải benchmark. Silhouette luôn bị
giới hạn trên cohort 3,000, không chạy full-pairwise 89,741.

## 8. Lỗi thường gặp

| Hiện tượng | Nguyên nhân / xử lý |
|---|---|
| `Raw dataset not found` | Đặt đúng CSV tại `data/raw/dataset.csv`; không dùng mirror |
| `License gate ... not cleared` | `MANIFEST.json` chưa có quyết định hợp lệ; xem `docs/DATASET_PROVENANCE.md` |
| `Holdout was already used` | Đúng thiết kế one-time; dùng artifact directory mới, không lạm dụng `--allow-repeat` |
| `selection.shortlist_k is empty` | Chạy baseline sweep, xem comparison, rồi điền shortlist vào config |
| `Sweep would run ... above threshold` | Dùng staged design hoặc cân nhắc `--allow-many-fits` có chủ đích |
| `VERIFY FAIL` | Không sửa tay artifact; kiểm tra hash/config snapshot/dataset bytes |
| t-SNE trông tách cụm | Chỉ là visualization; không dùng để chọn K hoặc tính metric |
| Điểm ARI genre thấp | Bình thường: genre không đồng nghĩa audio persona; đây là post-hoc diagnostic |

## 9. Phạm vi và phần chưa làm

P2 (decision-tree surrogate, Apriori/FP-Growth, DBSCAN, dashboard) **chưa triển khai** vì P0/P1
đã hoàn tất và các extension không cần cho câu hỏi nghiên cứu. Không có web app, Docker, Spotify
API runtime, recommender, supervised genre classification hay deep learning.

Việc còn lại cho con người: xác minh license/quyền phân phối trước khi public; quyết định có
commit/push lên remote hay không; rà soát và rehearsal thời lượng bộ slide bảo vệ bản PDF 12 trang
(đã thêm caveat feature-dominance). Clean-clone install sau audit đã chạy lại **PASS** ngày
2026-10-27. Bốn deck gốc và đề bài đã được đối chiếu ngày 2026-10-27 (xem `docs/REPORT.md` §11 và
`docs/DECISIONS.md` D-15).

## 10. Deliverable chính

| Deliverable | Đường dẫn |
|---|---|
| Báo cáo đầy đủ | [`docs/REPORT.md`](docs/REPORT.md) |
| Final code audit | [`docs/FINAL_CODE_AUDIT.md`](docs/FINAL_CODE_AUDIT.md) |
| Robustness/feature dominance | [`docs/ROBUSTNESS_ANALYSIS.md`](docs/ROBUSTNESS_ANALYSIS.md), [`artifacts/robustness/20261027T070249Z/`](artifacts/robustness/20261027T070249Z/) |
| Final readiness | [`docs/FINAL_READINESS.md`](docs/FINAL_READINESS.md) |
| Bảng máy sinh | [`artifacts/final-report-tables/REPORT_TABLES.md`](artifacts/final-report-tables/REPORT_TABLES.md) |
| Slide bảo vệ 12 slide | [`deliverables/BTL11_defense.pptx`](deliverables/BTL11_defense.pptx) |
| Slide bảo vệ — PDF render 12 trang | [`deliverables/BTL11_defense.pdf`](deliverables/BTL11_defense.pdf) |
| Run ledger + verify | [`artifacts/INDEX.csv`](artifacts/INDEX.csv), `artifacts/verification/` |
| Controlled GMM comparison | `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/` |
| Phương pháp / quyết định / provenance | `docs/METHODOLOGY.md`, `docs/DECISIONS.md`, `docs/DATASET_PROVENANCE.md` |
| Tái lập bằng CLI | `README.md` §2–3, `configs/base.yaml`, `requirements.lock`, `scripts/run_controlled_gmm.py` |

Không có số liệu nào trong slide hoặc báo cáo được nhập tay ngoài artifact; raw CSV vẫn không được
public vì license chưa được con người xác minh. Các phân tích robustness/controlled GMM là
**post-hoc exploratory**, không phải holdout thứ hai và không dùng để retune frozen model.
