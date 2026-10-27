# BTL11 — Music Genre & Audio Track Segmentation

**Báo cáo nghiên cứu phân cụm đặc trưng âm học trên Spotify Tracks Dataset**

Ngày khóa số liệu: **2026-10-27**  
Nguồn chuẩn: `docs/BTL11_SSOT.md`  
Bảng máy sinh: `artifacts/final-report-tables/REPORT_TABLES.md`  
Trạng thái: **CODE_COMPLETE / REAL_EXPERIMENTS_EXECUTED / PUBLIC_REDISTRIBUTION_BLOCKED**

> Mọi con số thực nghiệm trong báo cáo này được lấy từ `artifacts/runs/<run_id>/metrics.json`,
> `profiles.csv` hoặc bảng so sánh được sinh tự động. Không có bảng kết quả thủ công nào được
> chỉnh để “đẹp” hơn artifact. Các mục chưa chạy được đánh dấu rõ; báo cáo không tuyên bố điểm số
> hay rubric chưa được cung cấp.

---

## Tóm tắt

Đề tài yêu cầu phân đoạn các track theo **đặc trưng âm học** bằng phương pháp không giám sát, dùng
K-Means và Gaussian Mixture Model (GMM), có giảm chiều/trực quan hóa và mô tả cụm. Báo cáo xây
dựng pipeline tái lập trên đúng snapshot Kaggle Spotify Tracks Dataset gồm **114.000 dòng**,
**89.741 track duy nhất**, **114 giá trị thể loại** và **16.299 track đa thể loại**. Sau khi kiểm
tra trùng lặp và mâu thuẫn (0 ID có vector âm học xung đột), dữ liệu được chia theo ID thành
71.813 track development và 17.928 track holdout. Một cohort đánh giá cố định 3.000 track nằm
trong development được giữ **ngoài tập fit** (68.813 track).

Trên development evidence, mô hình cuối được khóa **trước** khi mở holdout là:

```text
K-Means, F9, RobustScaler, không PCA, K = 2, seed = 42
```

Kết quả chính:

- Development cohort (n = 3.000): Silhouette **0,5956**, Davies–Bouldin **0,6505**,
  Calinski–Harabasz **3240,62**; trên eval cohort hai cụm chiếm 19,63% và 80,37%
  (trên fit cohort: 19,55% và 80,45%).
- Holdout một lần (n = 3.000, không refit): Silhouette **0,5902**, DB **0,6593**,
  CH **2964,07**; chênh lệch nhỏ so với development.
- Stability: 5 seed refits cho ARI trung bình **0,99969**; 5 bootstrap refits cho ARI trung bình
  **0,99907** trên cùng 3.000 reference IDs.
- Refit toàn bộ 89.741 track: cụm 0 gồm 72.211 track (80,47%) — vocal/non-instrumental, năng lượng
  trung bình–cao; cụm 1 gồm 17.530 track (19,53%) — instrumental/ambient.
- **Giới hạn quan trọng (audit 2026-10-27):** trong không gian RobustScaler, `instrumentalness`
  chiếm **97,29%** between-cluster SS và **97,29%** assignment margin (cùng một đại lượng K=2);
  K-Means K=2 trên riêng feature này tái tạo partition với **ARI 0,9892** (7/3.000 track eval
  khác), còn bỏ nó đi đổi hẳn partition (ARI −0,0498).
  Mô hình cuối là **phân đoạn instrumental vs vocal/non-instrumental**, không phải cấu trúc
  đa biến cân bằng. Xem `docs/ROBUSTNESS_ANALYSIS.md`.
- Genre chỉ dùng hậu nghiệm: ARI **0,00407**, NMI **0,08169** trên 73.442 track đơn thể loại.
  Điều này khẳng định các cụm **không phải là thể loại** và không được thiết kế để tái tạo genre.

Kết luận trung thực: dữ liệu tạo thành **hai nhóm instrumental vs vocal/non-instrumental ổn định**
trên không gian F9 với robust scaling; đây là phân đoạn gần như đơn biến theo `instrumentalness`,
không phải bằng chứng cấu trúc đa biến cân bằng. Các phân đoạn K=6 giàu diễn giải hơn nhưng kém
ổn định hơn và có cụm nhỏ khoảng 1,1% (1,14% trên fit). GMM bổ sung xác suất thành viên mềm nhưng
hard partition của GMM khác đáng kể so với K-Means (ARI −0,02…0,31 trong so sánh controlled);
hai mô hình trả lời hai câu hỏi khác nhau (density fit vs hard segmentation). License của snapshot
vẫn là **cổng con người**: Kaggle ghi ODbL 1.0, mirror Hugging Face ghi BSD, dữ liệu có nguồn gốc
Spotify Web API; **không được public raw CSV** khi chưa xác minh.

---

## 1. Bài toán, câu hỏi nghiên cứu và phạm vi

### 1.1 Định nghĩa

Cho \(n\) track, mỗi track được biểu diễn bởi vector đặc trưng âm học

\[
\mathbf{x}_i=(x_{i1},\ldots,x_{id})\in\mathbb{R}^d,
\]

cần tìm phân hoạch \(f:\mathbb{R}^d\to\{1,\ldots,K\}\) (K-Means) hoặc phân phối hậu nghiệm
\(p(z_i=k\mid\mathbf{x}_i)\) (GMM) sao cho các track tương đồng về âm học nằm cùng nhóm. Bài toán
là **unsupervised**: không có nhãn ground-truth cho “cụm đúng”, và `track_genre` không phải target.

### 1.2 Câu hỏi nghiên cứu

1. Chỉ với các đặc trưng âm học, dữ liệu có cấu trúc cụm có ý nghĩa không?
2. K-Means (hard, centroid) và GMM (soft, ellipsoidal) khác nhau thế nào về phân đoạn và độ bất định?
3. K nào hợp lý khi xét đồng thời validity, cân bằng cụm, stability, diễn giải và parsimony?
4. Các cụm có thể mô tả thành “audio personas” mà không nhầm với thể loại không?

### 1.3 Vì sao không supervised genre classification

Nếu dùng genre để chọn K, chọn scaler hoặc chọn model thì bài toán trở thành tối ưu ngầm theo nhãn,
phá vỡ mục tiêu khám phá cấu trúc âm học. Genre chỉ được mở **sau khi khóa mô hình** để kiểm tra
hậu nghiệm. Báo cáo không dùng accuracy, không tuyên bố “dự đoán genre”, và không huấn luyện
classifier. Đây là khác biệt cốt lõi giữa unsupervised segmentation và supervised classification.

### 1.4 Phạm vi

Trong phạm vi: audit dữ liệu, dedup, split, EDA, F5/F9, Standard/Robust scaling, PCA, K-Means,
GMM, intrinsic metrics, stability, holdout, refit, profiles, PCA2D/t-SNE, Ward diagnostic và
genre post-hoc.

Ngoài phạm vi: recommender cá nhân hóa, Spotify API runtime, raw audio download, deep learning,
cloud deployment, dashboard, supervised genre prediction, association-rule mining bắt buộc.
Các P2 như decision-tree surrogate, Apriori/FP-Growth, DBSCAN và dashboard được **deferred có chủ ý**
sau khi P0/P1 hoàn tất.

---

## 2. Dữ liệu và provenance

### 2.1 Danh tính chính xác

| Trường | Giá trị |
|---|---|
| Dataset | Kaggle — Spotify Tracks Dataset |
| Kaggle id | `maharshipandya/-spotify-tracks-dataset` |
| Trang chuẩn | <https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset> |
| File thực nghiệm | `data/raw/dataset.csv` |
| SHA-256 | `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd` |
| Bytes | 20.118.244 |
| Rows | 114.000 |
| Unique `track_id` | 89.741 |
| Số genre thực đo | 114 |
| Track đa thể loại | 16.299 |
| Genre bridge `(track_id, genre)` | 113.550 |
| Raw file timestamp | 2022-10-22 (mtime filesystem, không phải bằng chứng thời điểm tải) |

Đây là **dataset duy nhất** được dùng cho kết quả thực nghiệm. GTZAN, FMA, Million Song Dataset,
các biến thể Spotify khác, mirror Hugging Face, CSV demo, dữ liệu scrape và Spotify Web API bị cấm
thay thế. Fixture tổng hợp trong `tests/` chỉ là test-only, không phải evidence.

### 2.2 Provenance

File do chủ dự án cung cấp tại repository root, được đưa vào `data/raw/` mà không đổi bytes.
Không có Kaggle API/CLI download, không tải mirror, không gọi Spotify API. `MANIFEST.json` ghi
nhận source, id, URL, hash, bytes, file mtime, audit time, license status và cờ redistribution.

Dataset card/Kaggle page mô tả “125 genres”, nhưng audit thực tế đếm **114** giá trị
`track_genre`; báo cáo dùng 114. Sự không đồng nhất mô tả–artifact là lý do không hard-code
số genre từ web.

### 2.3 License và quyền phân phối

Kiểm tra ngày 2026-10-27:

- Kaggle page metadata hiển thị **ODbL 1.0** (Open Data Commons Open Database License);
- Hugging Face mirror card hiển thị **BSD**;
- Nguồn mô tả dữ liệu được thu thập qua **Spotify Web API**;
- Spotify Developer Policy có các hạn chế đối với Spotify Platform/Content, bao gồm ML/AI.

Hai license công khai xung đột nhau; nguồn gốc Spotify tạo thêm rủi ro. Vì vậy trạng thái là
`pending_human_verification`, `redistribution_allowed=false`. Phân tích học thuật cục bộ được
chủ dự án cho phép cho bài tập, nhưng **không phải license phân phối**. Raw CSV bị gitignore và
không được công bố. Đây không phải tư vấn pháp lý. Hành động con người trước khi public được ghi
tại `docs/DATASET_PROVENANCE.md`.

---

## 3. Đơn vị phân tích, trùng lặp và nhãn genre đa trị

Một hàng CSV không đại diện cho một track duy nhất. Trong snapshot:

- 114.000 dòng tương ứng 89.741 `track_id`;
- 16.641 ID xuất hiện nhiều hơn một lần, tối đa 9 lần;
- 40.900 dòng thuộc các ID lặp;
- 16.299 track có từ hai genre trở lên;
- sau khi bỏ cột index export, có 894 dòng thuộc nhóm trùng hoàn toàn (450 dòng dư sau dòng đầu);
- **0 ID** có vector âm học xung đột quá `atol=1e-8`, `rtol=1e-6`.

Nếu giữ mọi dòng, track có nhiều genre sẽ bị nhân trọng số trong centroid/density. Pipeline do đó:

1. Kiểm tra schema và ID;
2. Phát hiện vector xung đột; nếu có, **fail closed** và ghi quarantine, không lấy dòng đầu/average;
3. Giữ một hàng đại diện deterministic cho mỗi ID;
4. Lưu tất cả cặp `(track_id, genre)` vào bảng many-to-many;
5. Không đưa `track_genre`, `track_id` hay metadata vào X.

Kết quả: `tracks.parquet` có đúng một hàng/ID, sort theo ID; `track_genres.parquet` có 113.550
cặp duy nhất. Track được gán nhiều genre **không** bị tính nhiều lần trong clustering.

### 3.1 Split

| Tập | Số ID | Vai trò |
|---|---:|---|
| Development | 71.813 | Chọn model, ablation, stability |
| Fit | 68.813 | Fit scaler/PCA/clusterer |
| Fixed eval cohort | 3.000 | Chấm Silhouette/DB/CH; **không fit** |
| Holdout | 17.928 | Một lần kiểm tra sau freeze |
| Holdout eval cohort | 3.000 | Silhouette/DB/CH holdout |

Split dùng hash ổn định của `seed:track_id`, không random theo hàng. `fit` và `eval` ID-disjoint;
`dev` và `holdout` ID-disjoint; kiểm tra assert trong code và test. Việc tách eval cohort khỏi fit
giảm lạc quan in-sample và ngăn “cùng một track vừa fit vừa chấm”.

---

## 4. EDA và tiền xử lý

### 4.1 Phân bố đặc trưng

EDA máy sinh nằm tại `artifacts/eda/20261027T031519Z/`. Một số thống kê trung vị toàn bộ
89.741 track:

| Feature | Median | Ý nghĩa ngắn |
|---|---:|---|
| acousticness | 0,188 | Đa số track không acoustic |
| danceability | 0,576 | Trung bình |
| energy | 0,676 | Thiên năng lượng |
| valence | 0,457 | Trung tính |
| tempo | 122,013 BPM | Nhịp phổ biến |
| loudness | −7,185 dB | Tương đối to |
| speechiness | 0,0489 | Chủ yếu không lời nói |
| instrumentalness | 0,000058 | Đa số có vocal |
| liveness | 0,132 | Ít live |

Các correlation mạnh nhất (Pearson, raw):

| Cặp | r |
|---|---:|
| energy–loudness | +0,7588 |
| acousticness–energy | −0,7326 |
| acousticness–loudness | −0,5827 |
| danceability–valence | +0,4926 |
| loudness–instrumentalness | −0,4347 |

Correlation khẳng định F9 có đa cộng tuyến; PCA có thể decorrelate nhưng không được mặc định thay
thế F9. Feature distributions và heatmap: `figures/feature_distributions.png`,
`figures/feature_correlation.png`.

### 4.2 Data-quality flags

- `tempo = 0`: **157** track duy nhất — giữ lại, ghi flag;
- `loudness > 0`: **90** dòng, tương ứng **68** track duy nhất — giữ lại, ghi flag;
- 7 đặc trưng dự kiến 0–1: **0** giá trị ngoài khoảng;
- audio feature null/nonfinite: **0**;
- missing metadata: 1 dòng ở `artists`, `album_name`, `track_name`; metadata không dùng cho X.

Không winsorize/xóa outlier mặc định. Outlier IQR được báo cáo trong audit; lý do là track khác biệt
có thể là tín hiệu thật, không phải lỗi.

### 4.3 Chuẩn hóa

K-Means dùng khoảng cách Euclidean, nên scale là bắt buộc. StandardScaler:

\[
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j},
\]

RobustScaler:

\[
z^{r}_{ij}=\frac{x_{ij}-\operatorname{median}(x_j)}
{Q_{0.75}(x_j)-Q_{0.25}(x_j)}.
\]

Scaler/PCA được fit **chỉ trên 68.813 fit IDs**; eval/holdout chỉ transform. Test adversarial
dịch chuyển holdout xác nhận mean của scaler không đổi khi transform holdout.

### 4.4 PCA

PCA chạy sau scaling, `whiten=False`, `svd_solver="full"`, chỉ trên fit cohort:

| Cấu hình | Components thực tế | Cumulative variance |
|---|---:|---:|
| PCA90 | 7 | 0,94823 |
| PCA95 | 8 | 0,98444 |

Explained variance ratios: 0,32158; 0,15835; 0,13879; 0,09994; 0,09704; 0,08170; 0,05083;
0,03621; phần còn lại 0,01556. PC1 tương quan dương với energy/loudness và âm với acousticness;
PC2 tương quan dương với danceability/valence; PC3 nhấn speechiness/liveness. Loadings đầy đủ:
`artifacts/pca/20261027T031522Z/tables/loadings_pca95.csv`.

PCA không phải “95% độ chính xác”. Với F9 chỉ 9 chiều, lợi ích chính là decorrelation, nén nhiễu,
trực quan và minh họa giảm chiều; no-PCA vẫn là representation đơn giản nhất. PCA2D dùng cho hình
chỉ là projection hiển thị, không thay thế F9.

---

## 5. Phương pháp

### 5.1 K-Means

Mục tiêu:

\[
\min_{C_1,\ldots,C_K} J=\sum_{k=1}^{K}\sum_{\mathbf{x}_i\in C_k}
\lVert\mathbf{x}_i-\boldsymbol{\mu}_k\rVert_2^2,
\]

với centroid \(\boldsymbol{\mu}_k=\frac{1}{|C_k|}\sum_{i\in C_k}\mathbf{x}_i\). scikit-learn
`KMeans(init='k-means++', n_init=20, max_iter=500, tol=1e-4, random_state=seed, algorithm='lloyd')`.
`k-means++` chọn seed xa nhau có xác suất; `n_init=20` giảm local optimum; `random_state` được ghi.
Inertia là SSE trên fit cohort; elbow chỉ là chẩn đoán, không phải tiêu chí chọn duy nhất.

### 5.2 GMM và EM

\[
p(\mathbf{x})=\sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}\mid\mu_k,\Sigma_k),
\qquad
\gamma_{ik}=\frac{\pi_k\mathcal{N}(\mathbf{x}_i\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal{N}(\mathbf{x}_i\mid\mu_j,\Sigma_j)}.
\]

EM lặp E-step/M-step; \(N_k=\sum_i\gamma_{ik}\),
\(\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}\mathbf{x}_i\),
\(\Sigma_k=\frac{1}{N_k}\sum_i\gamma_{ik}(\mathbf{x}_i-\mu_k)(\mathbf{x}_i-\mu_k)^T\),
\(\pi_k=N_k/n\). Hard label là \(\arg\max_k\gamma_{ik}\). Posterior không phải xác suất genre.

Độ bất định:

\[
\text{confidence}_i=\max_k\gamma_{ik},\qquad
H_i=-\sum_k\gamma_{ik}\log\gamma_{ik},\qquad
H^{norm}_i=H_i/\log K.
\]

BIC/AIC:

\[
\mathrm{BIC}=-2\log\hat L+p\ln n,\qquad \mathrm{AIC}=-2\log\hat L+2p.
\]

Với \(d=9,K=12\), số tham số là 227 (diag) và 659 (full). BIC chỉ được so trong cùng
representation/cohort; thay đổi dimension/scale làm thay đổi base measure. BIC thấp là fit density
tốt hơn có phạt complexity, không đảm bảo cụm ngữ nghĩa tốt hơn.

### 5.3 PCA, t-SNE, Ward

- PCA: các component trực giao tối đa variance; scaling phải trước PCA.
- t-SNE: KL divergence giữa phân phối láng giềng; non-convex, phụ thuộc perplexity/init. Chỉ dùng
  để hiển thị; không chọn K, không tính silhouette trên tọa độ t-SNE, không suy luận khoảng cách.
- Ward: mỗi merge tăng within-cluster variance ít nhất; chỉ chạy trên mẫu deterministic 2.000 vì
  agglomerative không có connectivity có chi phí lớn; không có `.predict` cho track mới.

### 5.4 Metric

- Silhouette \(s(i)=(b(i)-a(i))/\max(a(i),b(i))\) — maximize;
- Davies–Bouldin — minimize;
- Calinski–Harabasz — maximize;
- Inertia/SSE — minimize, dùng elbow;
- BIC/AIC — GMM, minimize, chỉ trong cùng space;
- ARI — stability, bất biến hoán vị nhãn;
- ARI/NMI vs genre — chỉ post-hoc, không tune.

Silhouette được tính trên cohort cố định 3.000; nếu số cụm quan sát \(<2\) hoặc \(>n-1\) thì
lưu `null` kèm lý do. Không materialize ma trận khoảng cách 89.741×89.741.

---

## 6. Thiết kế thực nghiệm

### 6.1 Ma trận thực nghiệm

| ID | Nội dung | Trạng thái | Artifact chính |
|---|---|---|---|
| E00 | K-Means không scale (diagnostic) | DONE | `artifacts/ablations/scaler/.../none_unscaled_*` |
| E01 | Standard vs Robust | DONE | `artifacts/ablations/scaler/20261027T032128Z/` |
| E02 | F5 vs F9 | DONE | `artifacts/ablations/feature_set/20261027T032140Z/` |
| E03 | K-Means K=2..12 | DONE | 11 run + comparison `20261027T070434Z` (cohort-aware) |
| E04 | K-Means 5 seeds + bootstrap | DONE | stability run `20261027T032241Z_ea0570dc_b202fa49_stability` |
| E05 | GMM diag/full K=2..12 | DONE | 22 run + comparison |
| E06 | GMM responsibilities | DONE | metrics + uncertainty tables |
| E07 | no-PCA/PCA90/PCA95 | DONE | `artifacts/ablations/pca/20261027T032204Z/` |
| E08 | PCA2D | DONE | `artifacts/posthoc/<final>/figures/pca2_scatter.png` |
| E09 | t-SNE sample | DONE | `artifacts/posthoc/<final>/figures/tsne2d.png` |
| E10 | Ward sample | DONE (corrected) | `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr` |
| E11 | Genre post-hoc | DONE | final run `tables/genre_*` |
| E12 | Final profiling | DONE | final run `profiles.csv` |
| E13 | Feature dominance + LOO robustness (hậu nghiệm, development-only) | DONE | `artifacts/robustness/20261027T070249Z/` |
| E14 | Controlled K-Means vs GMM (F9/noPCA, Standard+Robust, K=2/4) | DONE (exploratory) | `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/` |

### 6.2 Protocol

1. Fit scaler/PCA/estimator trên fit IDs.
2. Predict/score trên fixed eval cohort.
3. Sweep KMeans K=2..12 và GMM diag/full K=2..12 trên F9/Standard/no-PCA.
4. Profile các ứng viên để chọn shortlist K=2,4,6; đưa các ablation standout vào stability.
5. Stability 5 seeds + 5 bootstrap trên cùng 3.000 reference IDs.
6. Khóa final trước holdout; chỉ một lần holdout.
7. Refit toàn bộ track cho assignment/profile/visual; model refit có ID/hash riêng.
8. Genre chỉ mở sau freeze.

### 6.3 Vì sao hai lifecycle

Evaluation model được fit trên dev và dùng để so sánh/kiểm định holdout. Final refit được fit trên
toàn bộ 89.741 track để tạo output cuối; đây là **model khác** (n khác, hash khác). Final refit
không kế thừa holdout metric. Điều này tránh tuyên bố sai rằng model full-data đã được kiểm định
out-of-sample.

---

## 7. Kết quả thực nghiệm

Tất cả bảng dưới đây lấy từ artifact thật. Bảng đầy đủ và figure index:
`artifacts/final-report-tables/REPORT_TABLES.md`. Run ledger: `artifacts/INDEX.csv`.

### 7.1 K-Means baseline F9/Standard/no-PCA

11/11 fit hội tụ, 0 lỗi; cohort eval 3.000; fit 68.813.

| K | Silhouette ↑ | DB ↓ | CH ↑ | Min cluster share (fit) | Inertia fit |
|---:|---:|---:|---:|---:|---:|
| 2 | **0,25759** | 1,58475 | **800,62** | 0,24891 | 484.912,43 |
| 3 | 0,16535 | 1,87735 | 651,66 | 0,21734 | 430.566,71 |
| 4 | 0,17416 | 1,68773 | 588,53 | 0,07757 | 387.558,89 |
| 5 | 0,17762 | 1,50518 | 563,68 | 0,01193 | 349.801,84 |
| 6 | 0,18958 | 1,43414 | 556,75 | 0,01136 | 318.000,36 |
| 7 | 0,19515 | 1,37790 | 561,27 | 0,01131 | 289.373,46 |
| 8 | 0,17433 | 1,39507 | 529,78 | 0,01122 | 273.578,53 |
| 9 | 0,18020 | **1,37008** | 517,05 | 0,01014 | 259.282,33 |
| 10 | 0,16958 | 1,42494 | 495,14 | 0,01013 | 248.708,12 |
| 11 | 0,17335 | 1,43209 | 479,86 | 0,01013 | 239.395,61 |
| 12 | 0,17398 | 1,42832 | 459,79 | 0,01013 | 231.866,96 |

Đọc bảng:

- K=2 có silhouette/CH cao nhất và cụm nhỏ nhất vẫn 24,89%;
- K=9 có DB thấp nhất (1,37008), K=7 đứng thứ hai (1,37790); K=5–12 đều có cụm nhỏ khoảng 1%;
- Không có K nào “thắng mọi metric”. Đây là lý do cần multi-criteria + stability + profile.

Figure: `artifacts/comparisons/baseline_f9_standard/20261027T070434Z/elbow.png` và
`kmeans_metrics.png`. Bảng đầy đủ: `comparison.csv` cùng thư mục.

### 7.2 GMM baseline

22/22 fit diag/full K=2..12 hội tụ. BIC/AIC chỉ so trong cùng F9/Standard/no-PCA.

| K | Cov | BIC ↓ | AIC ↓ | Silhouette | DB ↓ |
|---:|---|---:|---:|---:|---:|
| 2 | diag | 1.203.265 | 1.202.927 | 0,11364 | 3,0570 |
| 2 | full | 1.397.164 | 1.396.167 | 0,09609 | 3,1629 |
| 4 | diag | 1.006.496 | 1.005.810 | 0,05391 | 2,5813 |
| 4 | full | 853.058 | 851.057 | 0,05469 | 2,5828 |
| 8 | full | 737.685 | 733.673 | −0,01182 | 4,5464 |
| 12 | diag | 767.035 | 764.960 | 0,00766 | 4,0323 |
| 12 | full | **678.842** | **672.820** | **−0,02341** | 3,5962 |

BIC giảm tới biên K=12 nhưng silhouette hard-label giảm/âm ở K cao. Đây là bằng chứng rằng
**BIC không phải tiêu chí chọn K duy nhất** và GMM đang fit density chứ không tạo phân đoạn tách biệt.
Không mở rộng K lên 20 vì validity không ủng hộ; quyết định ghi tại `docs/DECISIONS.md` D-04.

GMM full K=4 được giữ làm comparator mềm: BIC 853.058, AIC 851.057, mean posterior confidence
0,9578 trên eval; 0,33% track eval (10/3.000) và 0,30% assignment dev (214/71.813) có confidence <0,5; mean normalized entropy 0,0802. Đây là thông tin
hữu ích về track biên, không phải xác suất genre.

Audit-controlled 2026-10-27 (8 dòng, cùng fit/eval IDs, F9, scaler, K, seed; GMM `n_init=5`)
cho thấy hard partition của GMM khác K-Means: ARI **−0,0201…0,3063**, dù posterior tự tin
(0,947–0,997). Đây là so sánh density vs segmentation, không phải “GMM thua”; bảng đầy đủ ở §8.2
và `docs/ROBUSTNESS_ANALYSIS.md`.

Figure: `artifacts/comparisons/baseline_f9_standard/20261027T070434Z/gmm_bic_aic.png`.

### 7.3 Ablations

24 dòng ablation; 15 fit mới + 9 tái dùng baseline; 0 lỗi. Bảng đầy đủ:
`artifacts/ablations/*/20261027T0321*/comparison.csv`.

| Factor | Cấu hình | Silhouette | DB | CH | Min share |
|---|---|---:|---:|---:|---:|
| Scaler K2 | Standard | 0,25759 | 1,58475 | 800,62 | 0,24891 |
| Scaler K2 | Robust | **0,59562** | **0,65047** | **3240,62** | 0,19546 |
| Scaler K2 | None (E00) | 0,51673 | 0,65015 | 4883,66 | 0,49303 |
| Feature K2 | F9 | 0,25759 | 1,58475 | 800,62 | 0,24891 |
| Feature K2 | F5 | 0,297997 | 1,37000 | 1226,40 | 0,30267 |
| PCA K6 | no PCA | 0,18958 | 1,43414 | 556,75 | 0,01136 |
| PCA K6 | PCA90 (7 comp) | **0,20329** | **1,36126** | **614,50** | 0,01144 |
| PCA K6 | PCA95 (8 comp) | 0,19378 | 1,41415 | 572,94 | 0,01136 |

Cảnh báo diễn giải:

- Metric ở các scaler/PCA/feature space khác nhau **không cùng thang đo**; không xếp hạng tuyệt đối
  chỉ bằng silhouette. E00 unscaled có silhouette cao nhưng bị tempo/loudness chi phối; K=2 tạo
  49,3%/50,7% và không được khuyến nghị.
- Robust scaling K4/K6 tạo cụm nhỏ ~1,0%; do đó K=2 robust được ưu tiên hơn robust K4/K6.
- F5 compact hơn nhưng mất loudness/speechiness/instrumentalness/liveness — những feature tạo
  persona instrumental/speech. F9 được giữ làm primary.
- PCA90 cải thiện nhẹ nhưng thêm representation và làm loadings gián tiếp; no-PCA được giữ cho
  final vì đơn giản và diễn giải trực tiếp.

### 7.4 Stability

Run `20261027T032241Z_ea0570dc_b202fa49_stability`; 60/60 refits thành công; 5 seed + 5 bootstrap
mỗi finalist; tất cả dự đoán trên cùng 3.000 reference IDs.

| Finalist | Seed ARI mean | Bootstrap ARI mean | Pairwise ARI mean |
|---|---:|---:|---:|
| KMeans std K2 | 1,000000 | 0,991433 | 0,992008 |
| KMeans std K4 | 0,996929 | 0,980867 | 0,982174 |
| KMeans std K6 | 0,933423 | 0,916413 | 0,869597 |
| **KMeans robust K2** | **0,999692** | **0,999075** | **0,999177** |
| KMeans PCA90 K6 | 0,998739 | 0,982706 | 0,985122 |
| GMM full K4 | 0,978550 | 0,936991 | 0,925512 |

K=6 standard kém ổn định thật sự; không bị che giấu. Robust K2 gần như bất biến qua seed và
bootstrap. Figure: `artifacts/runs/20261027T032241Z_ea0570dc_b202fa49_stability/figures/stability.png`.

### 7.5 Freeze, holdout và refit

Final được khóa **trước** holdout:

```text
K-Means / F9 / RobustScaler / no PCA / K=2 / seed=42
```

Evaluation run:
`20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`  
Holdout run:
`20261027T032820Z_f7434a8c_b202fa49_holdout_20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_rob`

| Metric | Dev eval (n=3.000) | Holdout (n=3.000) | Delta |
|---|---:|---:|---:|
| Silhouette ↑ | 0,595621 | 0,590222 | −0,005400 |
| Davies–Bouldin ↓ | 0,650472 | 0,659256 | +0,008785 |
| Calinski–Harabasz ↑ | 3240,617 | 2964,073 | −276,545 |

Holdout không refit và không dùng để chọn. Kết quả tương đương development là **sanity check cấu
trúc**, không phải accuracy vì không có nhãn ground-truth. Figure:
`artifacts/runs/20261027T032820Z_.../figures/cluster_sizes_holdout.png`.

Final refit:
`20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2` — 89.741 track, fit 2,35 giây
(K-Means), model identity riêng. Metric mẫu verification 3.000 track là in-sample descriptive:
Silhouette 0,584193; DB 0,673045; CH 2912,169. Các số này **không** thay thế holdout.

### 7.6 Cụm cuối và profile đơn vị gốc

| Cụm | n | Share | Acoustic median | Dance median | Energy median | Instrumental median | Speech median | Valence median | Tempo median | Loudness median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 72.211 | 80,47% | 0,192 | 0,584 | 0,688 | 0,000005 | 0,0492 | 0,505 | 122,013 | −6,688 |
| 1 | 17.530 | 19,53% | 0,149 | 0,520 | 0,586 | **0,858** | 0,0482 | 0,229 | 122,013 | −10,660 |

Deviation chuẩn hóa so với toàn bộ:

- Cụm 0: loudness +0,20, valence +0,15, danceability +0,09; instrumentalness −0,48.
- Cụm 1: instrumentalness **+1,96**, acousticness +0,17; loudness −0,82, valence −0,63,
  danceability −0,36.

Diễn giải giả thuyết:

- **Cụm 0 — Vocal/non-instrumental, năng lượng trung bình–cao.** Đây là nhóm rộng, không phải
  “EDM”; nó chứa nhiều track có vocal và năng lượng khá cao. Boundary xa tâm có thể là spoken-word
  (ví dụ track có speechiness 0,955), nhưng median speechiness chỉ 0,049.
- **Cụm 1 — Instrumental/ambient.** Instrumentalness median 0,858, IQR 0,179; acousticness trung vị
  0,149 nhưng IQR 0,879; năng lượng và loudness thấp hơn.

Đây là **mô tả thống kê/giả thuyết**, không phải genre đã nghe kiểm chứng. Bảng đầy đủ:
`profiles.csv`, `tables/profiles_long.csv`, heatmap `figures/profile_heatmap.png`.

### 7.7 Track đại diện (metadata local, không nghe audio)

| Cụm | Loại | Track | Artist | Instrumental | Energy | Dance | Speech |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | near centroid | Aynalarımda | Stabil; Bedo | 0,000026 | 0,749 | 0,614 | 0,0746 |
| 0 | near centroid | Re Re Bajarangi | Kailash Kher | 0,000000 | 0,740 | 0,539 | 0,0895 |
| 0 | boundary | Viagra | Martin Lawrence | 0,000000 | 0,366 | 0,644 | 0,9550 |
| 1 | near centroid | Letters — NTO Remix | KAS:ST; NTO | 0,806 | 0,560 | 0,493 | 0,0518 |
| 1 | near centroid | CLOCK | Plaid | 0,822 | 0,522 | 0,453 | 0,0537 |
| 1 | boundary | Sonidos de Aves | Sonidos De Aves | 0,913 | 0,0173 | 0,411 | 0,7630 |

Track được chọn theo khoảng cách tới centroid trong không gian đã biến đổi hoặc là boundary xa tâm;
không có nghĩa pipeline đã nghe hay xác nhận chất lượng bài hát. Bảng đầy đủ:
`tables/representative_tracks.csv`.

### 7.8 Genre post-hoc

Chỉ mở sau freeze, trên assignment của final refit:

- Multi-label incidence: `tables/genre_cluster_incidence.csv`,
  heatmap top 20: `figures/genre_cluster_heatmap.png`.
- Single-genre-only cohort: **73.442** track; bỏ **16.299** track đa nhãn (không ép nhãn).
- ARI vs genre: **0,004067**; NMI: **0,081685**; cohort đơn nhãn có 112 genre, 2 cluster.

Top incidence:

| Cụm | Genre (top 5) | % unique tracks in cluster |
|---:|---|---|
| 1 | study | 5,20% |
| 1 | minimal-techno | 5,03% |
| 1 | sleep | 4,65% |
| 1 | new-age | 4,61% |
| 1 | detroit-techno | 4,57% |
| 0 | pagode / reggae / reggaeton / sertanejo | 1,38% mỗi genre |
| 0 | country / forro / mandopop | 1,38% mỗi genre |

Diễn giải: cluster instrumental/ambient tập trung nhiều ở study/sleep/minimal-techno/new-age/
ambient/classical; cluster vocal rộng phân bố gần như đều trên nhiều genre phổ thông. ARI/NMI thấp
không phải lỗi pipeline: genre taxonomy không định nghĩa acoustic personas; track multi-genre và
mỗi genre có khoảng 1.000 track khiến incidence bị giới hạn. Genre không được dùng để chọn model.

### 7.9 PCA2D và t-SNE

- PCA2D: 6.000 track deterministic; re-fit PCA(2) trên không gian đã transform để hiển thị;
  explained variance 55,70% + 25,03%. File:
  `artifacts/posthoc/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/tables/pca2_coordinates.csv`,
  `.../figures/pca2_scatter.png`. Đây không phải representation huấn luyện.
- t-SNE: 3.000 track, perplexity 30, `init=pca`, random_state 42, 1.000 iter; file:
  `artifacts/posthoc/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/tables/tsne_coordinates.csv`,
  `.../figures/tsne2d.png`. Nhãn được tô màu sau khi model đã khóa.
  Không dùng t-SNE để chọn K, không tính silhouette trên tọa độ t-SNE, không suy luận mật độ/khoảng
  cách 2D.

### 7.10 Ward diagnostic

Run corrected `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr` (audit
2026-10-27):

- mẫu deterministic **2.000** track từ fit IDs, cùng F9/Robust;
- transformer là **fitted transformer của frozen evaluation model** (không refit scaler trên
  mẫu), nên ARI so sánh thuần túy thuật toán Ward với K-Means trên cùng biểu diễn;
- Ward K=2; Silhouette 0,5945; DB 0,6142; CH 1938,4 trên **sample representation**;
- ARI với assignment của **frozen evaluation model** trên **cùng 2.000 ID**: **0,883442**.

Run lịch sử `20261027T032907Z_f7434a8c_b202fa49_ward_diagnostic` báo ARI 0,969205, nhưng run
đó refit `RobustScaler` trên chính mẫu 2.000 track, nên ARI trộn thay đổi thuật toán với thay đổi
scaler; con số 0,969205 không phải so sánh like-for-like. Run lịch sử vẫn được giữ nguyên làm
lịch sử; con số **0,883442** là con số controlled được dùng trong báo cáo.

Ward không có `.predict` cho track mới; metric sample không phải generalization. ARI 0,883 vẫn
cao và cho thấy cấu trúc phân đôi instrumental/vocal không phải riêng K-Means tạo ra, nhưng mức
đồng thuận thấp hơn con số 0,969 trước đây. Dendrogram:
`figures/ward_dendrogram.png` trong run corrected.

### 7.11 Kiểm chứng kỹ thuật và khoa học

- **63 runs completed; 0 experiment fit failed.** 56 candidate runs, 3 final refits, 1 stability,
  1 holdout, 2 Ward (historical + corrected); 8 candidate runs are the exploratory controlled GMM
  comparison. Index: `artifacts/INDEX.csv`.
- Verify: **63/63 runs PASS, 0 failures**. Key check counts: frozen 61, holdout 70, final refit 84,
  stability 68, old Ward 31, corrected Ward 31, GMM full K4 65, PCA90 K6 63; the eight exploratory
  GMM runs each pass 65 checks. Log: `artifacts/verification/*.json`; report rejects a final refit
  whose evaluation model differs from the holdout frozen run and one built from different dataset
  bytes.
- Pytest: **50 passed** (synthetic-only), gồm leakage, conflict, split, PCA, GMM, ARI, profile,
  failed-run, CLI, report-path, zero-IQR, empty-cluster, tampered-hash, broken-chain, config-lock
  và end-to-end robustness.
- Independent recomputation: silhouette/DB/CH of frozen, holdout and final refit match the stored
  values exactly; frozen evaluation model and final refit agree on all verification IDs (ARI 1,0).
- Robustness: `artifacts/robustness/20261027T070249Z/`; controlled GMM comparison:
  `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/`.
- Ruff: sạch.
- Mọi model run (candidate/holdout/final) có config snapshot, dataset/processed/split hashes,
  feature order, seed, metrics, model và assignment checksums; stability/Ward meta-runs lưu
  metrics/tables riêng. Verify re-predict và tính lại Silhouette/DB/CH (và inertia/BIC khi có).
- Hardware: i7-12700H, 20 logical CPU, 38,85 GiB RAM; Python 3.13.16; không GPU.
- Không có git commit/push vì không được yêu cầu; run.json ghi `git.available=false`.

### 7.12 Feature dominance và độ nhạy leave-one-feature-out (hậu nghiệm 2026-10-27)

Đây là phân tích **hậu nghiệm, exploratory, chỉ trên development IDs**; holdout không được dùng
và model cuối không đổi. Nguồn: `artifacts/robustness/20261027T070249Z/`
(`robustness.json`, các CSV và figures). Run được phân tích:
`20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42`.

**Scaler và phân bố.** Trong không gian RobustScaler, `instrumentalness` có IQR 0,0985, 33,34%
track đúng bằng 0 và 52,12% track < 1e-4; transformed variance là **10,82**, so với 5,03 của
`speechiness` và < 1,0 cho hầu hết feature còn lại. Vì RobustScaler chia theo IQR, feature này
chiếm ưu thế lớn trong khoảng cách Euclidean.

**Phân rã phương sai và assignment margin** (fit cohort, K=2):

| Feature | between share (η²) | share tổng between SS | share mean margin | flip share nếu bỏ |
|---|---:|---:|---:|---:|
| acousticness | 0,0075 | 0,0002 | 0,0002 | 0,0001 |
| danceability | 0,0313 | 0,0016 | 0,0016 | 0,0004 |
| energy | 0,0280 | 0,0011 | 0,0011 | 0,0003 |
| valence | 0,0956 | 0,0034 | 0,0034 | 0,0004 |
| tempo | 0,0027 | 0,0001 | 0,0001 | 0,0000 |
| loudness | 0,1631 | 0,0156 | 0,0156 | 0,0013 |
| speechiness | 0,0087 | 0,0042 | 0,0042 | 0,0007 |
| **instrumentalness** | **0,9311** | **0,9729** | **0,9729** | **0,3005** |
| liveness | 0,0069 | 0,0008 | 0,0008 | 0,0002 |

Nói cách khác, **97,29%** between-cluster separation (0,972878) và **97,29%** assignment margin
(0,972872) đến từ `instrumentalness`; bỏ feature này sẽ đổi assignment của **30,05%** track trong
fit cohort. Khi K=2, hai con số này là hai cách tính của **cùng một đại lượng** `δ²/Σδ²`, không
phải hai bằng chứng độc lập.
`speechiness` là nguồn within-cluster dispersion lớn nhất (50,77% tổng within SS) nhưng chỉ
đóng góp 0,42% margin — nó làm cụm loãng chứ không tách cụm.

**Single-feature K-Means (K=2, fit IDs, đánh giá trên cùng 3.000 eval IDs).**
`instrumentalness` một mình tái tạo partition với **ARI 0,9892** (19,6% / 80,4%): **7 trong
3.000** track eval khác assignment so với model đầy đủ, không phải 1. Ma trận contingency (hàng =
frozen 9-D, cột = 1-D instrumentalness) là [[585, 4], [3, 2408]]; kích thước cụm là 589/2.411 so
với 588/2.412, nên **chênh lệch thuần chỉ là +1** dù có 7 track đổi cụm (4 từ vocal sang
instrumental, 3 theo chiều ngược lại). Trên 68.813 fit IDs, cùng phép so sánh cho ARI 0,9905 với
141 track khác, xác nhận 0,9892 là giá trị trên eval cohort. Các feature khác đều thất bại: tốt
nhất là `loudness` với ARI 0,2208; `energy` 0,0289; `acousticness` 0,0131; `speechiness` −0,0501.

**Leave-one-feature-out.** Bỏ bất kỳ feature nào khác `instrumentalness` giữ ARI **≥ 0,9938** so
với model đầy đủ. Bỏ `instrumentalness` cho ARI **−0,0498** và một partition khác (6,5% / 93,5%);
partition thay thế có speechiness median 0,0465 so với 0,3430 — tức feature low-IQR tiếp theo
(`speechiness`) chiếm ưu thế. Mọi biến thể LOO đều ổn định qua 5 seed (seed ARI mean 1,0), nên
đây là thay đổi do **lựa chọn feature/biểu diễn**, không phải K-Means bất ổn.

**So sánh scaler có kiểm soát.** StandardScaler K=2/F9/noPCA/seed 42 tạo partition khác:
`energy` 29,46%, `acousticness` 26,48%, `loudness` 23,00% share tổng between; `instrumentalness`
chỉ 3,95%; single-feature ARI tốt nhất là `loudness` 0,4942 (instrumentalness 0,1300). Hai
partition chỉ đồng thuận **ARI 0,1350** trên cùng eval IDs. Silhouette 0,2576 (Standard) và
0,5956 (Robust) nằm trong hai distance geometry khác nhau, **không được xếp hạng trực tiếp**.

**Kết luận.** Đây là acoustic separation thật (raw `instrumentalness` bimodal, cụm
instrumental có 94,7% track ≥ 0,5 so với 0% ở cụm còn lại; ổn định qua seed/bootstrap), nhưng
bị **khuếch đại bởi IQR scaling** thành một phân đoạn gần như đơn biến. Không phải lỗi
implementation: số liệu tái tạo chính xác, không leakage, metric đúng. Báo cáo phải mô tả model
là **instrumental vs vocal/non-instrumental**, không phải “hai persona đa biến cân bằng”.
Chi tiết: `docs/ROBUSTNESS_ANALYSIS.md`.

---

## 8. Thảo luận

### 8.1 Hai persona instrumental/vocal, không phải sáu genre

K-Means robust K=2 tách được một trục âm học rất rõ: **instrumental/ambient** (instrumentalness
median 0,858, loudness thấp, valence thấp) và **vocal/non-instrumental năng lượng trung bình–cao**
(instrumentalness gần 0, loudness/valence/danceability cao hơn). Đây là kết quả đơn giản nhưng có
ý nghĩa và ổn định. Tuy nhiên, K=2 Standard **không phải** “cách đọc khác” của cùng partition: nó
tạo một phân hoạch khác (ARI **0,1350** với robust), theo trục `energy`/`acousticness`/`loudness`;
robust scaling làm nổi bật instrumentalness vì IQR của nó nhỏ ở phần lớn track.

Mức độ đơn biến được định lượng trong §7.12: `instrumentalness` chiếm **97,29%** between-cluster
separation và **97,29%** assignment margin; một K=2 trên riêng feature này tái tạo partition với
ARI **0,9892**. Vì vậy mô tả đúng là **instrumental vs vocal/non-instrumental**, không phải
“hai persona đa biến cân bằng”. Kết quả vẫn có giá trị cho mục tiêu phân đoạn instrumental/vocal,
nhưng không chứng minh cấu trúc đa biến cân bằng của cả chín feature.

K=6 standard cho nhiều persona hơn (acoustic, dance, high-energy, instrumental, speech-heavy, live)
và nhìn “đẹp” hơn khi thuyết trình, nhưng stability giảm mạnh (pairwise ARI 0,8696) và cụm speech
chỉ 1,14% trên fit. K=6 PCA90 ổn định hơn (0,9851) nhưng vẫn có cụm nhỏ và thêm một PCA representation. Vì
mục tiêu là segmentation có thể bảo vệ, K=2 robust được chọn; K=6 được báo cáo như trade-off.

### 8.2 K-Means vs GMM

GMM full K=4 (Standard) có BIC/AIC thấp hơn K=2 và posterior confidence cao (mean 0,9578), nhưng
silhouette hard-label chỉ 0,0547; GMM đang mô tả density, có thể dùng nhiều component để fit phân
phối phi-Gaussian mà không tạo cụm tách biệt.

So sánh controlled (audit 2026-10-27; cùng fit/eval IDs, F9, scaler, K, seed; GMM `n_init=5`,
K-Means `n_init=20`) cho 8 dòng, tất cả GMM hội tụ:

| Space | K | Cov | K-Means silhouette | GMM silhouette | ARI KM vs GMM | GMM confidence |
|---|---:|---|---:|---:|---:|---:|
| Robust | 2 | diag / full | 0,5956 | 0,2388 / 0,2401 | 0,3021 / 0,3063 | 0,9949 / 0,9965 |
| Robust | 4 | diag / full | 0,4884 | 0,0398 / 0,0674 | 0,1835 / 0,2538 | 0,9462 / 0,9628 |
| Standard | 2 | diag / full | 0,2576 | 0,1136 / 0,0961 | 0,0704 / −0,0201 | 0,9935 / 0,9510 |
| Standard | 4 | diag / full | 0,1742 | 0,0579 / 0,0547 | 0,1227 / 0,1125 | 0,9489 / 0,9577 |

Hai kết luận quan trọng:

1. **Hard partition của GMM khác K-Means** (ARI −0,0201…0,3063), dù posterior của GMM rất tự tin
   (mean max posterior 0,947–0,997). GMM tự tin về component của nó, không có nghĩa nó tái tạo
   partition K-Means.
2. **GMM không “thua” một cách tuyệt đối**: nó tối ưu likelihood/density, còn K-Means tối ưu SSE
   hard-label. Trong bảng controlled K=2/4, BIC full K=4 thấp nhất trong từng space (Standard
   853.059; Robust 819.184); baseline Standard K=2..12 vẫn giảm tới K=12 full (678.842), tức BIC
   density có thể tiếp tục giảm mà không tạo segmentation tốt hơn. BIC không được so chéo
   Standard/Robust.

Lưu ý nhỏ: baseline GMM full K=4 (`n_init=2`) có BIC 853.058; controlled `n_init=5` có 853.059;
khác biệt 1 đơn vị do tối ưu EM, không đổi kết luận.

GMM hữu ích để tìm track biên và diễn giải soft membership; K-Means robust K=2 là model chính cho
segmentation/reporting. Đây là kết luận **có số liệu**, không phải thiên vị thuật toán.

### 8.3 Scaling và BIC

E00 cho thấy nếu không scale, silhouette “cao” nhưng bị tempo/loudness chi phối và gần như chia đôi
dữ liệu theo scale, không theo persona. RobustScaler đạt silhouette cao hơn StandardScaler ở K=2
trong không gian của nó (0,5956 so với 0,2576), với stability gần 1 và profile rõ; nhưng hai
partition chỉ đồng thuận **ARI 0,1350**, và ưu thế silhouette của Robust gắn với `instrumentalness`
(97,29% between SS). StandardScaler cho một partition đa biến hơn (energy/acousticness/loudness)
nhưng silhouette thấp hơn trong space của nó. Không thể so raw silhouette giữa hai scaler như hai
model cùng không gian; báo cáo không tuyên bố “robust luôn tốt hơn”.

BIC của GMM giảm tới K=12, nhưng silhouette hard-label âm ở full K=7..12. Đây là ví dụ kinh điển:
BIC chọn density fit, không chọn semantic cluster. Việc không mở rộng K thể hiện kỷ luật chống
overfitting/scope creep.

### 8.4 Genre và multi-label

ARI 0,004/NMI 0,082 rất thấp. Kết quả này không phủ định clustering: cluster instrumental tập trung
ở study/sleep/minimal-techno/ambient/classical, trong khi cluster vocal trải rộng trên pop/rock/
reggae/pagode/v.v. Genre là taxonomy thương mại, không phải trục acoustic; nhiều track mang nhiều
genre; mỗi genre chỉ khoảng 1.000 track. Báo cáo không gọi cluster là genre, không dùng genre để
chọn K, và công khai cohort single-label/omitted.

### 8.5 PCA và t-SNE

PCA90 (7 component, 94,82%) cải thiện nhẹ silhouette K=6 (0,2033 vs 0,1896) và K=2 (0,2673 vs
0,2576), nhưng lợi ích nhỏ so với mất tính trực tiếp của loadings. No-PCA được chọn. PCA2D/t-SNE
chỉ để nhìn cấu trúc; hình t-SNE không được dùng để khẳng định “hai cụm xa nhau”. Ward trên mẫu
với **frozen transformer** đồng thuận ARI **0,8834** với K-Means (run
`20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr`); con số 0,969205 của run lịch sử
bị loại khỏi tuyên bố controlled vì run đó refit scaler trên mẫu. ARI 0,8834 vẫn củng cố cấu trúc
phân đôi instrumental/vocal, nhưng không phải generalization metric.

---

## 9. Hạn chế và threats to validity

### 9.1 Construct validity

- Spotify audio features là **descriptor do thuật toán tạo**, không phải raw waveform; chúng có thể
  bỏ qua timbre, harmony, lyrics và cảm nhận người nghe. `instrumentalness` đặc biệt là một ước
  lượng của Spotify, không phải phép đo trực tiếp.
- “Audio persona” là giả thuyết từ thống kê, không được xác nhận bằng listening study. Với K=2
  RobustScaler, “persona” thực chất là **instrumental vs vocal/non-instrumental**; mô hình gần
  như đơn biến theo `instrumentalness` (97,29% between SS), không phải cấu trúc đa biến cân bằng.
- Genre labels không đồng nhất với acoustic structure; multi-label được xử lý riêng.

### 9.2 Internal validity

- Không có nhãn ground-truth nên không thể báo accuracy/F1 cho cluster.
- Silhouette/DB/CH ưa geometry cầu/ellipsoid; K-Means có thể tạo cụm tách biệt giả tạo nếu scale
  sai. Chúng tôi dùng cohort ngoài fit, nhiều metric, stability và profile để giảm rủi ro.
- Kết quả phụ thuộc representation: F5/F9, scaler, PCA khác nhau; báo cáo không pool metric/BIC
  giữa các space. StandardScaler K=2 và RobustScaler K=2 chỉ đồng thuận **ARI 0,1350** trên cùng
  eval IDs; đây là hai phân đoạn khác nhau, không phải hai cách đọc của một partition.
- Feature dominance: bỏ `instrumentalness` đổi partition (ARI −0,0498) trong khi bỏ bất kỳ feature
  nào khác giữ ARI ≥ 0,9938; kết luận K=2 phụ thuộc vào một feature.
- Final selection dùng development evidence; holdout chỉ kiểm tra một lần. Không có guarantee
  ngoài phân phối Spotify snapshot. Phân tích robustness 2026-10-27 là **hậu nghiệm/exploratory**,
  không phải một holdout thứ hai.

### 9.3 External validity

- Snapshot 114.000 dòng, 114 genre; mỗi genre có khoảng 1.000 track, không đại diện thị trường.
- Không có track mới/album mới/hành vi người dùng; kết quả không phải recommender.
- K=2 rất rộng; các nhóm nhỏ (speech-heavy, live) bị gộp. Nếu mục tiêu là phân đoạn chi tiết,
  K=6 là ứng viên nhưng phải chấp nhận stability thấp hơn.

### 9.4 Statistical validity

- K-Means local optimum được giảm bằng k-means++/n_init=20/5 seed, nhưng không loại bỏ hoàn toàn.
- GMM full covariance có thể singular ở component nhỏ; reg_covar và convergence được log. Trong
  so sánh controlled, hard partition của GMM khác K-Means (ARI −0,0201…0,3063); posterior tự tin
  (0,947–0,997) nhưng không tái tạo K-Means partition.
- Silhouette trên 3.000 mẫu có phương sai lấy mẫu; stability ARI chỉ đo assignment, không đo
  semantic usefulness.
- Bootstrap tái sử dụng hàng, có thể tạo duplicate track trong fit; đây là chủ đích và được ghi.
- t-SNE không định lượng; Ward chỉ trên 2.000 mẫu, sample-only, không có `.predict`.
- Phân tích feature dominance/LOO là **hậu nghiệm, exploratory**, không có p-value hay hiệu chỉnh
  multiplicity; các con số được báo cáo như mô tả định lượng, không phải kiểm định suy diễn.
- K=2 phụ thuộc vào một feature: bỏ `instrumentalness` đổi partition (ARI −0,0498), trong khi bỏ
  feature khác giữ ARI ≥ 0,9938.

### 9.5 Compliance / reproducibility

- License chưa xác minh; raw CSV **không được public**.
- Không có git commit/remote; CI workflow chưa chạy trên GitHub. Local tests/lint/CLI/verify đã chạy
  trên **63/63 run**; clean-clone install sau audit đã chạy lại **PASS** ngày 2026-10-27 (venv mới,
  50 tests, ruff sạch, CLI OK).
- Determinism trong môi trường hiện tại; BLAS/platform khác có thể khác số thực nhỏ.
- Bốn deck gốc đã được đối chiếu trực tiếp ngày 2026-10-27; mapping §11 đã sửa ba điểm khác biệt
  (GMM/EM, leakage, Gini) và không còn phụ thuộc mô tả trung gian của SSOT.
- Bộ slide bảo vệ **12 slide** (`deliverables/BTL11_defense.pptx`) đã được cập nhật một caveat
  feature-dominance ở slide kết luận, lấy từ `artifacts/robustness/20261027T070249Z/`; không đổi
  số slide/thiết kế. Bản render PDF 12 trang đã có (`deliverables/BTL11_defense.pdf`); rehearsal
  visual/timing vẫn là cổng con người.
- Không có rubric chính thức; báo cáo không tuyên bố điểm.

---

## 10. Kết luận

1. Pipeline unsupervised hoàn chỉnh đã chạy trên đúng dataset: audit, dedup 89.741 track, split
   ID-disjoint, F5/F9, scaler, PCA, K-Means, GMM, validity, stability, holdout, refit, profile,
   PCA2D/t-SNE, Ward, genre post-hoc, feature-dominance/LOO robustness, controlled GMM comparison,
   CLI, tests và verify (63/63 run).
2. K-Means robust F9 K=2 là model cuối được khóa trước holdout. Nó tạo hai nhóm ổn định:
   instrumental/ambient (19,53%) và vocal/non-instrumental năng lượng trung bình–cao (80,47%).
   **Tuy nhiên**, đây là phân đoạn gần như đơn biến theo `instrumentalness` (97,29% between SS;
   single-feature ARI 0,9892), không phải cấu trúc chín chiều cân bằng. Mô tả đúng là
   **instrumental vs vocal/non-instrumental**.
3. Holdout xác nhận cấu trúc: silhouette 0,5902 so với 0,5956 dev; DB 0,6593 so với 0,6505;
   stability bootstrap 0,9991. Đây là sanity check, không phải accuracy; holdout không được dùng
   lại cho phân tích robustness.
4. GMM BIC thấp ở K cao nhưng hard-label separation kém; BIC không được dùng một mình. So sánh
   controlled cho thấy hard partition của GMM khác K-Means (ARI −0,0201…0,3063) dù posterior tự
   tin; GMM là density model/complement, không thay thế segmentation. K=6 cho profile giàu hơn
   nhưng kém ổn định; trade-off được công khai.
5. Genre không phải target: ARI 0,004/NMI 0,082; cluster instrumental tập trung ở
   study/sleep/minimal-techno/ambient, cluster vocal trải rộng. Kết luận là **instrumental/vocal
   acoustic personas**, không phải “genre discovery” và không phải “balanced multivariate
   personas”.
6. Giới hạn lớn nhất không phải code mà là **license/redistribution** và việc cần con người xác
   minh trước khi public; raw CSV chưa được công bố. Ward controlled đúng là ARI 0,8834 (không
   phải 0,9692 của run refit-scaler lịch sử); đây là sample-only diagnostic.

---

## 11. Traceability tới bốn deck môn học

> Bốn file `Data.pptx`, `Cluster Analysis.pptx`, `Classification.pptx`, `Pattern Mining.pptx` đã được
> đối chiếu trực tiếp ngày **2026-10-27** với bản lưu trong thư mục môn học. Bảng dưới ghi nội dung
> quan sát được (số slide 1-based), không phải suy đoán. Ba điểm lệch của bản nháp trước đã được sửa:
> (i) deck *Cluster Analysis* chỉ trình bày ý tưởng generative/likelihood, **không** trình bày đầy đủ
> GMM/EM; GMM/EM là yêu cầu của đề bài và được dẫn từ tài liệu chuẩn; (ii) phần evaluation của
> *Classification* là metric, over/underfitting, holdout, cross-validation và ROC, không bàn leakage;
> (iii) decision tree trong deck dùng information gain/gain ratio, không nhấn Gini.

| Deck / slide (đã kiểm tra) | Nội dung thực tế | Bằng chứng trong repo |
|---|---|---|
| Data 6–11 | data objects, attributes, attribute types, dimensionality/sparsity | schema 9 feature số, audit kiểu dữ liệu |
| Data 14–39 | central tendency, dispersion, skew, covariance/correlation, boxplot/histogram | EDA median/IQR/skew, correlation matrix |
| Data 41–59 | similarity/dissimilarity, z-score standardization, Minkowski/Euclidean, cosine, KL | Standard/Robust rationale, Euclidean distance |
| Data 60–79 | data quality, cleaning, transformation, min-max normalization, discretization/binning | audit, dedup, quality flags; không binning trong model |
| Data 86–108 | sampling, data reduction, PCA, SNE/nonlinear reduction | split, PCA90/95, PCA2D; t-SNE dùng cho hiển thị |
| Cluster 7–17 | cluster analysis definition, applications, requirements, methodologies | problem statement, thiết kế unsupervised |
| Cluster 19–25 | partitioning, K-Means objective, initialization và k-means++ | công thức, K sweep, `n_init=20`, seed stability |
| Cluster 26–35 | K-Medoids/PAM, K-Medians, K-Modes, kernel K-Means | không dùng; scope trade-off đã ghi |
| Cluster 37–45 | hierarchical/agglomerative, dendrogram, Ward criterion | Ward sample + dendrogram + ARI |
| Cluster 58–61 | probabilistic/generative clustering, likelihood-based cluster quality | động cơ cho GMM; GMM/EM/BIC lấy từ đề bài + tài liệu chuẩn, không từ deck |
| Cluster 63–73 | DBSCAN/OPTICS, tham số và reachability | không bắt buộc; deferred P2 |
| Cluster 84–103 | clustering tendency, K selection, elbow, extrinsic (matching, entropy/NMI, Jaccard), intrinsic (Dunn, silhouette) | multi-metric selection, stability, genre ARI/NMI |
| Classification 2–6 | supervised vs unsupervised, model construction/validation | chọn unsupervised; không train classifier |
| Classification 8–18 | decision-tree induction, information gain/gain ratio | deferred P2; không cần cho core |
| Classification 52–62 | evaluation metrics, over/underfitting, holdout, cross-validation, ROC | holdout discipline; không báo accuracy cho cluster |
| Pattern Mining 5–20 | transaction database, itemsets/support, association rules, Apriori | ngoài core; deferred P2 |
| Pattern Mining 26–47 | ECLAT, FP-Growth, lift/χ²/Kulczynski, null-invariance | ngoài core; deferred P2 |

---

## 12. Tái lập, artifact và phụ lục

### 12.1 Môi trường

Python 3.13.16; numpy 2.5.4; pandas 3.1.0; scipy 1.18.1; scikit-learn 1.9.1;
matplotlib 3.11.2; seaborn 0.13.2; joblib 1.6.0; pyarrow 26.0.0. Exact lock:
`requirements.lock`. CI dùng Python 3.12; code tương thích 3.11+.

### 12.2 Run ID chính

| Mục | Run ID / đường dẫn |
|---|---|
| Frozen evaluation | `20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_robust_nopca_k2_s42` |
| Holdout | `20261027T032820Z_f7434a8c_b202fa49_holdout_20261027T032109Z_07fbb11b_b202fa49_kmeans_f9_rob` |
| Final refit | `20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2` |
| Stability | `20261027T032241Z_ea0570dc_b202fa49_stability` |
| Ward (historical, sample-refit scaler) | `20261027T032907Z_f7434a8c_b202fa49_ward_diagnostic` |
| Ward (corrected, frozen transformer) | `20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr` |
| Robustness / dominance | `artifacts/robustness/20261027T070249Z/` |
| Controlled GMM comparison | `artifacts/comparisons/exploratory_controlled_gmm/20261027T065934Z/` |
| Baseline comparison (cohort-aware) | `artifacts/comparisons/baseline_f9_standard/20261027T070434Z/` |
| EDA | `artifacts/eda/20261027T031519Z/` |
| PCA | `artifacts/pca/20261027T031522Z/` |
| Report tables | `artifacts/final-report-tables/` |

### 12.3 Hình

- EDA: `artifacts/eda/20261027T031519Z/figures/`
- PCA diagnostics: `artifacts/pca/20261027T031522Z/figures/`
- Baseline: `artifacts/comparisons/baseline_f9_standard/20261027T070434Z/` (elbow, metrics, BIC/AIC)
- Ablations: `artifacts/ablations/{scaler,feature_set,pca}/*/metrics_by_variant.png`
- Stability: `artifacts/runs/20261027T032241Z_ea0570dc_b202fa49_stability/figures/stability.png`
- Final core: `artifacts/runs/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/figures/`
  (`profile_heatmap.png`, `genre_cluster_heatmap.png`, `cluster_sizes_all.png`)
- Post-hoc: `artifacts/posthoc/20261027T032943Z_f7434a8c_b202fa49_final_refit_kmeans_k2/figures/`
  (`pca2_scatter.png`, `tsne2d.png`)
- Holdout: `artifacts/runs/20261027T032820Z_.../figures/cluster_sizes_holdout.png`
- Ward corrected: `artifacts/runs/20261027T070309Z_f7434a8c_b202fa49_ward_diagnostic_frozen_repr/figures/ward_dendrogram.png`
- Robustness: `artifacts/robustness/20261027T070249Z/figures/feature_dominance.png`,
  `.../figures/loo_ari.png`
- Bộ slide bảo vệ: `deliverables/BTL11_defense.pptx` (12 slide; caveat feature-dominance ở slide
  kết luận; số liệu lấy từ `artifacts/final-report-tables/REPORT_TABLES.md`, các run ID ở §12.2 và
  `artifacts/robustness/20261027T070249Z/`).

### 12.4 Checklist trung thực

```text
[x] Dataset đúng, SHA-256 ghi nhận
[x] Unit = unique track_id, đa genre lưu riêng
[x] Không conflict bị che giấu (0 conflict thực tế)
[x] Split ID-disjoint, eval cohort ngoài fit
[x] Genre/metadata không vào X (test + bundle feature names)
[x] Scaler/PCA fit chỉ trên fit
[x] Silhouette/DB/CH cùng cohort; null + reason khi invalid
[x] BIC không so chéo representation; cohort_id trong comparison
[x] Stability ARI cùng reference IDs, permutation invariant; verify tính lại mean ARI
[x] Freeze trước holdout; holdout dùng một lần; không mở lại cho audit
[x] Final refit là model identity riêng
[x] Profiles đơn vị gốc + representative tracks + caveat
[x] Genre post-hoc multi-label, single-label cohort công khai
[x] Feature dominance/LOO robustness (development-only, exploratory)
[x] Controlled K-Means vs GMM cùng ID/feature/scaler/K/seed + ARI
[x] Ward corrected dùng frozen transformer (historical run được giữ, ghi rõ superseded)
[x] Verify hash/model/prediction/metric; 63/63 run PASS
[x] Tests + lint + CLI smoke; 50 tests pass
[x] Không commit/push raw data hoặc credentials
[ ] License/redistribution: chờ con người xác minh
[x] So khớp bốn slide gốc trước khi nộp (2026-10-27; §11)
[ ] CI remote: chưa chạy (không push)
[x] Clean-clone install sau audit: PASS 2026-10-27 (venv mới, 50 tests, ruff, CLI)
[x] Visual render deck: PDF 12 trang (LibreOffice)
[ ] Rehearsal visual/timing deck: cổng con người
```

---

## Tài liệu tham khảo

1. `docs/BTL11_SSOT.md` — normative SSOT và traceability.
2. MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations*.
3. Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding*. SODA.
4. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm*. JRSS-B. <https://doi.org/10.1111/j.2517-6161.1977.tb01600.x>
5. Rousseeuw, P. J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis*.
6. Davies, D. L., & Bouldin, D. W. (1979). *A Cluster Separation Measure*. IEEE TPAMI. <https://doi.org/10.1109/TPAMI.1979.4766909>
7. Calinski, T., & Harabasz, J. (1974). *A Dendrite Method for Cluster Analysis*.
8. Ward, J. H. (1963). *Hierarchical Grouping to Optimize an Objective Function*.
9. van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. JMLR. <https://www.jmlr.org/papers/v9/vandermaaten08a.html>
10. scikit-learn documentation: KMeans, GaussianMixture, PCA, t-SNE, silhouette, Davies–Bouldin, Calinski–Harabasz, AgglomerativeClustering.
11. Kaggle — Spotify Tracks Dataset. <https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset>
12. Hugging Face mirror. <https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset>
13. Spotify Developer Policy. <https://developer.spotify.com/policy>
14. Spotify developer update 2026-02-06. <https://developer.spotify.com/blog/2026-02-06-update-on-developer-access-and-platform-security>

---

*Báo cáo chỉ chứa kết quả chạy thực tế ngày 2026-10-27 trên dataset đã nêu. Các tuyên bố lý thuyết
không có artifact được ghi rõ là lý thuyết. Không có số liệu giả, không có run ID giả, không có
license approval giả định.*
