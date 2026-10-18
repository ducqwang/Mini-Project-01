# Mini Project 01: E-ELAN Architecture for Image Classification & Object Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Google Colab](<https://img.shields.io/badge/Run%20in-Google%20Colab-f9ab00.svg>)](https://colab.research.google.com/)

> **Học phần:** Một số vấn đề chọn lọc về Thị giác máy tính (MAT3563)
> **Khoa:** Toán - Cơ - Tin học, Trường Đại học Khoa học Tự nhiên – ĐHQGHN
> **Giảng viên hướng dẫn:** TS. Cao Văn Chung

---

## 👥 Nhóm sinh viên thực hiện

| STT | Họ và tên                  |   Mã sinh viên   |                   Lớp                   | Nhiệm vụ chính                                                                                                                                                                                                                                                               |
| :-: | :---------------------------- | :----------------: | :---------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  1  | **Nguyễn Đức Quang** | **23001918** | K68A4 Khoa học máy tính và thông tin | **Task 1 & 2:** Xây dựng Backbone E-ELAN, cơ chế lưu/nạp weights, inference ảnh đơn & thư mục, xuất CSV, tính Precision/Recall, vẽ Confusion Matrix, vẽ đồ thị biến thiên Loss & Accuracy.                                                            |
|  2  | **Hà Minh Quang**      | **23001916** | K68A4 Khoa học máy tính và thông tin | **Task 3:** Thu thập & chuẩn hóa dữ liệu Pascal VOC + Panda Detection (Kaggle) về chuẩn YOLO; thiết kế ghép nối E-ELAN Backbone (nạp tiền huấn luyện, tinh chỉnh end-to-end) + SimpleFPN + SSD Head; Anchor matching đa tỉ lệ, Multibox Loss (Hard Negative Mining 3:1), giải thuật NMS, đánh giá mAP@0.5. |

---

## 📌 Tổng quan dự án

Dự án nghiên cứu và phân tích khối kiến trúc **E-ELAN** (*Extended Efficient Layer Aggregation Network*) trong **YOLOv7**, từ đó ứng dụng để giải quyết hai bài toán thị giác máy tính:

1. **Phân loại ảnh (Image Classification):** Xây dựng mạng phân loại 3 lớp (`cats`, `dogs`, `panda`), bổ sung cơ chế lưu/nạp trọng số, chạy suy luận đơn và hàng loạt (xuất kết quả xác suất ra CSV), đánh giá định lượng Precision, Recall, F1-score và ma trận nhầm lẫn.
2. **Phát hiện đối tượng (Object Detection):** Tái sử dụng chính Backbone E-ELAN đã huấn luyện (nạp `backbone.pt` làm khởi tạo học chuyển giao, sau đó tinh chỉnh toàn bộ end-to-end — `freeze_backbone = False`), ghép nối với phần **Neck** (*SimpleFPN*) và phần **Head** (*SSD Head*) một giai đoạn để phát hiện 3 lớp đối tượng trên tập dữ liệu tự thu thập và chuẩn hóa.

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Input Image (320x320x3)                  │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                 E-ELAN BACKBONE (Pretrained từ NB01, Fine-tuned end-to-end)                │
│       Stem (s=2) ──► Stage 1 (s=4) ──► Stage 2 (s=8, C3) ──► Stage 3 (s=16, C4) ──► Stage 4 (s=32, C5) │
└───────────────────────────┬───────────────────┬───────────────────┬────────────────────────┘
                            │ C3                │ C4                │ C5
                            ▼                   ▼                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                       SIMPLE FPN NECK (Top-Down Pathway)                                   │
│            P3 (stride 8)       ◄───────  P4 (stride 16)      ◄───────  P5 (stride 32)      │
└─────────────────────┬─────────────────────────┬───────────────────────────┬────────────────┘
                      │                         │                           │
                      ▼                         ▼                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                    SSD MULTI-SCALE HEAD (Shared Convolutions)                              │
│   • Class Branch:  Scores (Background + 3 Classes)                                         │
│   • Box Branch:    Offsets (Δx, Δy, Δw, Δh) cho 6 Anchors/Cell (Tổng 12.600 Anchors)       │
└───────────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │          Decode Offsets + NMS (IoU=0.45)               │
                    │       ==> Final Detections (Class, Conf, BBoxes)       │
                    └────────────────────────────────────────────────────────┘
```

---

## 📂 Cấu trúc thư mục

```text
Mini Project 01/
├── notebooks/
│   ├── 01_classification_eelan.ipynb    # Task 1 + 2: Huấn luyện phân loại, lưu/nạp weights, inference, metrics
│   └── 02_detection_eelan.ipynb         # Task 3: Build dataset, ghép FPN + SSD, Multibox loss, NMS, mAP@0.5
├── data/
│   ├── classification/                  # Dữ liệu phân loại (train, val chia theo thư mục lớp)
│   │   ├── train/ {cats, dogs, panda}
│   │   └── val/   {cats, dogs, panda}
│   └── detection/                       # Dữ liệu phát hiện đối tượng chuẩn định dạng YOLO
│       ├── images/ {train, val}/        # Ảnh (.jpg, .png)
│       ├── labels/ {train, val}/        # Nhãn bounding box (.txt)
│       └── classes.txt                  # Danh sách 3 lớp: cat, dog, panda
├── weights/                             # Nơi lưu trữ trọng số mô hình
│   ├── best_cls.pt                      # Trọng số tốt nhất của mô hình phân loại
│   ├── backbone.pt                      # Trọng số phần Backbone E-ELAN (dùng cho detection)
│   └── best_det.pt                      # Trọng số tốt nhất của mô hình phát hiện đối tượng
├── results/                             # Kết quả thực nghiệm và trực quan hóa (đã trích từ output notebook)
│   ├── cls_training_curves.png          # Đồ thị biến thiên Loss & Accuracy qua 20 epoch
│   ├── cls_confusion_matrix.png         # Ma trận nhầm lẫn phân loại trên tập val
│   ├── cls_inference_samples.png        # Lưới ảnh minh họa suy luận phân loại (nhãn + xác suất)
│   ├── det_loss_curve.png               # Đồ thị biến thiên Multibox Loss khi train detection
│   ├── det_vis_grid.png                 # Lưới 6 ảnh phát hiện (bbox + nhãn + confidence)
│   ├── det_vis_sample_01..05.png        # 5 ảnh phát hiện riêng lẻ ở độ phân giải đầy đủ
│   ├── (khi chạy lại) cls_precision_recall.txt, cls_predictions.csv, cls_history.csv
│   └── (khi chạy lại) det_metrics.txt, det_predictions.csv, det_vis/
├── report/
│   └── main.tex                         # Báo cáo học thuật LaTeX chuẩn 5 chương (không chứa mã code)
├── CNN_ELAN_Class.ipynb                 # Mã nguồn minh họa ban đầu của giảng viên
├── Advanced_CV.pdf                      # Tài liệu lý thuyết về khối E-ELAN (YOLOv7)
├── HUONG_DAN_CHAY_THUC_NGHIEM.txt       # Hướng dẫn chi tiết chạy & chấm thực nghiệm (nguồn, dữ liệu, cấu hình)
├── HUONG_DAN_DATA_KAGGLE_COLAB.txt      # Hướng dẫn thêm dataset & chạy trên Kaggle/Colab/Local
├── README.md                            # Tài liệu tổng quan dự án (file này)
└── .gitignore                           # Quản lý các tệp loại trừ của Git
```

---

## ⚙️ Cài đặt & Yêu cầu môi trường

### 1. Yêu cầu hệ thống

- **Hệ điều hành:** Linux, macOS, hoặc Windows
- **Python:** $\ge 3.9$
- **Khuyến nghị phần cứng:** GPU NVIDIA (hỗ trợ CUDA) với VRAM $\ge 8\text{ GB}$. Toàn bộ thực nghiệm trong báo cáo được chạy trên **Kaggle Notebooks (GPU NVIDIA Tesla T4 16GB)**; có thể chạy tương đương trên **Google Colab GPU T4**.

### 2. Cài đặt thư viện

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pillow opencv-python scikit-learn tqdm
```

---

## 🚀 Hướng dẫn thực thi

Mã nguồn trong cả hai notebook đều tích hợp sẵn cơ chế **tự động nhận diện môi trường** (biến `ENV` ở cell đầu):

- Khi chạy trên **Kaggle** (`ENV='kaggle'`): Đọc dữ liệu tại `/kaggle/input`, ghi weights/results tại `/kaggle/working/MiniProject01`. Đây là môi trường đã dùng để tạo kết quả trong báo cáo.
- Khi chạy trên **Google Colab** (`ENV='colab'`): Tự động mount Google Drive và trỏ vào thư mục `/content/drive/MyDrive/MiniProject01`.
- Khi chạy **Local** (`ENV='local'`): Tự động trỏ vào đường dẫn cục bộ dự án.

> Chi tiết cách thêm dataset và chạy trên từng nền tảng xem tại [`HUONG_DAN_DATA_KAGGLE_COLAB.txt`](HUONG_DAN_DATA_KAGGLE_COLAB.txt) và [`HUONG_DAN_CHAY_THUC_NGHIEM.txt`](HUONG_DAN_CHAY_THUC_NGHIEM.txt).

### Bước 1: Huấn luyện phân loại ảnh & Trích xuất Backbone (Notebook 01)

1. Mở `notebooks/01_classification_eelan.ipynb`.
2. Kiểm tra dữ liệu tại `data/classification/` (bộ ảnh Kaggle Dog, Cat, Panda).
3. Chạy toàn bộ notebook (**Run All**):
   - Mô hình huấn luyện qua 20 epochs với optimizer AdamW và Cosine Annealing LR.
   - Xuất trọng số `weights/best_cls.pt` (val acc tốt nhất) và tách riêng trọng số `weights/backbone.pt`.
   - Lưu đồ thị `results/cls_training_curves.png`, ma trận nhầm lẫn `results/cls_confusion_matrix.png`, bảng chỉ số `results/cls_precision_recall.txt` và lịch sử `results/cls_history.csv`.
4. **Kiểm tra suy luận (Inference):**
   - Đặt đường dẫn ảnh hoặc thư mục vào biến `INFER_PATH` ở cell cuối cùng để chạy suy luận và xuất kết quả ra `results/cls_predictions.csv`.

### Bước 2: Huấn luyện mô hình Phát hiện đối tượng (Notebook 02)

1. Mở `notebooks/02_detection_eelan.ipynb`.
2. Yêu cầu tiên quyết: Đã có tệp `weights/backbone.pt` từ Bước 1.
3. Tại Mục 2 (Xây dựng dữ liệu):
   - Tự động tải và trích xuất nhãn `cat`, `dog` từ Pascal VOC 2007 (`BUILD_VOC = True`, cần bật Internet).
   - Nạp dữ liệu `panda` từ tập **Panda Detection (Kaggle, định dạng YOLO)** để đồng bộ về chuẩn YOLO 3 lớp.
4. Chạy huấn luyện (**Run All**):
   - Nạp trọng số `backbone.pt` làm khởi tạo; cấu hình `freeze_backbone = False` nên Backbone được tinh chỉnh cùng Neck + Head (tổng 3.41M tham số).
   - Huấn luyện qua 30 epochs bằng Multibox Loss kết hợp Hard Negative Mining (3:1); chọn checkpoint tốt nhất theo val loss.
   - Xuất trọng số `weights/best_det.pt`.
   - Đánh giá $\text{mAP}@0.5$ (lưu `results/det_metrics.txt`), vẽ bounding box trực quan hóa vào `results/det_vis/` và xuất kết quả ra `results/det_predictions.csv`.

---


## 📊 Tóm tắt kết quả thực nghiệm

### 1. Kết quả Phân loại ảnh (Task 1 & Task 2)

- **Dữ liệu:** 3.000 ảnh (1.000 ảnh/lớp), tách ngẫu nhiên 90/10 → val 300 ảnh (cats 90, dogs 109, panda 101).
- **Độ chính xác kiểm định cao nhất:** **85.67%** (tại epoch 18).
- **Diễn biến huấn luyện (20 epoch):** train loss $0.9372 \rightarrow 0.2507$, train acc $53.96\% \rightarrow 89.19\%$.
- **Chi tiết từng lớp:**
  - `panda`: Precision **0.980**, Recall **0.980**, F1 **0.980** (đặc trưng lông tương phản đen-trắng rõ nét).
  - `dogs`: Precision **0.815**, Recall **0.807**, F1 **0.811**.
  - `cats`: Precision **0.769**, Recall **0.778**, F1 **0.774** (lớp yếu nhất, nhầm lẫn chéo với `dogs`).
  - **Macro Average F1:** **0.855**.

### 2. Kết quả Phát hiện đối tượng (Task 3)

- **Dữ liệu:** 1.049 ảnh / 1.245 bbox (VOC 2007 cat+dog, Panda Detection), tách train 877 / val 172 ảnh.
- **Độ đo toàn diện:** **$\text{mAP}@0.5 = 0.512$**
- **Average Precision theo từng lớp (IoU = 0.5):**
  - `panda`: $\text{AP} = \mathbf{0.829}$ (P = 0.573, R = 0.896)
  - `cat`: $\text{AP} = \mathbf{0.432}$ (P = 0.325, R = 0.641)
  - `dog`: $\text{AP} = \mathbf{0.275}$ (P = 0.269, R = 0.589)
- **Nhận xét:** Khởi tạo Backbone từ NB01 giúp loss huấn luyện giảm rất nhanh ($13.83 \rightarrow 1.98$). Vì `freeze_backbone = False`, mô hình có xu hướng quá khớp nhẹ sau epoch 18 (val loss chạm đáy $4.28$) nên chọn checkpoint tốt nhất theo val loss. Hiệu năng phân hóa mạnh giữa `panda` và `cat`/`dog` do dữ liệu nhỏ, mất cân bằng và đối tượng VOC biến động lớn.

---

## 📖 Báo cáo học thuật LaTeX

Báo cáo đầy đủ được biên soạn bằng LaTeX tại thư mục [`report/main.tex`](report/main.tex), tuân thủ nghiêm ngặt quy định không đưa mã nguồn vào bài làm:

- **Trang bìa trang trọng:** Khung viền đôi chuẩn mẫu HUS - ĐHQGHN.
- **5 Chương chuyên sâu:**
  1. *Chương 1:* Đặt vấn đề và Mục tiêu nghiên cứu.
  2. *Chương 2:* Phương pháp và Kiến trúc mô hình chi tiết (toán học E-ELAN, sơ đồ TikZ, SimpleFPN, SSD Head, Anchor Matching, Multibox Loss, NMS).
  3. *Chương 3:* Dữ liệu và Quy trình chuẩn bị.
  4. *Chương 4:* Thực nghiệm và Đánh giá nhận xét.
  5. *Chương 5:* Kết luận và Hướng phát triển.
- Tự động hiển thị placeholder nếu thiếu tệp ảnh kết quả, đảm bảo biên dịch PDF thành công trên mọi hệ thống (Overleaf, Prism, TeXLive).

---

## 📚 Tài liệu tham khảo

1. Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao. *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors*. CVPR 2023.
2. Chien-Yao Wang et al. *Designing Network Design Strategies Through Gradient Path Analysis (ELAN)*. arXiv 2022.
3. Wei Liu et al. *SSD: Single Shot MultiBox Detector*. ECCV 2016.
4. Tsung-Yi Lin et al. *Feature Pyramid Networks for Object Detection*. CVPR 2017.
5. Mark Everingham et al. *The PASCAL Visual Object Classes (VOC) Challenge*. IJCV 2010.
6. TS. Cao Văn Chung. *Bài giảng Một số vấn đề chọn lọc về Thị giác máy tính (MAT3563)*. Trường ĐHKHTN, ĐHQGHN, 2026.
