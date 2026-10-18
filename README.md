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
|  2  | **Hà Minh Quang**      | **23001916** | K68A4 Khoa học máy tính và thông tin | **Task 3:** Thu thập & chuẩn hóa dữ liệu Pascal VOC + Roboflow về chuẩn YOLO; thiết kế ghép nối E-ELAN Backbone (frozen) + SimpleFPN + SSD Head; Anchor matching đa tỉ lệ, Multibox Loss (Hard Negative Mining 3:1), giải thuật NMS, đánh giá mAP@0.5. |

---

## 📌 Tổng quan dự án

Dự án nghiên cứu và phân tích khối kiến trúc **E-ELAN** (*Extended Efficient Layer Aggregation Network*) trong **YOLOv7**, từ đó ứng dụng để giải quyết hai bài toán thị giác máy tính:

1. **Phân loại ảnh (Image Classification):** Xây dựng mạng phân loại 3 lớp (`cats`, `dogs`, `panda`), bổ sung cơ chế lưu/nạp trọng số, chạy suy luận đơn và hàng loạt (xuất kết quả xác suất ra CSV), đánh giá định lượng Precision, Recall, F1-score và ma trận nhầm lẫn.
2. **Phát hiện đối tượng (Object Detection):** Tái sử dụng chính Backbone E-ELAN đã huấn luyện (ở chế độ đóng băng trọng số), ghép nối với phần **Neck** (*SimpleFPN*) và phần **Head** (*SSD Head*) một giai đoạn để phát hiện 3 lớp đối tượng trên tập dữ liệu tự thu thập và chuẩn hóa.

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Input Image (320x320x3)                  │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                       E-ELAN BACKBONE (Pretrained & Frozen)                                │
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
├── results/                             # Kết quả thực nghiệm và trực quan hóa
│   ├── cls_training_curves.png          # Đồ thị biến thiên Loss & Accuracy qua các epoch
│   ├── cls_confusion_matrix.png         # Ma trận nhầm lẫn phân loại
│   ├── cls_precision_recall.txt         # Bảng số liệu chi tiết Precision, Recall, F1
│   ├── cls_predictions.csv              # Kết quả suy luận phân loại (Image, Predict, Probabilities)
│   ├── det_loss_curve.png               # Đồ thị biến thiên Multibox Loss khi train detection
│   ├── det_metrics.txt                  # Bảng số liệu AP@0.5 từng lớp và mAP@0.5
│   ├── det_predictions.csv              # Kết quả suy luận detection (BBoxes, Class, Confidence)
│   └── det_vis/                         # Ảnh kết quả trực quan hóa sau khi vẽ bounding box
├── report/
│   └── main.tex                         # Báo cáo học thuật LaTeX chuẩn 5 chương (không chứa mã code)
├── CNN_ELAN_Class.ipynb                 # Mã nguồn minh họa ban đầu của giảng viên
├── Advanced_CV.pdf                      # Tài liệu lý thuyết về khối E-ELAN (YOLOv7)
├── README_run.txt                       # Hướng dẫn chi tiết chạy thực nghiệm bằng tiếng Việt có dấu
├── README.md                            # Tài liệu tổng quan dự án (file này)
└── .gitignore                           # Quản lý các tệp loại trừ của Git
```

---

## ⚙️ Cài đặt & Yêu cầu môi trường

### 1. Yêu cầu hệ thống

- **Hệ điều hành:** Linux, macOS, hoặc Windows
- **Python:** $\ge 3.9$
- **Khuyến nghị phần cứng:** GPU NVIDIA (hỗ trợ CUDA) với VRAM $\ge 8\text{ GB}$ (khuyên dùng **Google Colab GPU T4**).

### 2. Cài đặt thư viện

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pillow opencv-python scikit-learn tqdm
```

---

## 🚀 Hướng dẫn thực thi

Mã nguồn trong cả hai notebook đều tích hợp sẵn cơ chế **tự động nhận diện môi trường**:

- Khi chạy trên **Google Colab**: Tự động mount Google Drive và trỏ vào thư mục `/content/drive/MyDrive/MiniProject01`.
- Khi chạy **Local**: Tự động trỏ vào đường dẫn cục bộ dự án.

### Bước 1: Huấn luyện phân loại ảnh & Trích xuất Backbone (Notebook 01)

1. Mở `notebooks/01_classification_eelan.ipynb`.
2. Kiểm tra dữ liệu tại `data/classification/` (bộ ảnh Kaggle Dog, Cat, Panda).
3. Chạy toàn bộ notebook (**Run All**):
   - Mô hình huấn luyện qua 20 epochs với optimizer AdamW và Cosine Annealing LR.
   - Xuất trọng số `weights/best_cls.pt` và tách riêng trọng số `weights/backbone.pt`.
   - Lưu đồ thị `results/cls_training_curves.png` và ma trận nhầm lẫn `results/cls_confusion_matrix.png`.
4. **Kiểm tra suy luận (Inference):**
   - Đặt đường dẫn ảnh hoặc thư mục vào biến `INFER_PATH` ở cell cuối cùng để chạy suy luận và xuất kết quả ra `results/cls_predictions.csv`.

### Bước 2: Huấn luyện mô hình Phát hiện đối tượng (Notebook 02)

1. Mở `notebooks/02_detection_eelan.ipynb`.
2. Yêu cầu tiên quyết: Đã có tệp `weights/backbone.pt` từ Bước 1.
3. Tại Mục 2 (Xây dựng dữ liệu):
   - Tự động tải và trích xuất nhãn `cat`, `dog` từ Pascal VOC 2007 (`BUILD_VOC = True`).
   - Nạp dữ liệu `panda` từ Roboflow để đồng bộ về định dạng YOLO.
4. Chạy huấn luyện (**Run All**):
   - Nạp trọng số và đóng băng Backbone E-ELAN.
   - Huấn luyện Neck (SimpleFPN) và Head (SSD Head) qua 30 epochs bằng Multibox Loss kết hợp Hard Negative Mining (3:1).
   - Xuất trọng số `weights/best_det.pt`.
   - Đánh giá $\text{mAP}@0.5$, vẽ bounding box trực quan hóa vào `results/det_vis/` và xuất kết quả ra `results/det_predictions.csv`.

---


