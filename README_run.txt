================================================================================
BỘ MÔN: MỘT SỐ VẤN ĐỀ CHỌN LỌC VỀ THỊ GIÁC MÁY TÍNH (MAT3563)
BÁO CÁO VÀ HƯỚNG DẪN THỰC THI MINI PROJECT 01
Đề tài: Phân tích khối E-ELAN và ứng dụng trong Phân loại ảnh & Phát hiện đối tượng
Giảng viên hướng dẫn: TS. Cao Văn Chung
================================================================================

THÔNG TIN NHÓM THỰC HIỆN (Nhóm tối đa 02 thành viên):
  1. Nguyễn Đức Quang - MSSV: 23001918
  2. Hà Minh Quang    - MSSV: 23001916

PHÂN CÔNG NHIỆM VỤ CHI TIẾT (Tất cả thành viên đều trực tiếp viết code):
  - Nguyễn Đức Quang (23001918):
      + Phụ trách chính Notebook 01 (Phân loại ảnh - Task 1 & Task 2):
        * Xây dựng kiến trúc backbone E-ELAN trích xuất đặc trưng đa tỉ lệ (C3, C4, C5).
        * Hiện thực cơ chế lưu trọng số (save checkpoint) sau khi huấn luyện và nạp trọng số
          (load weights) từ đường dẫn chỉ định (Task 1).
        * Xây dựng pipeline suy luận (inference) cho ảnh đơn và toàn bộ thư mục ảnh, xuất kết quả
          lớp dự đoán kèm xác suất (probability) ra định dạng bảng và tệp CSV.
        * Tính toán và in ra độ đo Precision, Recall, F1-score (per-class và Macro-average)
          trên tập dữ liệu có nhãn; vẽ đồ thị ma trận nhầm lẫn (Confusion Matrix) (Task 2).
        * Vẽ đồ thị biểu diễn biến thiên của hàm mất mát (Loss) và độ chính xác (Accuracy)
          qua từng epoch huấn luyện (Task 2).

  - Hà Minh Quang (23001916):
      + Phụ trách chính Notebook 02 (Phát hiện đối tượng - Task 3):
        * Thu thập và xây dựng tập dữ liệu phát hiện đối tượng 3 lớp (cat, dog, panda).
        * Tự động tải và trích xuất dữ liệu cat/dog từ Pascal VOC 2007; tải dữ liệu panda từ
          Roboflow Universe (~500 ảnh có nhãn bounding box).
        * Đồng bộ hóa và chuẩn hóa toàn bộ nhãn bounding box từ các nguồn khác nhau về cùng
          01 định dạng thống nhất: định dạng chuẩn YOLO (class_id center_x center_y width height).
        * Ghép backbone E-ELAN (đã đóng băng weights từ bài toán phân loại) với phần Neck
          (SimpleFPN đa tỉ lệ) và phần Head (SSD Head - One-stage detector).
        * Thiết kế cơ chế tạo Anchor đa kích thước, hàm gán nhãn Anchor Matching (IoU threshold),
          và hàm mất mát Multibox Loss (Smooth L1 Loss + Cross Entropy + Hard Negative Mining 3:1).
        * Hiện thực giải thuật giải mã hộp và khử trùng lặp phi cực đại (Non-Maximum Suppression - NMS),
          vẽ bounding box trực quan hóa, xuất thông tin dự đoán (BBoxes, Class, Confidence) ra tệp CSV.
        * Đánh giá hiệu năng mô hình phát hiện bằng các độ đo AP@0.5 từng lớp và mAP@0.5.

  - Phối hợp chung:
      + Cả 2 thành viên cùng rà soát, kiểm thử mã nguồn trên môi trường Google Colab (GPU).
      + Phân tích lý thuyết khối E-ELAN, tổng hợp các bảng biểu thực nghiệm và biên soạn báo cáo PDF.

--------------------------------------------------------------------------------
1. CẤU TRÚC THƯ MỤC DỰ ÁN
--------------------------------------------------------------------------------
Mini Project 01/
  ├── notebooks/
  │     ├── 01_classification_eelan.ipynb   # Task 1 + 2: Huấn luyện phân loại, lưu/nạp weights,
  │     │                                   #             suy luận, đồ thị loss/acc, precision/recall.
  │     └── 02_detection_eelan.ipynb        # Task 3: Xây dựng dữ liệu, ghép E-ELAN + FPN + SSD Head,
  │                                         #         huấn luyện, NMS, trực quan hóa và tính mAP@0.5.
  ├── data/
  │     ├── classification/                 # Dữ liệu phân loại (train, val chia theo thư mục lớp)
  │     │     ├── train/ {cats, dogs, panda}
  │     │     └── val/   {cats, dogs, panda}
  │     └── detection/                      # Dữ liệu phát hiện đối tượng chuẩn định dạng YOLO
  │           ├── images/ {train, val}/     # Ảnh đầu vào (.jpg, .png)
  │           ├── labels/ {train, val}/     # Nhãn bounding box (.txt theo chuẩn YOLO)
  │           └── classes.txt               # Danh sách 3 lớp: cat, dog, panda
  ├── weights/                              # Thư mục lưu trữ trọng số mô hình sau huấn luyện
  │     ├── best_cls.pt                     # Trọng số tốt nhất của toàn bộ mạng phân loại
  │     ├── backbone.pt                     # Trọng số riêng của phần Backbone E-ELAN (cho Task 3)
  │     └── best_det.pt                     # Trọng số tốt nhất của mô hình phát hiện đối tượng
  ├── results/                              # Thư mục lưu trữ kết quả thực nghiệm
  │     ├── cls_training_curves.png         # Đồ thị biến thiên Train/Val Loss và Accuracy
  │     ├── cls_confusion_matrix.png        # Ma trận nhầm lẫn của mô hình phân loại
  │     ├── cls_precision_recall.txt        # Bảng số liệu chi tiết Precision, Recall, F1 theo từng lớp
  │     ├── cls_predictions.csv             # Kết quả suy luận phân loại (Image, Predict, Probabilities)
  │     ├── det_loss_curve.png              # Đồ thị biến thiên Loss trong quá trình huấn luyện phát hiện
  │     ├── det_metrics.txt                 # Bảng số liệu chi tiết AP@0.5 từng lớp và mAP@0.5
  │     ├── det_predictions.csv             # Kết quả suy luận detection (Image, Class, Conf, BBoxes)
  │     └── det_vis/                        # Các ảnh xuất ra sau khi vẽ Bounding Box dự đoán
  ├── report/
  │     └── main.tex                        # Mã nguồn báo cáo chi tiết bằng LaTeX
  ├── CNN_ELAN_Class.ipynb                  # Mã nguồn gốc minh họa khối E-ELAN của giảng viên
  ├── Advanced_CV.pdf                       # Tài liệu mô tả lý thuyết khối E-ELAN (YOLOv7)
  └── README_run.txt                        # Tệp hướng dẫn này

--------------------------------------------------------------------------------
2. YÊU CẦU MÔI TRƯỜNG & HỆ THỐNG
--------------------------------------------------------------------------------
- Nền tảng khuyến nghị: Google Colab (sử dụng GPU T4 miễn phí) kết hợp Google Drive.
  Lý do: Quá trình huấn luyện mạng tích chập sâu và trích xuất đặc trưng yêu cầu tính toán CUDA
  nhanh chóng và ổn định.
- Phiên bản Python khuyến nghị: Python >= 3.9
- Các gói thư viện chính cần thiết:
    + torch, torchvision (PyTorch với hỗ trợ CUDA)
    + scikit-learn (tính toán confusion matrix, classification report)
    + matplotlib, pillow, opencv-python (xử lý và trực quan hóa hình ảnh)
    + tqdm (thanh tiến trình)
- Mã nguồn trong 2 notebook đã được lập trình sẵn cơ chế tự động nhận diện môi trường:
    * Nếu chạy trên Google Colab: Tự động Mount Google Drive và liên kết tới thư mục:
      `/content/drive/MyDrive/MiniProject01`
    * Nếu chạy trên máy cục bộ (Local): Tự động trỏ về đường dẫn:
      `D:/CV nâng cao/Project/Mini Project 01`

--------------------------------------------------------------------------------
3. NGUỒN DỮ LIỆU & QUY TRÌNH CHUẨN BỊ
--------------------------------------------------------------------------------
3.1. Dữ liệu Phân loại (Classification - Task 1 & 2):
  - Bộ dữ liệu: "Animal Image Dataset: DOG, CAT and PANDA" (khoảng 1.000 ảnh cho mỗi lớp).
  - Cấu trúc thư mục:
      data/classification/train/{cats, dogs, panda}/
      data/classification/val/{cats, dogs, panda}/
  - Lưu ý: Nếu tập val chưa được phân chia sẵn, notebook sẽ tự động trích ngẫu nhiên 10%
    từ tập train để tạo tập validation độc lập, đảm bảo tính khách quan khi đánh giá.

3.2. Dữ liệu Phát hiện đối tượng (Object Detection - Task 3):
  - Nhãn Cat và Dog:
      Tự động tải từ tập chuẩn quốc tế Pascal VOC 2007 (thư viện torchvision.datasets.VOCDetection
      với tùy chọn download=True). Notebook lọc riêng các ảnh có chứa đối tượng 'cat' và 'dog',
      sau đó chuyển đổi tọa độ Bounding Box dạng (xmin, ymin, xmax, ymax) sang dạng chuẩn YOLO.
  - Nhãn Panda:
      Thu thập từ Roboflow Universe (từ khóa tìm kiếm: "class:panda", quy mô xấp xỉ 500 ảnh).
      Được giải nén và nạp qua hàm `ingest_yolo_folder` có sẵn trong notebook để chuẩn hóa nhãn.
  - Định dạng chuẩn hóa thống nhất (YOLO format):
      Mỗi ảnh đi kèm một tệp `.txt` cùng tên với cấu trúc từng dòng:
      `<class_id> <center_x> <center_y> <width> <height>`
      Trong đó:
        + class_id: 0 = cat, 1 = dog, 2 = panda
        + center_x, center_y, width, height: tọa độ và kích thước đã được chuẩn hóa trong khoảng [0, 1].

--------------------------------------------------------------------------------
4. HƯỚNG DẪN CHI TIẾT CÁC BƯỚC THỰC THI (STEP-BY-STEP)
--------------------------------------------------------------------------------
BƯỚC 1: Thực thi Notebook 01 - Phân loại ảnh (Task 1 + Task 2)
  1. Mở tệp `notebooks/01_classification_eelan.ipynb` trên Google Colab hoặc Jupyter Lab.
  2. Chọn phần cứng: Runtime -> Change runtime type -> Hardware accelerator: GPU (T4).
  3. Chạy toàn bộ các ô lệnh (Run All):
     - Mô hình khởi tạo kiến trúc đa tầng tích hợp khối E-ELAN.
     - Quá trình huấn luyện diễn ra qua 20 epochs với bộ tối ưu AdamW và Cosine Annealing Learning Rate.
     - Sau khi hoàn thành, mô hình tự động:
       + Lưu trọng số tốt nhất vào `weights/best_cls.pt`.
       + Tách riêng và lưu trọng số phần trích xuất đặc trưng vào `weights/backbone.pt`.
       + Vẽ và lưu đồ thị biến thiên Loss & Accuracy vào `results/cls_training_curves.png`.
       + Tính toán và xuất chỉ số Precision, Recall ra `results/cls_precision_recall.txt`.
       + Vẽ ma trận nhầm lẫn Confusion Matrix ra `results/cls_confusion_matrix.png`.
  4. Kiểm tra chức năng suy luận (Inference):
     - Ở khối cuối cùng của notebook, người chấm có thể chỉnh sửa biến `INFER_PATH` trỏ tới:
       + Đường dẫn 01 tệp ảnh đơn lẻ (.jpg, .png).
       + Hoặc đường dẫn tới một thư mục chứa nhiều ảnh.
     - Notebook sẽ nạp trọng số từ `weights/best_cls.pt`, chạy suy luận, hiển thị ảnh kèm lớp
       dự đoán, xác suất tự tin (%) và lưu toàn bộ kết quả vào tệp `results/cls_predictions.csv`.

BƯỚC 2: Thực thi Notebook 02 - Phát hiện đối tượng (Task 3)
  1. Yêu cầu tiên quyết: Phải có tệp `weights/backbone.pt` đã được sinh ra từ Bước 1.
  2. Mở tệp `notebooks/02_detection_eelan.ipynb` trên Google Colab (GPU).
  3. Tại Mục 2 (Xây dựng dữ liệu):
     - Bật cờ `BUILD_VOC = True` để hệ thống tự động tải và trích xuất dữ liệu Pascal VOC.
     - Chạy ô lệnh nạp dữ liệu Panda từ Roboflow để hoàn thiện tập dataset tổng hợp.
  4. Chạy các ô tiếp theo để khởi tạo mô hình:
     - Nạp trọng số từ `weights/backbone.pt` vào Backbone E-ELAN và kích hoạt chế độ ĐÓNG BĂNG
       (freeze_backbone = True) nhằm bảo toàn đặc trưng đã học từ bài toán phân loại.
     - Khởi tạo phần Neck (SimpleFPN) và Head (SSD Head), cấu hình tập Anchor đa tỉ lệ trên 3 mức P3, P4, P5.
  5. Tiến hành huấn luyện mô hình phát hiện đối tượng:
     - Tối ưu hóa hàm mất mát Multibox Loss (Smooth L1 hồi quy tọa độ + Cross Entropy phân loại nhãn).
     - Áp dụng Hard Negative Mining theo tỉ lệ 3:1 giữa mẫu âm (nền) và mẫu dương (vật thể).
     - Lưu trọng số mô hình tốt nhất vào `weights/best_det.pt`.
     - Vẽ và lưu đường cong biến thiên của Loss vào `results/det_loss_curve.png`.
  6. Đánh giá và kiểm tra suy luận:
     - Chạy khối đánh giá trên tập kiểm thử (validation set): Tính toán chỉ số Average Precision (AP@0.5)
       cho từng lớp và chỉ số mAP@0.5 toàn diện, lưu kết quả tại `results/det_metrics.txt`.
     - Chạy khối suy luận: Ứng dụng giải thuật Non-Maximum Suppression (NMS) để lọc bỏ các hộp trùng lặp,
       vẽ bounding box và nhãn lên ảnh, lưu ảnh trực quan vào thư mục `results/det_vis/`
       và xuất danh sách chi tiết các phát hiện (Image, Class, Confidence, Bounding Box) ra `results/det_predictions.csv`.

--------------------------------------------------------------------------------
5. BẢNG TỔNG HỢP SIÊU THAM SỐ THỰC NGHIỆM (HYPERPARAMETERS)
--------------------------------------------------------------------------------
* Cấu hình bài toán Phân loại ảnh (Classification):
    - Kích thước ảnh đầu vào (Image Size): 224 x 224 pixels
    - Số lượng epochs: 20
    - Kích thước batch (Batch Size): 32
    - Tốc độ học khởi tạo (Learning Rate): 0.002 (2e-3) kèm Cosine LR Scheduler
    - Bộ tối ưu hóa (Optimizer): AdamW (Weight Decay: 0.05)
    - Kênh đặc trưng Backbone: (64, 128, 192, 256) qua 4 stages
    - Tham số khối E-ELAN: Expand ratio m = 2.0, Groups g = 2, 4 nhánh đa chiều sâu

* Cấu hình bài toán Phát hiện đối tượng (Object Detection):
    - Kích thước ảnh đầu vào: 320 x 320 pixels
    - Tầng trích xuất đặc trưng (Feature Strides): 8 (P3), 16 (P4), 32 (P5)
    - Kích thước Anchor cơ sở (Anchor Bases): [32, 80, 160] pixels
    - Tỉ lệ Anchor (Aspect Ratios): [1.0, 2.0, 0.5] kết hợp 2 Scales [1.0, 1.4] -> 6 Anchors/ô lưới
    - Số lượng epochs: 30
    - Kích thước batch: 16
    - Trạng thái Backbone: Đóng băng hoàn toàn (freeze_backbone = True)
    - Ngưỡng IoU Anchor Matching: Dương >= 0.5, Âm < 0.4 (vùng giữa bị bỏ qua)
    - Tỉ lệ Hard Negative Mining: 3 mẫu âm / 1 mẫu dương
    - Ngưỡng tự tin phát hiện (Confidence Threshold): 0.30
    - Ngưỡng chồng lấn lọc hộp NMS (NMS IoU Threshold): 0.45
================================================================================
