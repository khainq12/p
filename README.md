# hawkbot semantic mapping

Pipeline định vị tuyệt đối vật thể (chậu cây) trong bản đồ 3D ngữ nghĩa cho
robot "hawkbot" (ESP32-CAM QVGA + ROS2 Jazzy), kết hợp hướng từ cụm điểm
ngữ nghĩa MapAnything (khớp toàn cục Umeyama) và khoảng cách đơn-ảnh — thay
cho tam giác hóa tia camera dựa trên odometry (đã bỏ, xem lý do ở dưới).

**Đọc [HANDOFF.md](HANDOFF.md) trước tiên** — tóm tắt đầy đủ phương pháp
hiện tại, dữ liệu đã thu thập, cấu trúc bài báo, việc đang dang dở, và các
bẫy/lỗi đã gặp. File này là bàn giao cho phiên làm việc mới.

`[ISSUE.md](ISSUE.md)` mang tính lịch sử — ghi lại quá trình điều tra gốc rễ
vì sao tam giác hóa tia camera sai (odometry yaw trôi khi rẽ), dẫn tới quyết
định đổi sang phương pháp hiện tại. Không cần đọc trừ khi tò mò.

## Cấu trúc

- `pipeline/` — code xử lý: tam giác hóa (`triangulate_plant_*.py`), kiểm tra độ
  tin cậy kết quả (`reliability_check.py`, `run_reliability_all.py`), dựng
  marker/viewer 3D (`build_markers_new36.py`, `extract_viewer_new36.py`,
  `compute_scale_new36.py`), vẽ hình cho báo cáo (`plot_*.py`), định vị bằng
  ArUco marker (`aruco_localize.py` — đã thử, bỏ vì QVGA + marker 15cm cho
  nhiễu quá lớn ở khoảng cách >2m), chọn khung hình (`select_frames_smart.py`).
- `results/` — kết quả tam giác hóa dạng JSON cho từng lần capture đã xử lý.
- `paper/` — bài báo cáo LaTeX (`bao_cao_hawkbot.tex`) và hình minh họa.

## Môi trường chạy

Xử lý (YOLO, tam giác hóa) chạy trên server GPU riêng (Python 3.13, thư viện
`ultralytics`, `opencv-python`, `numpy`). Dữ liệu ảnh gốc mỗi capture không nằm
trong repo do dung lượng lớn.
