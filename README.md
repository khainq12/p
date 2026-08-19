# hawkbot semantic mapping

Pipeline định vị vật thể (chậu cây) bằng YOLO + tam giác hóa tia camera dựa trên
odometry của robot "hawkbot" (ESP32-CAM QVGA + ROS2 Jazzy).

Xem **[ISSUE.md](ISSUE.md)** để biết vấn đề đang gặp phải và những gì đã thử /
đã loại trừ / còn cần làm tiếp.

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
