# Vấn đề hiện tại: tam giác hóa vị trí vật thể (chậu cây) sai lệch so với thực tế

## Bối cảnh

Pipeline: ESP32-CAM (320x240, QVGA) gắn trên robot "hawkbot" (ROS2 Jazzy, chạy qua
docker `hawkbot-bringup` local, không qua SSH) → ghi lại chuỗi khung hình + pose
odometry (`poses.json`) mỗi lần "capture" → YOLO phát hiện chậu cây (`potted plant`)
trong từng khung → dùng vị trí camera (từ odometry) + hướng tia nhìn (từ pixel +
intrinsics) để tam giác hóa vị trí 3D của cây bằng least-squares ray intersection
(`pipeline/triangulate_plant_newXX.py`).

Mục tiêu: so sánh vị trí tam giác hóa được với khoảng cách đo tay bằng thước
(ground truth = camera cách chậu cây đúng **3.00m**, đo nhiều lần, xác nhận chắc
chắn cây chưa bao giờ ở gần hơn 3m).

## Đã xác nhận KHÔNG phải nguyên nhân

- **Sai lệch calib camera / độ phân giải**: `esp32cam_intrinsics_qvga.json` đã
  kiểm tra khớp đúng độ phân giải ảnh thật (320x240), reprojection error thấp
  (0.12px).
- **Sai tỉ lệ (scale) odometry vị trí**: kiểm chứng 2 lần độc lập bằng cách đẩy
  robot đúng 5m thực tế, đo được 5.30m và 5.14m (~3-6% sai số) → không đủ để
  giải thích sai lệch 2-7 lần đã quan sát.
- **Độ cao/độ nghiêng gắn camera**: đã kiểm tra qua ảnh chụp thực tế, camera gần
  sát đất, không nghiêng.

## Đã xác nhận LÀ nguyên nhân (2 tầng)

### Tầng 1: chuyển động xuyên tâm (radial) làm tam giác hóa mất ổn định

Khi camera di chuyển gần như thẳng về phía vật thể (baseline lớn nhưng góc thị
sai gần vật thể nhỏ), least-squares ray intersection bị suy biến hình học, cho
kết quả khoảng cách bị đánh giá thấp hơn thực tế rất nhiều. Ngược lại, chuyển
động theo cung tròn (khoảng cách đến vật gần như không đổi) thì tam giác hóa ổn
định và chính xác.

→ Đã viết `pipeline/reliability_check.py` để tự động phát hiện: coefficient of
variation của khoảng cách camera→điểm qua từng khung, độ dốc xu hướng, góc thị
sai giữa 2 tia xa nhau nhất.

### Tầng 2: bộ lọc tầng 1 vẫn "pass" nhầm khi baseline bị "giả lập"

Phát hiện thêm: nhiều khung hình liên tiếp chụp lúc robot **đứng yên** (do dừng
lại quan sát) bị đếm là nhiều tia độc lập, dù chúng gần như trùng gốc tia. Điều
này khiến baseline "trông" đủ lớn (theo max pairwise distance) và góc thị sai
tính từ điểm tam giác hóa (circular — tự tham chiếu) trông hợp lý, nhưng thực tế
chỉ có 2 vị trí camera thực sự độc lập → nghiệm least-squares bị nhiễu góc nhỏ
(pixel jitter, sai số undistort) "hút" về rất gần camera.

→ Vá `reliability_check.py` thêm 2 điều kiện:
1. Gộp các vị trí camera cách nhau < 5cm thành 1 (loại "tia giả lập").
2. Đối chiếu khoảng cách tam giác hóa với ước lượng đơn-ảnh (monocular) từ kích
   thước bbox YOLO (biết trước chiều cao thật của cây = 1.25m,
   `dist = fy * real_height / bbox_height_px`, `fy=427.6`) — lệch quá 2 lần thì
   loại nhóm đó.

Kết quả: chạy lại toàn bộ capture new44–new50, mọi nhóm "trông đáng tin" trước
đây (sau khi vá tầng 1) đều bị loại đúng ở tầng 2. Khoảng cách đơn-ảnh độc lập
đo được ở TẤT CẢ các capture đều ổn định 2.9–3.2m, khớp hoàn toàn với thước đo
tay.

## VẤN ĐỀ CÒN MỞ (chưa giải quyết): odometry YAW trôi khi robot rẽ/xoay tại chỗ

Đây là phần đang cần xử lý tiếp.

Ở capture `new50`, robot dừng đúng 4 điểm quan sát tách biệt thật sự (đúng theo
khuyến nghị đã rút ra ở trên) — baseline thật 1.9m, khoảng cách đơn-ảnh ổn định
3.11–3.21m suốt 4 điểm (xác nhận cây không hề di chuyển gần hơn). Nhưng tam giác
hóa vẫn cho kết quả sai (điểm hội tụ cách gốc tọa độ chỉ ~0.98m thay vì khớp
hướng ra ~3m).

Thử giải ngược góc lệch phương vị (yaw) camera cần cộng thêm vào từng tia để dữ
liệu tự nhất quán với khoảng cách đơn-ảnh đã biết:
- `new50` (4 điểm dừng, có rẽ/xoay giữa các điểm): góc lệch tối ưu **≈ -34°**.
- `new36` (đi theo cung tròn liên tục, không dừng-xoay-dừng): góc lệch tối ưu
  **≈ +2°** (gần như không cần chỉnh).

Cùng một camera, cùng một robot, nhưng góc lệch cần thiết khác nhau hoàn toàn
giữa 2 lần capture → **không phải lỗi lắp đặt camera cố định**, mà nhiều khả
năng là: **odometry bánh xe (wheel encoder) ước lượng hướng quay (yaw) kém
chính xác, đặc biệt trôi nặng khi robot xoay/rẽ tại chỗ** (điểm yếu kinh điển
của differential-drive odometry — sai số quãng đường đã kiểm chứng nhỏ, nhưng
sai số góc quay chưa từng được kiểm chứng riêng và có thể lớn hơn nhiều, nhất
là khi có trượt bánh lúc pivot).

### Việc cần làm tiếp / cần ý kiến

1. Kiểm chứng độc lập độ chính xác của yaw odometry (ví dụ: quay robot đúng
   90°/180° tại chỗ nhiều lần, so với giá trị odometry báo cáo — tương tự cách
   đã làm với quãng đường thẳng 5m).
2. Xem robot có cảm biến IMU sẵn có không (nếu ROS2 Jazzy driver có publish
   `/imu`) — nếu có, cân nhắc dùng yaw từ IMU (hoặc fuse qua `robot_localization`)
   thay vì thuần wheel odometry cho bước tam giác hóa.
3. Trong lúc chưa sửa được gốc rễ: ưu tiên capture theo kiểu cung tròn liên
   tục, mượt (như `new36`), tránh dừng hẳn - xoay tại chỗ - dừng hẳn giữa các
   điểm quan sát.
4. Cân nhắc thêm bước tự động: dùng góc lệch ước lượng được (delta-fit ở
   `run_reliability_all.py` / phần cuối investigation) làm một correction step
   runtime, NHƯNG cần thêm dữ liệu để xác nhận nó có thực sự là hàm của "tổng
   góc đã xoay trong capture" hay không trước khi tin dùng.

## Cách chạy lại

```bash
# 1. Tam giác hóa 1 capture (cần sửa CAP_DIR trong file cho đúng capture)
python3 pipeline/triangulate_plant_new36.py

# 2. Kiểm tra độ tin cậy từng nhóm kết quả
python3 pipeline/run_reliability_all.py
```

Dữ liệu ảnh gốc từng capture (`hawkbot_capture_newXX/`, hàng trăm ảnh mỗi lần)
KHÔNG nằm trong repo này (quá nặng) — đang lưu tại server xử lý GPU
(`/home/khai/semantic-mapping/hawkbot_capture_newXX/`). File JSON kết quả tam
giác hóa (`results/`) và bài báo LaTeX (`paper/`) đã đưa đầy đủ vào repo.
