# Bàn giao phiên làm việc — Định vị vật thể ngữ nghĩa Hawkbot

Đọc file này để hiểu **toàn bộ tiến trình đã làm** và **tiếp tục ngay** mà
không cần hỏi lại người dùng những gì đã biết. Cập nhật lần cuối: xem lịch sử
commit của file này trên GitHub.

## 1. Bối cảnh dự án

Đề tài NCKH: định vị tuyệt đối vật thể (chậu cây) trong bản đồ 3D ngữ nghĩa
cho robot giao hàng tự hành **Hawkbot** (ROS2 Jazzy, chạy qua container
Docker `hawkbot-bringup` local trên máy — **không qua SSH**, dùng lệnh
`docker exec hawkbot-bringup ...` / `docker cp ...`), camera **ESP32-CAM**
(QVGA 320×240). Xử lý nặng (YOLO, MapAnything) chạy trên server GPU từ xa
**avis-1** (SSH alias, user `khai`, thư mục làm việc
`/home/khai/semantic-mapping`, GPU **RTX 5000 Ada 32GB VRAM** — đủ mạnh,

**Kết nối avis-1**: alias đã cấu hình sẵn trong `~/.ssh/config` (nằm ở home
directory, KHÔNG phải `/tmp` nên bền vững qua các lần sandbox reset):
```
Host avis-1
  HostName avis-1
  Port 2202
  User khai
```
Hostname `avis-1` phân giải qua mạng **Tailscale** (đã cài, đã kết nối sẵn
trên máy — `tailscale status` để kiểm tra). Vì vậy một phiên Claude Code MỚI
trên **CÙNG máy sandbox này** chỉ cần gọi thẳng `ssh avis-1 "..."`, không cần
cấu hình gì thêm. SSH thỉnh thoảng timeout (`port 2202 Connection timed
out`) — không phải lỗi, thử lại vài lần cách nhau ~1-2 phút.
KHÔNG phải nút thắt).

Repo: **https://github.com/khainq12/p**

## 2. Phương pháp hiện tại (đã chốt, đừng quay lại tam giác hoá)

**Lịch sử quan trọng**: bài báo TỪNG dùng tam giác hoá tia camera từ
odometry, nhưng đã CHỨNG MINH sai vì **wheel-odometry YAW trôi mạnh khi robot
rẽ/xoay tại chỗ** (chi tiết điều tra gốc rễ ở `ISSUE.md` — mang tính lịch
sử, KHÔNG cần đọc lại trừ khi tò mò). Đã bỏ hẳn hướng đó.

**Phương pháp thay thế hiện dùng** (`pipeline/localize_semantic_object.py`),
kết hợp 2 tín hiệu độc lập, không dựa vào pose odometry của từng khung đơn lẻ:

1. **Hướng (bearing)**: tâm cụm điểm 3D ngữ nghĩa lớn nhất (DBSCAN, lọc theo
   lớp COCO mục tiêu) trong không gian riêng của MapAnything, quy đổi sang
   toạ độ thực bằng **phép khớp toàn cục Umeyama** (so khớp hình dạng quỹ đạo
   camera của MapAnything với quỹ đạo odometry, tính MỘT LẦN cho cả quỹ đạo
   — ít nhạy nhiễu cục bộ hơn tam giác hoá từng tia).
2. **Khoảng cách (đơn-ảnh/monocular)**: từ MỘT khung hình duy nhất, dùng
   chiều cao khung bao (bbox) YOLO + chiều cao thật đã biết của vật thể
   (chậu cây = 1.25m) + tiêu cự camera đã hiệu chỉnh ($f_y = 427.60$):
   $d = f_y \cdot H / h_{px}$. Hoàn toàn KHÔNG dùng odometry.

Vị trí cuối = vị trí camera tham chiếu + hướng × khoảng cách.

**Quy trình đủ 8 bước (B1–B8)**, mô tả chi tiết trong `paper/bao_cao_hawkbot.tex`
mục 3.3:
B1 thu thập → B2 chọn khung ưu tiên → B3 dựng bản đồ 3D (MapAnything, thuần
hình học) → B4 gán nhãn ngữ nghĩa (YOLO → điểm 3D, Thuật toán 1) → B5 khớp
toàn cục Umeyama → B6 hướng đối tượng (DBSCAN) → B7 khoảng cách đơn-ảnh → B8
kết hợp.

### Script chạy 1 capture mới từ đầu (thứ tự bắt buộc)

Trên **avis-1**, thư mục `/home/khai/semantic-mapping`:

```bash
# 1. Chọn 130 khung ưu tiên (chú ý: --classes có dấu cách, không gạch dưới)
/home/khai/miniforge3/bin/python3 select_frames_smart.py \
  --frames_dir hawkbot_capture_newNN --out_dir hawkbot_capture_newNN/frames_130 \
  --classes 'potted plant'

# 2. LỖI HAY GẶP: selection_meta.json bị ghi lộn vào trong frames_130/ — di chuyển ra ngoài
mv hawkbot_capture_newNN/frames_130/selection_meta.json hawkbot_capture_newNN/selection_meta.json

# 3. Dựng bản đồ 3D bằng MapAnything — BẮT BUỘC dùng env `mapanything` (có GPU),
#    KHÔNG dùng env base (base có torch nhưng driver CUDA không khớp, tự động
#    rơi về CPU rất chậm mà không báo lỗi rõ ràng)
/home/khai/miniforge3/envs/mapanything/bin/python3 extract_cam_positions_newNN.py

# 4. Tính scale (Umeyama fit với odometry) — dùng env base (chỉ cần numpy)
/home/khai/miniforge3/bin/python3 compute_scale_newNN.py

# 5. YOLO + MapAnything gán nhãn ngữ nghĩa đầy đủ — env mapanything
/home/khai/miniforge3/envs/mapanything/bin/python3 yolo_mapanything.py \
  --input hawkbot_capture_newNN/frames_130 --output_dir hawkbot_semantic_output_newNN --conf 0.3

# 6. Định vị cuối cùng — env base
/home/khai/miniforge3/bin/python3 localize_semantic_object.py newNN
```

Các script `extract_cam_positions_newNN.py` / `compute_scale_newNN.py` là
bản sao-sửa-tên theo từng capture (không có bản tổng quát dùng chung — cần
`sed 's/newXX/newNN/g'` từ bản gần nhất mỗi lần capture mới).

**Bẫy đã gặp và đã vá**:
- Với capture nhiều khung (>200, vd. quét cả phòng), `yolo_mapanything.py`
  bản GỐC bị OOM (out of memory) ở 2 chỗ: (a) bước xuất mesh
  `original_3d_map.glb` (không cần thiết, đã bỏ), (b) vòng lặp Python thuần
  gán màu từng pixel (đã vector hoá bằng numpy). Bản vá lưu riêng
  `yolo_mapanything_newNN.py` trên avis-1 (không ghi đè bản gốc). Với ≤130
  khung thì bản GỐC chạy bình thường, không cần vá.
- `localize_semantic_object.py` mặc định dùng `frame_000000.jpg` làm khung
  tham chiếu để đo bbox — nếu khung đó KHÔNG thấy vật mục tiêu (robot mới
  bắt đầu quay hướng khác), script báo lỗi `ValueError: khong tim thay...`.
  Cách xử lý: chạy YOLO nhanh trên vài khung đầu để tìm khung có phát hiện
  (conf ≥ 0.2), rồi gọi `localize()` trực tiếp qua Python với tham số
  `ref_frame='frame_00000X.jpg'` khác thay vì chạy `__main__` mặc định.

## 3. Dữ liệu đã thu thập — ĐỪNG capture lại, dùng số liệu có sẵn

**16 phiên chụp định vị 1 vật thể** (chậu cây), số liệu đầy đủ ở Bảng 2 trong bài:

| Khoảng cách thực địa | Capture ID (avis-1) | Số phiên |
|---|---|---|
| 3m | `new36, new46, new47, new48, new49, new50` | 6 |
| 5m | `new52, new53, new54, new55, new56` | 5 |
| 7m | `new57, new58, new59, new60, new61` | 5 |

Kết quả tóm tắt (đã có trong bài, KHÔNG cần đo lại):
- 3m: 2.77/2.76/3.13/3.08/3.26/3.12 m — TB 3.02m, std 0.19m, sai lệch TB +0.6%
- 5m: 4.49/5.07/5.00/5.06/5.01 m — TB 4.93m, std 0.22m, sai lệch TB −1.5%
- 7m: 6.21/6.22/6.17/6.34/6.13 m — TB 6.21m, std **rất nhỏ 0.07m**, sai lệch
  TB **−11.2% (sai số hệ thống, không phải nhiễu)**

**1 phiên quét toàn phòng** (không nhắm 1 vật cụ thể): `new51`, 239 khung
hình → 38.8 triệu điểm, YOLO gán nhãn 12 lớp COCO. Dùng cho Hình 2 trong bài
(`paper_figs/room_scan_photo.png` + `room_scan_topdown.png`, do người dùng tự
chụp màn hình viewer 3D rồi cung cấp, KHÔNG phải tôi tự render).

**Lưu ý về capture trên robot thật (nếu làm thêm)**: script
`auto_capture_simple.py` trong container mặc định ghi vào
`/root/hawkbot_capture_new3` nhưng người dùng thường tự gõ
`--out /root/hawkbot_capture_new3j` và **DÙNG LẠI ĐÚNG TÊN NÀY MỖI LẦN**
(chưa từng đổi sang tên khác dù đã được nhắc) → mỗi capture mới **ghi đè**
lên đầu các file `frame_000000.jpg...` của capture trước (frame index luôn
bắt đầu lại từ 0), để lại các file thừa ở cuối chưa bị ghi đè. Quy trình an
toàn đã dùng nhiều lần:
1. Check `stat -c '%y %n' .../frame_000000.jpg .../poses.json` để biết có
   capture mới không (thời gian đổi).
2. Nhị phân tìm ranh giới: so `stat` của vài frame cuối để biết đúng bao
   nhiêu frame là MỚI (só khớp với số lượng `poses.json` báo — field
   `n poses` = số frame mới).
3. Chỉ `docker cp` đúng dải frame mới + `poses.json` ra, đặt tên capture mới
   (newNN tiếp theo), rồi mới xử lý tiếp theo pipeline ở trên.

## 4. Cấu trúc bài báo (`paper/bao_cao_hawkbot.tex`)

XeLaTeX + TinyTeX (`~/.TinyTeX`, đã cài đủ package kể cả `stfloats`,
`multirow`, `siunitx`, `placeins`, `booktabs`...), 2 cột, ~9 trang hiện tại.
Biên dịch:
```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
cd <thư mục chứa .tex và paper_figs/>
xelatex -interaction=nonstopmode bao_cao_hawkbot.tex   # chạy 2 lần liên tiếp
```

Mục lục hiện tại: 1 Giới thiệu → 2 Công trình liên quan → 3 Phương pháp
(3.1 phần cứng, 3.2 hạn chế phần cứng, 3.3 pipeline B1–B8 + Thuật toán 1, 3.4
Công thức gồm cả điều kiện toán học cho **nhập nhằng khoảng cách–kích thước**
scale-distance ambiguity) → 4 Kết quả thực nghiệm (4.1 định lượng + Bảng 2,
4.2 trực quan hoá, 4.3 mở rộng quét phòng, 4.4 đánh giá theo khoảng cách) →
5 Thảo luận (5.1 vì sao khớp toàn cục ổn định hơn + **Bảng 3** so sánh thực
nghiệm khoảng cách đơn-ảnh vs khoảng cách 3D thô từ MapAnything (kết quả: 3D
thô TỆ HƠN NHIỀU, có ca lệch −80%, xác nhận lựa chọn thiết kế), 5.2 hướng
phần cứng đã thử/chưa khả thi, 5.3 hạn chế) → 6 Kết luận → 9 tài liệu tham
khảo.

**Danh sách hình đang dùng** (số hình sẽ tự đổi theo LaTeX, đây là tên
label cố định để tra):
- `fig:pipeline-new` (Hình 1) — sơ đồ pipeline, `paper_figs/fig_pipeline_final.png`.
  **Đã xuất thêm bản draw.io**: `paper/diagrams/hinh1_pipeline.drawio`.
- `fig:room-scan` (Hình 2) — ảnh thực tế + bản đồ điểm quét phòng, 2 subfigure.
- `fig:results-bar` — biểu đồ cột 6 phiên 3m.
- `fig:6cap-3d` — **ĐANG CẦN THAY** (xem mục 5 bên dưới).
- `fig:six-viewers` — lưới 6 ảnh viewer 3D (do người dùng cung cấp, `paperfig_newXX.png`).
- `fig:real-photo` — ảnh môi trường + 1 khung YOLO phát hiện chậu cây.

**Vấn đề bố cục 2 cột đã xử lý (rất nhiều công sức, đọc kỹ trước khi động vào)**:
LaTeX 2 cột mặc định lấp ĐẦY cột trái trước rồi mới sang cột phải — KHÔNG tự
cân bằng. Đã dùng 3 kỹ thuật kết hợp:
1. `\usepackage{stfloats}` + đặt `figure*` (ảnh full-width) ở `[!b]` (cuối
   trang) thay vì `[t]` (đầu trang) — cho phép chữ lấp đầy 2 cột TRƯỚC khi
   hình full-width xuất hiện bên dưới.
2. Đổi `[H]` (từ package `float`, ép cứng không cho trôi) thành `[t]` cho
   các hình 1-cột nhỏ — `[H]` là nguyên nhân khiến nhiều hình dồn cục vào 1
   cột, bỏ trống hẳn cột kia.
3. Chèn `\newpage` thủ công giữa 2 đoạn/hình khi cần — trong môi trường
   `twocolumn`, `\newpage` = ngắt sang CỘT kế tiếp (không phải trang mới) nếu
   chưa ở cột cuối cùng của trang.

Trang 4 và trang 6 đã cải thiện nhiều nhưng **chưa hoàn hảo tuyệt đối** (vẫn
còn khoảng trắng vừa phải, đặc biệt trang 6 dưới cột phải — do ngay sau đó
là `fig:six-viewers` rất lớn bắt buộc nằm đầu trang mới). Đã báo người dùng
đây gần như là giới hạn thực tế của LaTeX two-column không dùng `multicols`
(đổi sang `multicols` là thay đổi cấu trúc lớn, rủi ro cao, CHƯA làm).

## 5. VIỆC ĐANG DANG DỞ — làm tiếp ngay

Người dùng chê hình `fig:6cap-3d` (quỹ đạo 3D matplotlib, mỗi phiên một hệ
toạ độ riêng — khó đọc độ hội tụ) và đề xuất thay bằng biểu đồ "bia bắn
cung" (dartboard): tâm = vị trí thật, 6 tam giác quanh tâm ở đúng bán kính =
khoảng cách ước lượng từng phiên, góc chỉ để tách hình cho dễ nhìn (KHÔNG
biểu diễn hướng thật — vì các phiên không chung hệ quy chiếu la bàn tuyệt
đối, đã giải thích rõ với người dùng lý do không vẽ hướng thật).

Đã vẽ và **gửi bản xem trước cho người dùng** (`fig_dartboard_v2.png`,
script `plot_dartboard.py` — **CẢ HAI CHỈ NẰM Ở /tmp SCRATCHPAD, CHƯA COMMIT
VÀO GIT, có thể đã MẤT nếu sandbox reset lần nữa — cần vẽ lại nếu không còn**).
Script dùng dữ liệu có sẵn trong Bảng 2 (KHÔNG cần SSH avis-1):
```python
labels = ['Phien 1'..'Phien 6']
dist = [2.77, 2.76, 3.13, 3.08, 3.26, 3.12]   # từ Bảng 2, nhóm 3m
gt = 3.00; std = 0.19
angles_deg = [0, 60, 120, 180, 240, 300]  # chỉ để tách hình, ghi rõ trong caption
# vẽ polar: tâm đen = vị trí thật, vòng nét đứt bán kính=gt, dải mờ ±std,
# 6 tam giác màu xanh lá (#1a8f5e, khớp tông màu các hình khác trong bài)
```

**Đang CHỜ người dùng duyệt hình** trước khi thay vào
`bao_cao_hawkbot.tex` (thay thế toàn bộ block `\begin{figure}...fig:6cap-3d...\end{figure}`,
xoá script cũ `plot_6captures_3d.py`/`plot_6captures_3d_v2.py` khỏi luồng
build nếu không dùng nữa). Việc cần làm khi được duyệt:
1. Vẽ lại hình (nếu file /tmp đã mất) bằng script trên, lưu **NGAY vào
   `paper/paper_figs/` trong repo git** (đừng chỉ để ở /tmp — sandbox đã
   reset mất dữ liệu 2 lần trong phiên này).
2. Sửa `bao_cao_hawkbot.tex`: thay ảnh + caption của `fig:6cap-3d`, cập nhật
   câu văn tham chiếu hình nếu cần (mô tả nội dung hình đã đổi khác hẳn).
3. Biên dịch lại, đọc lại TOÀN BỘ các trang bằng cách render PNG + đọc ảnh để
   kiểm tra không lỗi, không tràn trang, không vỡ bố cục 2 cột (xem mục 4).
4. Đóng gói `bao_cao_hawkbot.pdf` + `.tex` + `paper_figs/`, gửi người dùng,
   copy vào git repo, commit + `git push`.

## 6. Bài học vận hành quan trọng (đọc để khỏi lặp lại lỗi)

- **`/tmp` scratchpad KHÔNG bền vững** — đã bị xoá sạch ít nhất 2 lần trong
  phiên này (do session hết hạn mức rồi reset). MỌI file quan trọng (hình,
  script tạo hình, `.drawio`...) phải **copy vào git repo và `git push`
  NGAY** sau khi tạo xong, đừng để tích luỹ nhiều việc rồi mới lưu.
- **SSH tới avis-1 thỉnh thoảng timeout** (`port 2202 Connection timed out`)
  — không phải lỗi, chỉ cần thử lại (`timeout 20 ssh avis-1 "echo ok"`), có
  lúc phải đợi vài phút rồi mới kết nối lại được.
- **Không tự ý capture lại dữ liệu robot** — 16 phiên đã đủ dùng cho bài,
  đừng đề xuất capture thêm trừ khi người dùng yêu cầu rõ.
- **Đừng viết filler/đệm chữ chỉ để lấp khoảng trắng trang** — nội dung thêm
  vào phải có ý nghĩa thật (đã từng thêm câu preview hợp lý về việc mở rộng
  5m/7m để vừa lấp trang vừa đúng nội dung, đó là ví dụ TỐT; đừng lặp lại
  cùng một câu vô nghĩa).
- Toàn bộ trích dẫn trong bài (9 tài liệu tham khảo) đã được kiểm tra kỹ,
  chỉ dùng nguồn có độ tin cậy cao (YOLO, Umeyama, DBSCAN, ORB-SLAM, Kimera,
  SemanticFusion, ROS, MapAnything model-card) — nếu cần thêm trích dẫn mới,
  giữ nguyên tắc KHÔNG bịa chi tiết trích dẫn không chắc chắn.
