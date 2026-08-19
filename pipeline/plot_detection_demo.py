import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

CAP = 'new29'
yolo = YOLO('yolo11n.pt')
plant_id = [k for k, v in yolo.model.names.items() if v == 'potted plant'][0]

frames = ['frame_000005.jpg', 'frame_000035.jpg']
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=180)

for ax, fname in zip(axes, frames):
    img = cv2.imread(f'/home/khai/semantic-mapping/hawkbot_capture_{CAP}/{fname}')
    res = yolo.predict(source=img, conf=0.3, save=False, verbose=False)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    if res[0].boxes is not None:
        for box in res[0].boxes:
            if int(box.cls[0]) == plant_id:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='#22c55e', linewidth=2.2)
                ax.add_patch(rect)
                ax.text(x1, y1 - 4, f'potted plant {conf:.2f}', color='white', fontsize=8,
                        bbox=dict(facecolor='#22c55e', pad=1.5, edgecolor='none'))
    ax.set_title(fname, fontsize=10)
    ax.axis('off')

fig.suptitle('Phát hiện YOLO11n trên khung hình ESP32-CAM (320×240, QVGA)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig('/home/khai/semantic-mapping/fig_yolo_detection.png', dpi=180, bbox_inches='tight')
print('saved fig_yolo_detection.png')
