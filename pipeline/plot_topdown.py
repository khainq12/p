import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

CAP = 'new29'
poses = json.load(open(f'/home/khai/semantic-mapping/hawkbot_capture_{CAP}/poses.json'))['poses']
xs = [p['position']['x'] for p in poses]
ys = [p['position']['y'] for p in poses]
ts = [p['t'] - poses[0]['t'] for p in poses]

tri = json.load(open(f'/home/khai/semantic-mapping/triangulated_plant_{CAP}.json'))
# khop nhan theo khung hinh dau tien cua tung nhom (KHONG dua vao thu tu mang,
# vi file JSON luu theo thu tu sap xep residual, khac voi thu tu A/B/C/D)
label_by_first_frame = {
    0: ('A (0-9), b=0.86m, CAO', '#22c55e'),
    29: ('B (29-41), b=0.89m, CAO', '#4ade80'),
    80: ('C (80-86), b=0.08m, THẤP', '#60a5fa'),
    120: ('D (120-129), b~0, THẤP', '#f472b6'),
}
offset_by_first_frame = {0: (0, 22), 29: (0, -30), 80: (-15, 26), 120: (45, 22)}

fig, ax = plt.subplots(figsize=(10, 5), dpi=180)

sc = ax.scatter(xs, ys, c=ts, cmap='plasma', s=22, zorder=2)
ax.plot(xs, ys, color='#999999', linewidth=0.6, alpha=0.5, zorder=1)
ax.scatter([xs[0]], [ys[0]], marker='^', s=220, c='#8b96a3', edgecolors='black', linewidths=1.3, zorder=5, label='Xuất phát (odom gốc)')

for vp in tri:
    first_frame = min(vp['orig_frames'])
    if first_frame not in label_by_first_frame:
        continue
    lbl, col = label_by_first_frame[first_frame]
    off = offset_by_first_frame[first_frame]
    px, py = vp['point'][0], vp['point'][1]
    ax.scatter([px], [py], marker='*', s=420, c=col, edgecolors='black', linewidths=1.3, zorder=6)
    ax.annotate(lbl, (px, py), textcoords='offset points', xytext=off, fontsize=8.5, ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=col, alpha=0.95))

cbar = fig.colorbar(sc, ax=ax, pad=0.015, shrink=0.85)
cbar.set_label('Thời gian từ lúc bắt đầu capture (s)', fontsize=9)

ax.set_xlabel('x (m, hệ toạ độ odom)', fontsize=11)
ax.set_ylabel('y (m, hệ toạ độ odom)', fontsize=11)
ax.set_title('Quỹ đạo robot (odometry) và các ước lượng vị trí chậu cây — Capture 29', fontsize=12, fontweight='bold')
ax.set_box_aspect(0.42)
ax.set_ylim(-0.42, 0.30)
ax.grid(True, alpha=0.25)
ax.legend(loc='lower right', fontsize=9)

fig.tight_layout()
fig.savefig('/home/khai/semantic-mapping/fig_topdown_new29.png', dpi=180, bbox_inches='tight')
print('saved fig_topdown_new29.png')
