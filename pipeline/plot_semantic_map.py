import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

CAP = 'new29'
d = np.load(f'/home/khai/semantic-mapping/hawkbot_semantic_output_{CAP}/detections.npz')
pts = d['points'].astype(np.float32)
colors = d['colors'].astype(np.float32)
maxc = colors.max(axis=1)
mask_norm = maxc <= 1.0
colors[mask_norm] *= 255
colors = np.clip(colors, 0, 255).astype(np.uint8)
labels = d['labels']

lo = np.percentile(pts, 1, axis=0)
hi = np.percentile(pts, 99, axis=0)
mask = np.all((pts >= lo) & (pts <= hi), axis=1)
pts_f = pts[mask]
colors_f = colors[mask]
labels_f = labels[mask]

# subsample for plotting speed
N_PLOT = 120000
if len(pts_f) > N_PLOT:
    idx = np.random.choice(len(pts_f), N_PLOT, replace=False)
    pts_p = pts_f[idx]
    colors_p = colors_f[idx]
    labels_p = labels_f[idx]
else:
    pts_p, colors_p, labels_p = pts_f, colors_f, labels_f

markers_fixed = json.load(open(f'/home/khai/semantic-mapping/markers_{CAP}_raw_fixed.json'))
marker_raw = np.array(markers_fixed['markers_raw_fixed'])
start_raw_override = np.array(markers_fixed['start_raw'])
marker_labels = ['Lần A (0-9), b=0.86m', 'Lần B (29-41), b=0.89m', 'Lần C (80-86), b=0.08m', 'Lần D (120-129), b~0']
marker_colors = ['#ff3b30', '#ff9500', '#60a5fa', '#f472b6']  # A/B do/cam de noi bat tren nen cay xanh la
start_raw = start_raw_override

fig = plt.figure(figsize=(9, 8), dpi=180)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(pts_p[:, 0], pts_p[:, 2], pts_p[:, 1], c=colors_p / 255.0, s=0.8, alpha=0.55, linewidths=0)

plant_mask = labels_p == 58
if plant_mask.sum() > 0:
    ax.scatter(pts_p[plant_mask, 0], pts_p[plant_mask, 2], pts_p[plant_mask, 1],
               c='#16a34a', s=3.0, alpha=0.85, label='Điểm gán nhãn "potted plant"')

ax.view_init(elev=32, azim=-100)
ax.set_axis_off()
try:
    ax.set_box_aspect([1, 1, 0.6])
except Exception:
    pass
fig.canvas.draw()  # can ve 1 lan de proj3d.proj_transform dung ma tran chieu hien tai

# ve marker o toa do 2D tren cung (khong bi 3D z-order che khuat)
from mpl_toolkits.mplot3d import proj3d
ax2d = fig.add_axes([0, 0, 1, 1])
ax2d.set_xlim(0, 1); ax2d.set_ylim(0, 1)
ax2d.axis('off')
ax2d.set_zorder(100)

def project_to_fig(x, y, z):
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    disp = ax.transData.transform((x2, y2))
    return fig.transFigure.inverted().transform(disp)

for j in range(len(marker_raw)):
    fx, fy = project_to_fig(marker_raw[j, 0], marker_raw[j, 2], marker_raw[j, 1])
    ax2d.scatter([fx], [fy], c=marker_colors[j], s=280, marker='o',
                 edgecolors='white', linewidths=2.4, zorder=200)
fx, fy = project_to_fig(start_raw[0], start_raw[2], start_raw[1])
ax2d.scatter([fx], [fy], c='#8b96a3', s=200, marker='^', edgecolors='black', linewidths=1.3, zorder=200)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, markeredgecolor='black')
           for c in marker_colors] + [plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#8b96a3', markersize=10, markeredgecolor='black')]
fig.legend(handles, marker_labels + ['Xuất phát'], loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Bản đồ ngữ nghĩa 3D — Capture 29 (MapAnything, 130 khung hình)', fontsize=13, fontweight='bold', y=0.97)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig('/home/khai/semantic-mapping/fig_semantic_map.png', dpi=180, bbox_inches='tight')
print('saved fig_semantic_map.png')
