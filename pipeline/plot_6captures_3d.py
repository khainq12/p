import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

CAPS = ['36', '46', '47', '48', '49', '50']
COLORS = {'36': '#22c55e', '46': '#a855f7', '47': '#f59e0b', '48': '#3b82f6', '49': '#ef4444', '50': '#06b6d4'}
REAL_PLANT_HEIGHT = 1.25
CAM_HEIGHT = 0.12

BASE = '/home/khai/semantic-mapping'

fig = plt.figure(figsize=(10, 8.5), dpi=200)
ax = fig.add_subplot(111, projection='3d')

all_final = {}
for cap in CAPS:
    poses = json.load(open(f'{BASE}/hawkbot_capture_new{cap}/poses.json'))['poses']
    xs = [p['position']['x'] for p in poses if 'position' in p]
    ys = [p['position']['y'] for p in poses if 'position' in p]
    zs = [CAM_HEIGHT] * len(xs)
    ax.plot(xs, ys, zs, color=COLORS[cap], linewidth=1.6, alpha=0.85, zorder=3, label=f'new{cap} quy dao camera')
    ax.scatter([xs[0]], [ys[0]], [zs[0]], color=COLORS[cap], s=40, marker='s', zorder=4, edgecolors='black', linewidths=0.6)

    res = json.load(open(f'{BASE}/localize_result_new{cap}.json'))
    fx, fy = res['final_position_xy']
    all_final[cap] = (fx, fy, res['distance_m'])
    ax.plot([fx, fx], [fy, fy], [0, REAL_PLANT_HEIGHT], color=COLORS[cap], linewidth=2.2, linestyle='--', alpha=0.9, zorder=5)
    ax.scatter([fx], [fy], [REAL_PLANT_HEIGHT], color=COLORS[cap], s=220, marker='^', zorder=10,
               edgecolors='black', linewidths=1.2)

ax.set_xlabel('X (m, he toa do rieng cua tung capture)', fontsize=10)
ax.set_ylabel('Y (m)', fontsize=10)
ax.set_zlabel('Do cao (m)', fontsize=10)
ax.set_title('Vi tri chau cay uoc luong boi 6 lan capture doc lap\n(huong MapAnything + khoang cach bbox don-anh)',
              fontsize=12, fontweight='bold')
ax.view_init(elev=28, azim=-60)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig(f'{BASE}/fig_6captures_3d.png', dpi=200, bbox_inches='tight')
print('saved fig_6captures_3d.png')

for cap, (fx, fy, d) in all_final.items():
    print(f'new{cap}: final=({fx:.2f},{fy:.2f})  distance={d:.2f}m')

dists = np.array([v[2] for v in all_final.values()])
print(f'\nTB={dists.mean():.3f}m  std={dists.std():.3f}m ({100*dists.std()/dists.mean():.1f}%)')
