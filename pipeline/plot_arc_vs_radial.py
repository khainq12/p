import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

CAP = 'new36'
poses = json.load(open(f'/home/khai/semantic-mapping/hawkbot_capture_{CAP}/poses.json'))['poses']
pos_by_idx = {}
for p in poses:
    idx = int(p['frame'].split('_')[1].split('.')[0])
    pos_by_idx[idx] = (p['position']['x'], p['position']['y'])

tri = json.load(open(f'/home/khai/semantic-mapping/triangulated_plant_{CAP}.json'))

arc_group = None
radial_group = None
for vp in tri:
    if abs(vp['point'][0] - 1.192) < 0.01 and abs(vp['point'][1] - 1.350) < 0.01:
        arc_group = vp
    if abs(vp['point'][0] - 0.664) < 0.01 and abs(vp['point'][1] - (-0.861)) < 0.01:
        radial_group = vp

def dist_series(vp):
    px, py = vp['point'][0], vp['point'][1]
    frames = sorted(set(vp['orig_frames']))
    dists = [np.hypot(px - pos_by_idx[f][0], py - pos_by_idx[f][1]) for f in frames]
    return frames, dists

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)

f_arc, d_arc = dist_series(arc_group)
f_rad, d_rad = dist_series(radial_group)

t_arc = np.linspace(0, 1, len(d_arc))
t_rad = np.linspace(0, 1, len(d_rad))

ax.plot(t_arc, d_arc, 'o-', color='#22c55e', linewidth=2, markersize=7,
        label=f'Arc pass (frames {f_arc[0]}-{f_arc[-1]}, baseline 0.90m) → 2.97m')
ax.plot(t_rad, d_rad, 's-', color='#ef4444', linewidth=2, markersize=7,
        label=f'Straight-approach pass (frames {f_rad[0]}-{f_rad[-1]}, baseline 0.65m) → 0.71m')

ax.axhline(3.00, color='#6b7280', linestyle='--', linewidth=1.3, label='Tape-measured ground truth: 3.00m')

ax.set_xlabel('Normalized progress through the observation pass', fontsize=11)
ax.set_ylabel('Camera-to-target distance (m)', fontsize=11)
ax.set_title('Camera-to-target distance during observation:\narc motion (stable) vs. straight-line approach (decaying)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='center left', fontsize=9, framealpha=0.95)
ax.set_ylim(0, 3.6)

fig.tight_layout()
fig.savefig('/home/khai/semantic-mapping/fig_arc_vs_radial.png', dpi=200, bbox_inches='tight')
print('saved fig_arc_vs_radial.png')
