import json
import numpy as np
import base64

NPZ_PATH = '/home/khai/semantic-mapping/hawkbot_semantic_output_new36/detections.npz'

d = np.load(NPZ_PATH)
pts = d['points'].astype(np.float32)
colors = d['colors'].astype(np.float32)
maxc = colors.max(axis=1)
mask_norm = maxc <= 1.0
colors[mask_norm] *= 255
colors = np.clip(colors, 0, 255).astype(np.uint8)

lo = np.percentile(pts, 1, axis=0)
hi = np.percentile(pts, 99, axis=0)
mask = np.all((pts >= lo) & (pts <= hi), axis=1)
pts_f = pts[mask]
center_raw = pts_f.mean(axis=0)
target_std = 0.284
scale = target_std / pts_f.std()
print('center_raw', center_raw, 'scale', scale)

markers = json.load(open('/home/khai/semantic-mapping/markers_new36.json'))
bottle_centroids = [mr['raw'] for mr in markers['markers_raw']]

MAX_PTS = 750000
RADIUS_REAL_M = 0.20
gf = json.load(open('/home/khai/semantic-mapping/global_fit_new36.json'))
RADIUS_RAW = RADIUS_REAL_M / gf['s']

near_bottle_mask = np.zeros(len(pts), dtype=bool)
for c in bottle_centroids:
    c = np.array(c)
    d2 = np.sum((pts - c) ** 2, axis=1)
    near_bottle_mask |= (d2 <= RADIUS_RAW ** 2)

print('so diem GAN vi tri chai:', near_bottle_mask.sum())

guaranteed_idx = np.where(near_bottle_mask)[0]
rest_idx = np.where(~near_bottle_mask)[0]
if len(guaranteed_idx) > MAX_PTS // 2:
    guaranteed_idx = np.random.choice(guaranteed_idx, MAX_PTS // 2, replace=False)
remaining_budget = max(MAX_PTS - len(guaranteed_idx), 0)
if len(rest_idx) > remaining_budget:
    rest_sub = np.random.choice(rest_idx, remaining_budget, replace=False)
else:
    rest_sub = rest_idx

final_idx = np.concatenate([guaranteed_idx, rest_sub])
pts_out = pts[final_idx]
colors_out = colors[final_idx]
print('tong so diem hien thi cuoi cung:', len(pts_out))

norm_pts = (pts_out - center_raw) * scale
buf = norm_pts.astype(np.float32).tobytes() + colors_out.tobytes()
b64 = base64.b64encode(buf).decode('ascii')
with open('/home/khai/semantic-mapping/viewer_pc_new36_b64.txt', 'w') as f:
    f.write(b64)
print('da luu viewer_pc_new36_b64.txt')

with open('/home/khai/semantic-mapping/norm_transform_new36.json', 'w') as f:
    json.dump({'center_raw': center_raw.tolist(), 'scale': float(scale)}, f)

# trajectory (raw cam positions) -> viewer space, de ve duong di robot
cam_pos = np.load('/home/khai/semantic-mapping/cam_positions_new36.npy')
traj_viewer = ((cam_pos - center_raw) * scale).tolist()
with open('/home/khai/semantic-mapping/traj_new36_viewerspace.json', 'w') as f:
    json.dump(traj_viewer, f)
print('da luu traj_new36_viewerspace.json')

# markers -> viewer space
marker_viewer = []
for c in bottle_centroids:
    v = ((np.array(c) - center_raw) * scale).tolist()
    marker_viewer.append(v)
start_v = ((np.array(markers['start_raw']) - center_raw) * scale).tolist()
marker_viewer.append(start_v)
with open('/home/khai/semantic-mapping/markers_new36_viewerspace.json', 'w') as f:
    json.dump(marker_viewer, f)
print('da luu markers_new36_viewerspace.json:', marker_viewer)
