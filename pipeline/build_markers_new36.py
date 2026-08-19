import json
import os
import numpy as np

BASE = '/home/khai/semantic-mapping'
CAPTURE = 'new36'

poses = json.load(open(f'{BASE}/hawkbot_capture_{CAPTURE}/poses.json'))['poses']
pos_by_frame = {p['frame']: p['position'] for p in poses}

cands = [
    {'point': [1.192, 1.350], 'n_rays': 8, 'label': 'Vong cung on dinh (khung 88-145), baseline=0.90m, khoang cach camera-vat ON DINH 2.83-3.24m -- DANG TIN CAY CAO'},
    {'point': [0.664, -0.861], 'n_rays': 9, 'label': 'Di thang lai gan (khung 104-141), baseline=0.65m, khoang cach camera-vat GIAM DAN 1.00-0.55m -- DANG TIN CAY THAP (suy bien)'},
]

cam_pos = np.load(f'{BASE}/cam_positions_{CAPTURE}.npy')
frames_130 = sorted(f for f in os.listdir(f'{BASE}/hawkbot_capture_{CAPTURE}/frames_130') if f.endswith('.jpg'))
targets = []
for fname in frames_130:
    full = f'{BASE}/hawkbot_capture_{CAPTURE}/frames_130/{fname}'
    t = os.path.basename(os.readlink(full)) if os.path.islink(full) else fname
    targets.append(t)
odom_xy = np.array([[pos_by_frame[t]['x'], pos_by_frame[t]['y']] for t in targets])

npz = np.load(f'{BASE}/hawkbot_semantic_output_{CAPTURE}/detections.npz')
pts_all = npz['points'].astype(np.float32)
lo = np.percentile(pts_all, 1, axis=0)
hi = np.percentile(pts_all, 99, axis=0)
diag = np.linalg.norm(hi - lo)

gf = json.load(open(f'{BASE}/global_fit_{CAPTURE}.json'))
s_g, R_g = gf['s'], np.array(gf['R'])

markers_raw = []
for c in cands:
    target_xy = np.array(c['point'])
    dists = np.linalg.norm(odom_xy - target_xy, axis=1)
    nn_idx = np.argsort(dists)[:5]
    nn_odom = odom_xy[nn_idx]
    nn_raw = cam_pos[nn_idx][:, [0, 2]]
    nearest_odom = nn_odom[0]
    nearest_raw = nn_raw[0]
    offset_odom = target_xy - nearest_odom
    offset_raw = R_g.T @ (offset_odom / s_g)
    fallback_raw = nearest_raw + offset_raw

    raw_xz = fallback_raw
    if len(nn_idx) >= 2 and np.std(nn_odom, axis=0).sum() > 0.05:
        w = 1.0 / (dists[nn_idx] + 0.05)
        w = w / w.sum()
        mu_odom = (w[:, None] * nn_odom).sum(axis=0)
        mu_raw = (w[:, None] * nn_raw).sum(axis=0)
        Yc = nn_odom - mu_odom
        Xc = nn_raw - mu_raw
        Sigma = (w[:, None, None] * (Yc[:, :, None] @ Xc[:, None, :])).sum(axis=0)
        U, D, Vt = np.linalg.svd(Sigma)
        Sm = np.eye(2)
        if np.linalg.det(U @ Vt) < 0:
            Sm[-1, -1] = -1
        R_local = U @ Sm @ Vt
        var_x = (w * (Xc ** 2).sum(axis=1)).sum()
        if var_x > 1e-6:
            s_local = np.trace(np.diag(D) @ Sm) / var_x
            offset_odom2 = target_xy - mu_odom
            offset_raw2 = R_local.T @ (offset_odom2 / s_local)
            candidate = mu_raw + offset_raw2
            if np.linalg.norm(candidate - nearest_raw) < diag * 0.5:
                raw_xz = candidate
    Y = float(cam_pos[nn_idx[0], 1])
    markers_raw.append({'raw': [float(raw_xz[0]), Y, float(raw_xz[1])], 'real_xy': c['point'],
                         'n_rays': c['n_rays'], 'label': c['label']})

start_i = None
for i, fname in enumerate(frames_130):
    full = f'{BASE}/hawkbot_capture_{CAPTURE}/frames_130/{fname}'
    target = os.path.basename(os.readlink(full)) if os.path.islink(full) else fname
    if target == 'frame_000000.jpg':
        start_i = i
        break
start_raw = cam_pos[start_i].tolist()
start_odom = pos_by_frame['frame_000000.jpg']

out = {'markers_raw': markers_raw, 'start_raw': start_raw, 'start_odom': [start_odom['x'], start_odom['y']]}
json.dump(out, open(f'{BASE}/markers_{CAPTURE}.json', 'w'), indent=2)
for m in markers_raw:
    d = np.linalg.norm(np.array(m['real_xy']) - np.array([start_odom['x'], start_odom['y']]))
    print(m['label'], m['real_xy'], 'rays=', m['n_rays'], 'cach_xp=%.2fm' % d)
