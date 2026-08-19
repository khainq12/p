import json
import numpy as np
import os

CAM_POS_PATH = '/home/khai/semantic-mapping/cam_positions_new36.npy'
POSES_PATH = '/home/khai/semantic-mapping/hawkbot_capture_new36/poses.json'
FRAMES_130_DIR = '/home/khai/semantic-mapping/hawkbot_capture_new36/frames_130'

cam_pos = np.load(CAM_POS_PATH)
frame_files = sorted(os.listdir(FRAMES_130_DIR))
odom = json.load(open(POSES_PATH))
odom_by_frame = {p['frame']: p['position'] for p in odom['poses'] if 'position' in p}
odom_pos_full = np.full((len(frame_files), 3), np.nan)
for i, fname in enumerate(frame_files):
    full = os.path.join(FRAMES_130_DIR, fname)
    target = os.path.basename(os.readlink(full)) if os.path.islink(full) else fname
    if target in odom_by_frame:
        p = odom_by_frame[target]
        odom_pos_full[i] = [p['x'], p['y'], p['z']]
valid_mask = ~np.isnan(odom_pos_full).any(axis=1)
X_all = cam_pos[valid_mask][:, [0, 2]]
Y_all = odom_pos_full[valid_mask][:, [0, 1]]

def weighted_umeyama(X, Y, w):
    w = w / w.sum()
    mu_x = (w[:, None] * X).sum(axis=0); mu_y = (w[:, None] * Y).sum(axis=0)
    Xc = X - mu_x; Yc = Y - mu_y
    Sigma = (w[:, None, None] * (Yc[:, :, None] @ Xc[:, None, :])).sum(axis=0)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U @ Vt) < 0: S[-1, -1] = -1
    R = U @ S @ Vt
    var_x = (w * (Xc ** 2).sum(axis=1)).sum()
    s = np.trace(np.diag(D) @ S) / var_x
    t = mu_y - s * R @ mu_x
    return s, R, t

s, R, t = weighted_umeyama(X_all, Y_all, np.ones(len(X_all)))
print('scale s =', s, ' (so voi lan truoc ~0.184, ty le =', 0.184/s, 'lan)')
pred = (s * (X_all @ R.T)) + t
resid = np.linalg.norm(pred - Y_all, axis=1)
print('RMS residual:', np.sqrt((resid**2).mean()), 'median:', np.median(resid))

with open('/home/khai/semantic-mapping/global_fit_new36.json','w') as f:
    json.dump({'s':float(s),'R':R.tolist(),'t':t.tolist()}, f)
