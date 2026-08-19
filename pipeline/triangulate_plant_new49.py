"""
Tam giac hoa vi tri CHAU CAY (potted plant) thay vi chai - vat to hon, YOLO on
dinh hon, it nhay voi nhieu pixel. Dung TOAN BO khung goc (khong gioi han
trong frames_130 uu tien chai) de khong bo sot khung co chau cay.
"""
import json
import os
import numpy as np
import cv2
from ultralytics import YOLO

CAP_DIR = '/home/khai/semantic-mapping/hawkbot_capture_new49'
POSES_PATH = f'{CAP_DIR}/poses.json'
INTRINSICS_JSON = '/home/khai/semantic-mapping/esp32cam_intrinsics_qvga.json'

CAM_FORWARD_OFFSET = 0.08
CAM_HEIGHT = 0.12
COLOR_SIM_THRESH = 0.55

calib = json.load(open(INTRINSICS_JSON))
K = np.array(calib['K'], dtype=np.float32)
dist = np.array(calib['dist'], dtype=np.float32)

odom = json.load(open(POSES_PATH))
odom_by_frame = {p['frame']: p for p in odom['poses'] if 'position' in p}

frame_files = sorted(f for f in os.listdir(CAP_DIR) if f.endswith('.jpg'))
print('so khung goc:', len(frame_files))


def quat_to_R(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-9:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z, W = x * s, y * s, z * s, w * s
    return np.array([
        [1 - (y * Y + z * Z), x * Y - w * Z, x * Z + w * Y],
        [x * Y + w * Z, 1 - (x * X + z * Z), y * Z - w * X],
        [x * Z - w * Y, y * Z + w * X, 1 - (x * X + y * Y)],
    ])


def color_hist(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return np.zeros(32)
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


yolo = YOLO('yolo11n.pt')
plant_id = [k for k, v in yolo.model.names.items() if v == 'potted plant'][0]

all_rays = []
n_hit = 0
for i, fname in enumerate(frame_files):
    full = os.path.join(CAP_DIR, fname)
    if fname not in odom_by_frame:
        continue
    p = odom_by_frame[fname]
    pos = p['position']
    ori = p['orientation']
    R_base = quat_to_R(ori['x'], ori['y'], ori['z'], ori['w'])
    cam_origin = np.array([pos['x'], pos['y'], pos['z'] + CAM_HEIGHT]) + R_base @ np.array([CAM_FORWARD_OFFSET, 0, 0])

    img = cv2.imread(full)
    res = yolo.predict(source=img, conf=0.3, save=False, verbose=False)
    dets = []
    if res[0].boxes is not None:
        for box in res[0].boxes:
            if int(box.cls[0]) == plant_id:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                dets.append(((x1 + x2) / 2, (y1 + y2) / 2, float(box.conf[0]), (x1, y1, x2, y2)))
    if dets:
        n_hit += 1
    for (px, py, conf, box) in dets:
        pts = np.array([[[px, py]]], dtype=np.float32)
        und = cv2.undistortPoints(pts, K, dist, P=K)
        ux, uy = und[0, 0]
        ray_cam = np.linalg.inv(K) @ np.array([ux, uy, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_robot = np.array([ray_cam[2], ray_cam[0], -ray_cam[1]])
        ray_world = R_base @ ray_robot
        ray_world = ray_world / np.linalg.norm(ray_world)
        hist = color_hist(img, box)
        all_rays.append({'frame': i, 'origin': cam_origin, 'dir': ray_world, 'conf': conf, 'hist': hist,
                          'orig_idx': int(fname.split('_')[1].split('.')[0]), 'bbox_h': float(box[3] - box[1])})

print('so khung co chau cay:', n_hit)
print('tong so ray:', len(all_rays))


def triangulate(rays):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for r in rays:
        d = r['dir']
        o = r['origin']
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def residuals_of(pt, rays):
    out = []
    for r in rays:
        diff = pt - r['origin']
        perp = diff - np.dot(diff, r['dir']) * r['dir']
        out.append(float(np.linalg.norm(perp)))
    return out


def robust_triangulate(rays, max_outlier_ratio=0.4):
    cur = list(rays)
    n0 = len(cur)
    removed = []
    while len(cur) >= 2:
        pt = triangulate(cur)
        resid = residuals_of(pt, cur)
        med = np.median(resid)
        worst_i = int(np.argmax(resid))
        worst_r = resid[worst_i]
        if worst_r > 2.5 * med and worst_r > 0.15 and len(removed) < max_outlier_ratio * n0:
            removed.append(cur[worst_i]['frame'])
            del cur[worst_i]
            continue
        break
    pt = triangulate(cur)
    resid = residuals_of(pt, cur)
    return pt, cur, resid, removed


def split_by_color(rays):
    clusters = []
    for r in rays:
        best_c, best_sim = None, -1
        for c in clusters:
            sim = cv2.compareHist(r['hist'].astype('float32'), c['hist_mean'].astype('float32'), cv2.HISTCMP_CORREL)
            if sim > best_sim:
                best_sim, best_c = sim, c
        if best_c is not None and best_sim >= COLOR_SIM_THRESH:
            best_c['rays'].append(r)
            hs = [x['hist'] for x in best_c['rays']]
            best_c['hist_mean'] = np.mean(hs, axis=0)
        else:
            clusters.append({'rays': [r], 'hist_mean': r['hist'].copy()})
    return [c['rays'] for c in clusters]


hit_indices = sorted(set(r['frame'] for r in all_rays))
visits = []
cur = []
for i in hit_indices:
    if cur and i - cur[-1] > 15:
        visits.append(cur)
        cur = []
    cur.append(i)
if cur:
    visits.append(cur)
print('so lan ghe qua (truoc khi tach mau):', len(visits))

odom_xy_by_orig = {int(f.split('_')[1].split('.')[0]): (odom_by_frame[f]['position']['x'], odom_by_frame[f]['position']['y'])
                    for f in odom_by_frame}

visit_points = []
for v in visits:
    rays = [r for r in all_rays if r['frame'] in v]
    color_groups = split_by_color(rays)
    for g in color_groups:
        if len(g) < 2:
            continue
        pt, kept_rays, resid, removed = robust_triangulate(g)
        if len(kept_rays) < 2:
            continue
        orig_frames = [r['orig_idx'] for r in kept_rays]
        odom_pts = np.array([odom_xy_by_orig[f] for f in orig_frames])
        baseline = float(np.max(np.linalg.norm(odom_pts[:, None, :] - odom_pts[None, :, :], axis=-1))) if len(odom_pts) >= 2 else 0.0
        visit_points.append({
            'point': pt.tolist(), 'n_rays': len(kept_rays),
            'orig_frames': orig_frames,
            'mean_residual': float(np.mean(resid)), 'max_residual': float(np.max(resid)),
            'baseline': baseline,
            'bbox_h': [float(r['bbox_h']) for r in kept_rays],
        })

visit_points.sort(key=lambda vp: vp['mean_residual'])
print()
print('cac diem tam giac dac (chau cay):')
for vp in visit_points:
    print('  n_rays=%2d  mean_resid=%.3fm  baseline=%.3fm  point_xy=(%.3f,%.3f)  orig_frames=%s' % (
        vp['n_rays'], vp['mean_residual'], vp['baseline'],
        vp['point'][0], vp['point'][1], vp['orig_frames']))

with open('/home/khai/semantic-mapping/triangulated_plant_new49.json', 'w') as f:
    json.dump(visit_points, f, indent=2)
print('\nda luu triangulated_plant_new49.json')
