"""
Dinh vi vat the ngu nghia (vd: chau cay) KHONG dung tam giac hoa tia tu
odometry (da chung minh khong dang tin cay do yaw odometry troi khi robot
re/xoay tai cho). Thay vao do, ket hop 2 phuong phap da kiem chung DOC LAP,
khong phu thuoc pose tuyet doi tung khung:

  1. HUONG (bearing): centroid cua cum diem 3D ngu nghia (nhan dien qua
     segmentation, luu trong detections.npz) trong khong gian rieng cua
     MapAnything, quy doi sang toa do that qua GFIT (global similarity fit
     - phep khop TOAN CUC hinh dang quy dao MapAnything voi quy dao
     odometry, it nhay voi nhieu yaw tung khung don le hon nhieu so voi
     tam giac hoa tung tia).
  2. KHOANG CACH (distance): uoc luong don-anh (monocular) tu kich thuoc
     bbox YOLO + chieu cao that da biet cua vat the - hoan toan KHONG dung
     odometry, chi dung 1 khung + intrinsics camera.

Vi tri cuoi = vi tri camera (frame tham chieu) + huong * khoang_cach.

Yeu cau du lieu co san cho capture (da chay MapAnything inference):
  - hawkbot_semantic_output_<CAP>/detections.npz  (points, labels, colors)
  - cam_positions_<CAP>.npy                        (camera pose MapAnything)
  - global_fit_<CAP>.json                          (s, R, t tu compute_scale)
  - hawkbot_capture_<CAP>/poses.json               (odometry)
  - hawkbot_capture_<CAP>/frame_XXXXXX.jpg         (anh goc, de chay YOLO bbox)
"""
import json
import os
import sys
import numpy as np
import cv2

BASE = '/home/khai/semantic-mapping'
INTRINSICS_JSON = f'{BASE}/esp32cam_intrinsics_qvga.json'
CAM_HEIGHT = 0.12
CAM_FORWARD_OFFSET = 0.08


def quat_to_R(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    X, Y, Z, W = x * s, y * s, z * s, w * s
    return np.array([
        [1 - (y * Y + z * Z), x * Y - w * Z, x * Z + w * Y],
        [x * Y + w * Z, 1 - (x * X + z * Z), y * Z - w * X],
        [x * Z - w * Y, y * Z + w * X, 1 - (x * X + y * Y)],
    ])


def semantic_cluster_bearing(cap, coco_class_id, cam_ref_xy, n_sub=30000, eps=0.15, min_samples=15, seed=0):
    """Tra ve vector don vi (huong, khong gian that) tu cam_ref_xy den cum
    diem lon nhat cua lop ngu nghia coco_class_id."""
    from sklearn.cluster import DBSCAN

    npz = np.load(f'{BASE}/hawkbot_semantic_output_{cap}/detections.npz')
    pts = npz['points']
    labels = npz['labels']
    cls_pts = pts[labels == coco_class_id]
    if len(cls_pts) < min_samples:
        raise ValueError(f'qua it diem cho class {coco_class_id}: {len(cls_pts)}')

    rng = np.random.default_rng(seed)
    sub = cls_pts if len(cls_pts) <= n_sub else cls_pts[rng.choice(len(cls_pts), size=n_sub, replace=False)]

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(sub)
    cl = db.labels_
    uniq, counts = np.unique(cl[cl >= 0], return_counts=True)
    if len(uniq) == 0:
        raise ValueError('DBSCAN khong tim duoc cum nao - thu tang eps')
    best = uniq[np.argmax(counts)]
    cluster_pts = sub[cl == best]
    centroid_raw = cluster_pts.mean(axis=0)

    gf = json.load(open(f'{BASE}/global_fit_{cap}.json'))
    s_g, R_g, t_g = gf['s'], np.array(gf['R']), np.array(gf['t'])
    centroid_xy = s_g * (R_g @ centroid_raw[[0, 2]]) + t_g

    direction = centroid_xy - cam_ref_xy
    norm = np.linalg.norm(direction)
    return direction / norm, {'centroid_raw': centroid_raw.tolist(), 'centroid_xy': centroid_xy.tolist(),
                               'cluster_n_points': int(len(cluster_pts)), 'sub_n_points': int(len(sub))}


def monocular_distance(cap, frame_name, coco_class_name, real_height_m, conf_thresh=0.2):
    """Uoc luong khoang cach camera->vat qua kich thuoc bbox YOLO, dung
    chieu cao that da biet. Tra ve (distance_m, bbox_h_px, conf)."""
    from ultralytics import YOLO

    calib = json.load(open(INTRINSICS_JSON))
    fy = calib['K'][1][1]

    img = cv2.imread(f'{BASE}/hawkbot_capture_{cap}/{frame_name}')
    yolo = YOLO('yolo11n.pt')
    cls_id = [k for k, v in yolo.model.names.items() if v == coco_class_name][0]
    res = yolo.predict(source=img, conf=conf_thresh, save=False, verbose=False)
    best_h, best_conf = 0, 0
    for box in res[0].boxes:
        if int(box.cls[0]) == cls_id:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            if (y2 - y1) > best_h:
                best_h = y2 - y1
                best_conf = float(box.conf[0])
    if best_h == 0:
        raise ValueError(f'khong tim thay {coco_class_name} trong {frame_name}')
    best_h = float(best_h)
    return float(fy) * real_height_m / best_h, best_h, best_conf


def localize(cap, coco_class_id, coco_class_name, real_height_m, ref_frame='frame_000000.jpg'):
    poses = json.load(open(f'{BASE}/hawkbot_capture_{cap}/poses.json'))['poses']
    p_ref = [p for p in poses if p['frame'] == ref_frame][0]
    pos, ori = p_ref['position'], p_ref['orientation']
    R_ref = quat_to_R(ori['x'], ori['y'], ori['z'], ori['w'])
    cam_ref_xy = np.array([pos['x'], pos['y']]) + (R_ref @ np.array([CAM_FORWARD_OFFSET, 0, 0]))[:2]

    direction, dbg = semantic_cluster_bearing(cap, coco_class_id, cam_ref_xy)
    distance, bbox_h, conf = monocular_distance(cap, ref_frame, coco_class_name, real_height_m)

    final_xy = cam_ref_xy + direction * distance
    return {
        'final_position_xy': final_xy.tolist(),
        'cam_ref_xy': cam_ref_xy.tolist(),
        'distance_m': distance,
        'direction': direction.tolist(),
        'bbox_h_px': bbox_h,
        'bbox_conf': conf,
        'semantic_cluster_debug': dbg,
    }


if __name__ == '__main__':
    cap = sys.argv[1] if len(sys.argv) > 1 else 'new36'
    result = localize(cap, coco_class_id=58, coco_class_name='potted plant', real_height_m=1.25)
    print(json.dumps(result, indent=2))
