"""
Dinh vi TUYET DOI vi tri camera bang mã ArUco dan o vi tri BIET TRUOC trong
phong, khong phu thuoc odom (tranh loi troi tich luy da xac nhan qua nhieu
capture). Dung de "neo" lai vi tri that dinh ky trong luc capture.

Cach dung: dan marker ID=0 (DICT_4X4_50, canh 100mm) thang dung, do dac:
  - MARKER_WORLD_X, MARKER_WORLD_Y: tam marker, met, cung he toa do voi odom
  - MARKER_WORLD_YAW_DEG: huong PHAP TUYEN mat truoc marker chi ra, do theo
    cung quy uoc yaw cua odom (0 do = huong +X cua odom)
"""
import json
import numpy as np
import cv2

INTRINSICS_JSON = '/home/khai/semantic-mapping/esp32cam_intrinsics_qvga.json'
MARKER_SIZE_M = 0.150  # canh den that, met (do bang thuoc sau khi in: 15cm)
MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0

# ==== Tinh tu odom THAT o frame_000000 cua new24 (odom lien tuc, KHONG reset
# ve goc moi lan capture) + robot cach marker 1.5m, huong thang vao marker luc
# do: frame0 pos=(-1.0674,0.4658), yaw=143.69 deg =>
MARKER_WORLD_X = -2.276107023374141
MARKER_WORLD_Y = 1.3540721045074104
MARKER_WORLD_YAW_DEG = 323.6880395361055
# ================================================


def load_K():
    calib = json.load(open(INTRINSICS_JSON))
    K = np.array(calib['K'], dtype=np.float32)
    dist = np.array(calib['dist'], dtype=np.float32)
    return K, dist


def detect_and_localize(img, K, dist, marker_world_x, marker_world_y, marker_world_yaw_deg):
    """Tra ve (cam_x, cam_y, cam_yaw_deg) neu thay marker, None neu khong."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or MARKER_ID not in ids.flatten():
        return None

    idx = list(ids.flatten()).index(MARKER_ID)
    c = corners[idx][0]  # (4,2): top-left, top-right, bottom-right, bottom-left

    L = MARKER_SIZE_M / 2
    obj_pts = np.array([
        [-L, L, 0], [L, L, 0], [L, -L, 0], [-L, -L, 0],
    ], dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, c, K, dist)
    if not ok:
        return None

    R_marker_to_cam, _ = cv2.Rodrigues(rvec)
    t_marker_to_cam = tvec.reshape(3)

    # vi tri camera trong he toa do marker (marker la goc)
    cam_in_marker = -R_marker_to_cam.T @ t_marker_to_cam

    # marker mounted THANG DUNG (khong roll/pitch), Y_m=len tren trung voi
    # world-up. X_m=ngang mat marker (phai, nhin tu truoc marker), Z_m=phap
    # tuyen mat marker, huong ra phia camera (X_m x Y_m = Z_m, he thuan tay
    # phai). yaw = huong cua Z_m trong world (0 do = +X cua odom).
    # => Z_m_world = (cos(yaw), sin(yaw)), X_m_world = Z_m_world xoay +90 do
    #    = (-sin(yaw), cos(yaw))
    yaw = np.radians(marker_world_yaw_deg)
    X_m, Z_m = cam_in_marker[0], cam_in_marker[2]
    cam_world_xy = (
        Z_m * np.array([np.cos(yaw), np.sin(yaw)])
        + X_m * np.array([-np.sin(yaw), np.cos(yaw)])
        + np.array([marker_world_x, marker_world_y])
    )

    return {
        'cam_x': float(cam_world_xy[0]),
        'cam_y': float(cam_world_xy[1]),
        'dist_to_marker': float(np.linalg.norm(t_marker_to_cam)),
        'corners_px': c.tolist(),
    }


if __name__ == '__main__':
    import sys
    K, dist = load_K()
    print('K:\n', K)
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if MARKER_WORLD_X is None:
            print('CHUA DIEN vi tri marker that (MARKER_WORLD_X/Y/YAW) - chi test detect')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
            detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(gray)
            print('ids phat hien:', ids)
        else:
            r = detect_and_localize(img, K, dist, MARKER_WORLD_X, MARKER_WORLD_Y, MARKER_WORLD_YAW_DEG)
            print(r)
