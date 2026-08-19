import json
import sys
import numpy as np

sys.path.insert(0, '/home/khai/semantic-mapping')
import reliability_check as rc

CAM_FORWARD_OFFSET = 0.08
CAM_HEIGHT = 0.12


def quat_to_R(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    X, Y, Z, W = x * s, y * s, z * s, w * s
    return np.array([
        [1 - (y * Y + z * Z), x * Y - w * Z, x * Z + w * Y],
        [x * Y + w * Z, 1 - (x * X + z * Z), y * Z - w * X],
        [x * Z - w * Y, y * Z + w * X, 1 - (x * X + y * Y)],
    ])


for cap in [44, 45, 46, 47, 48, 49]:
    print(f'========== new{cap} ==========')
    cap_dir = f'/home/khai/semantic-mapping/hawkbot_capture_new{cap}'
    poses = json.load(open(f'{cap_dir}/poses.json'))['poses']
    odom_by_idx = {}
    for p in poses:
        idx = int(p['frame'].split('_')[1].split('.')[0])
        odom_by_idx[idx] = p

    tri = json.load(open(f'/home/khai/semantic-mapping/triangulated_plant_new{cap}.json'))
    any_qualify = False
    for vp in tri:
        if vp['baseline'] < 0.15:
            continue
        any_qualify = True
        rays = []
        for f in vp['orig_frames']:
            p = odom_by_idx[f]
            pos = p['position']
            ori = p['orientation']
            R = quat_to_R(ori['x'], ori['y'], ori['z'], ori['w'])
            cam_origin = np.array([pos['x'], pos['y'], pos['z'] + CAM_HEIGHT]) + R @ np.array([CAM_FORWARD_OFFSET, 0, 0])
            rays.append({'origin': cam_origin, 'frame': f})
        bbox_h = vp.get('bbox_h')
        result = rc.evaluate_group(vp['point'], rays, vp['baseline'], bbox_heights=bbox_h)
        dist_start = float(np.linalg.norm(np.array(vp['point'][:2])))
        print('  point=(%.3f,%.3f) cach_xuat_phat=%.3fm n_rays_raw=%d baseline_raw=%.3fm' % (
            vp['point'][0], vp['point'][1], dist_start, vp['n_rays'], vp['baseline']))
        print('    ->', result)
    if not any_qualify:
        print('  (khong co nhom nao baseline >= 0.15m)')
