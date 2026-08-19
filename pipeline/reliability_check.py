"""
Ham kiem tra do tin cay tu dong cho 1 nhom tia da tam giac hoa, dua tren
phat hien: baseline (met) khong du, can kiem tra ca (1) do on dinh cua
khoang cach camera->diem qua tung khung, (2) goc thi sai thuc su giua
tia dau va tia cuoi tai diem giao nhau, (3) cac vi tri camera gan trung
nhau khong duoc tinh la tia doc lap (chong "gia lap tia" khi robot dung
yen nhieu khung lien tiep), va (4) doi chieu khoang cach tam giac hoa voi
uoc luong don-anh tu kich thuoc bbox (dua vao chieu cao that cua vat) -
day la cach duy nhat phat hien truong hop nghiem LSQ bi "hut" ve gan
camera do nhieu goc lan at tin hieu thi sai that (baseline nho so voi
khoang cach that).
"""
import numpy as np

# hieu chuan tu capture 36 (chau cay that cao 1.25m, da xac thuc bang
# thuoc day + tam giac hoa cung tron + don-anh, ca 3 cung hoi tu ~3.00m)
FY_PX = 427.60
REAL_HEIGHT_M = 1.25
FY_H = FY_PX * REAL_HEIGHT_M  # px*m, dist = FY_H / bbox_h_px

DEDUP_RADIUS_M = 0.05  # vi tri camera gan hon nguong nay coi la "cung 1 cho"
BBOX_DIST_RATIO_MAX = 2.0  # lech qua 2 lan so voi don-anh -> loai


def dedup_origins(rays, radius=DEDUP_RADIUS_M):
    """Gop cac tia co goc (vi tri camera) gan trung nhau thanh 1 nhom dai
    dien (tia dau tien trong cum), tra ve danh sach tia da loc.
    Muc dich: tranh dem nhieu khung hinh lien tiep luc robot dung yen la
    nhieu quan sat doc lap, vi chung khong mang them thong tin thi sai."""
    kept = []
    used = [False] * len(rays)
    for i, r in enumerate(rays):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(rays)):
            if used[j]:
                continue
            if np.linalg.norm(np.array(rays[i]['origin']) - np.array(rays[j]['origin'])) < radius:
                cluster.append(j)
                used[j] = True
        kept.append(rays[cluster[0]])
    return kept


def check_distance_stability(point, rays):
    """rays: list of {'origin': np.array(3,), 'frame': int}
    Tra ve (is_stable: bool, cv: float, trend_slope: float)
    cv = coefficient of variation (std/mean) cua khoang cach camera->diem
    trend_slope = do doc xu huong (hoi quy tuyen tinh) chuan hoa theo mean
    """
    dists = np.array([np.linalg.norm(np.array(point) - r['origin']) for r in rays])
    mean_d = dists.mean()
    if mean_d < 1e-6:
        return False, np.inf, np.inf
    cv = dists.std() / mean_d

    t = np.arange(len(dists))
    if len(t) >= 3:
        slope = np.polyfit(t, dists, 1)[0]
        norm_slope = abs(slope) * len(dists) / mean_d  # tong bien thien / trung binh
    else:
        norm_slope = 0.0

    # nguong: bien thien tuong doi duoi 20% VA khong co xu huong don dieu manh
    is_stable = (cv < 0.20) and (norm_slope < 0.35)
    return is_stable, float(cv), float(norm_slope)


def parallax_angle_deg(point, ray_a_origin, ray_b_origin):
    """Goc thi sai (do) tai diem giao nhau, giua 2 tia xa nhau nhat trong nhom."""
    p = np.array(point)
    va = p - np.array(ray_a_origin)
    vb = p - np.array(ray_b_origin)
    va_n = va / (np.linalg.norm(va) + 1e-9)
    vb_n = vb / (np.linalg.norm(vb) + 1e-9)
    cos_ang = np.clip(np.dot(va_n, vb_n), -1.0, 1.0)
    return np.degrees(np.arccos(cos_ang))


def bbox_distance_check(point, rays, bbox_heights):
    """So sanh khoang cach camera->diem (tam giac hoa) voi khoang cach
    don-anh tu kich thuoc bbox (dist = FY_H / bbox_h), cho tung khung.
    Tra ve (is_consistent, tri_dist_median, mono_dist_median, ratio)."""
    if not bbox_heights or len(bbox_heights) != len(rays):
        return True, None, None, None  # khong co du lieu bbox -> bo qua kiem tra nay
    tri_dists = np.array([np.linalg.norm(np.array(point) - r['origin']) for r in rays])
    mono_dists = np.array([FY_H / h for h in bbox_heights if h > 1e-3])
    if len(mono_dists) == 0:
        return True, None, None, None
    tri_med = float(np.median(tri_dists))
    mono_med = float(np.median(mono_dists))
    if min(tri_med, mono_med) < 1e-6:
        return False, tri_med, mono_med, np.inf
    ratio = max(tri_med, mono_med) / min(tri_med, mono_med)
    return (ratio <= BBOX_DIST_RATIO_MAX), tri_med, mono_med, float(ratio)


def evaluate_group(point, rays, baseline, min_parallax_deg=15.0, bbox_heights=None):
    """Tong hop: tra ve dict ket qua danh gia do tin cay cho 1 nhom.
    rays da sap xep theo thu tu khung hinh (thoi gian).
    bbox_heights (tuy chon): list chieu cao bbox (px) cung thu tu voi rays,
    dung de doi chieu khoang cach doc lap qua kich thuoc vat."""
    rays_dedup = dedup_origins(rays)
    n_effective = len(rays_dedup)

    is_stable, cv, norm_slope = check_distance_stability(point, rays_dedup)

    origins = [r['origin'] for r in rays_dedup]
    max_d = -1
    best_pair = (0, 0)
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            d = np.linalg.norm(np.array(origins[i]) - np.array(origins[j]))
            if d > max_d:
                max_d = d
                best_pair = (i, j)
    angle = parallax_angle_deg(point, origins[best_pair[0]], origins[best_pair[1]]) if n_effective >= 2 else 0.0
    effective_baseline = float(max_d) if max_d >= 0 else 0.0

    bbox_ok, tri_med, mono_med, ratio = bbox_distance_check(point, rays, bbox_heights)

    enough_independent = n_effective >= 3

    reliable = (
        is_stable and (angle >= min_parallax_deg) and (effective_baseline >= 0.3)
        and enough_independent and bbox_ok
    )

    if reliable:
        reason = 'OK'
    elif not enough_independent:
        reason = 'qua it vi tri camera doc lap sau khi gop trung (%d < 3, tu %d tia goc)' % (n_effective, len(rays))
    elif not is_stable:
        reason = 'khoang cach khong on dinh (cv=%.2f, trend=%.2f)' % (cv, norm_slope)
    elif angle < min_parallax_deg:
        reason = 'goc thi sai qua nho (%.1f deg)' % angle
    elif effective_baseline < 0.3:
        reason = 'baseline thuc su (sau gop trung) qua nho (%.2fm, goc %.2fm)' % (effective_baseline, baseline)
    elif not bbox_ok:
        reason = 'lech voi uoc luong don-anh tu bbox (tam giac=%.2fm vs don-anh=%.2fm, ty le=%.1fx)' % (tri_med, mono_med, ratio)
    else:
        reason = 'khong xac dinh'

    return {
        'reliable': bool(reliable),
        'distance_cv': cv,
        'distance_trend_slope_norm': norm_slope,
        'parallax_angle_deg': float(angle),
        'baseline_raw': baseline,
        'baseline_effective': effective_baseline,
        'n_rays_raw': len(rays),
        'n_rays_effective': n_effective,
        'mono_check_ratio': ratio,
        'tri_dist_m': tri_med,
        'mono_dist_m': mono_med,
        'reason': reason,
    }


if __name__ == '__main__':
    point = [0.0, 0.0, 0.0]
    rays_arc = [{'origin': np.array([3 * np.cos(a), 3 * np.sin(a), 0]), 'frame': i}
                for i, a in enumerate(np.linspace(-0.3, 0.3, 8))]
    print('Cung tron:', evaluate_group(point, rays_arc, baseline=1.78))

    rays_radial = [{'origin': np.array([3 - 0.34 * i, 0.05 * i, 0]), 'frame': i}
                   for i in range(8)]
    print('Xuyen tam:', evaluate_group(point, rays_radial, baseline=1.78))

    # gia lap dung y het truong hop new49: 8 vi tri gan trung nhau + 2 vi tri
    # lech nhe, nhung diem tam giac hoa bi hut ve gan (0.29m) trong khi bbox
    # noi that (~3.2m)
    origins_bug = [np.array([0.076, -1.873, 0.12])] * 8 + [np.array([0.397, -1.930, 0.12])] * 2
    point_bug = [0.297, -1.580, 0.12]
    rays_bug = [{'origin': o, 'frame': i} for i, o in enumerate(origins_bug)]
    bbox_h_bug = [165, 165.7, 165.6, 172.5, 172.2, 172.0, 171.8, 161.9, 170.7, 168.4]
    print('Truong hop new49 (gia lap):', evaluate_group(point_bug, rays_bug, baseline=0.326, bbox_heights=bbox_h_bug))
