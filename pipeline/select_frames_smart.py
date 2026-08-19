#!/usr/bin/env python3
"""
Chon 130 khung hinh THONG MINH tu toan bo frames da capture, thay vi
lay ngau nhien deu (np.linspace) nhu truoc - tranh truong hop lay mau
ngau nhien bo lot cac khung co chai/vat the quan trong.

Cach lam:
1. Chay YOLO tren TOAN BO khung hinh goc (khong phai chi 130 khung).
2. Voi moi lop quan trong (mac dinh: bottle), gom cac khung phat hien
   duoc thanh tung "lan ghe qua" (nhom theo thoi gian lien tuc), roi
   LAY DEU trong tung lan ghe qua (toi da N_PER_VISIT khung/lan) de dam
   bao co nhieu goc nhin khac nhau cho cung 1 vat (tang baseline, giam
   loi tam giac dac gan - dung nguyen nhan da phat hien voi Chai 3).
3. Phan con lai cua ngan sach 130 khung -> lay deu (linspace) tren TOAN
   BO trajectory de dam bao bao phu chung ca phong (cac vat khac, tuong,
   do noi that...).

Usage: python3 select_frames_smart.py --frames_dir DIR --out_dir OUT
       [--target_total 130] [--classes bottle] [--conf 0.25]
       [--n_per_visit 4] [--gap_sec 3.0]
"""
import argparse
import json
import os
import shutil

import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_dir', required=True, help='Thu muc chua TOAN BO khung hinh goc')
    ap.add_argument('--out_dir', required=True, help='Thu muc dich se tao symlink 130 khung da chon')
    ap.add_argument('--target_total', type=int, default=130)
    ap.add_argument('--classes', nargs='+', default=['bottle'])
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--n_per_visit', type=int, default=4,
                     help='So khung toi da lay tu moi lan robot ghe qua 1 vat, de tang goc nhin')
    ap.add_argument('--gap_frames', type=int, default=8,
                     help='So khung lien tiep KHONG phat hien duoc thi coi la ket thuc 1 lan ghe qua')
    ap.add_argument('--yolo_model', default='yolo11n.pt')
    args = ap.parse_args()

    frame_files = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith(('.jpg', '.png')))
    n_total = len(frame_files)
    print(f'Tong so khung goc: {n_total}')

    print('Dang chay YOLO tren toan bo khung hinh de tim frame co vat quan trong...')
    yolo = YOLO(args.yolo_model)
    target_ids = {i for i, name in yolo.model.names.items() if name in args.classes}
    if not target_ids:
        raise ValueError(f'Khong tim thay lop {args.classes} trong model')

    hit_indices = []
    for i, fname in enumerate(frame_files):
        res = yolo.predict(source=os.path.join(args.frames_dir, fname), conf=args.conf, save=False, verbose=False)
        if res[0].boxes is not None:
            cls_ids = set(int(c) for c in res[0].boxes.cls.cpu().numpy())
            if cls_ids & target_ids:
                hit_indices.append(i)
        if (i + 1) % 50 == 0:
            print(f'  ... da xu ly {i+1}/{n_total} khung, tim thay {len(hit_indices)} khung co vat muc tieu')

    print(f'Tong so khung phat hien duoc {args.classes}: {len(hit_indices)}')

    # gom thanh tung "lan ghe qua" theo khoang cach frame lien tuc
    visits = []
    cur = []
    for idx in hit_indices:
        if cur and idx - cur[-1] > args.gap_frames:
            visits.append(cur)
            cur = []
        cur.append(idx)
    if cur:
        visits.append(cur)
    print(f'So lan ghe qua vat rieng biet: {len(visits)}')

    selected = set()
    for v in visits:
        if len(v) <= args.n_per_visit:
            selected.update(v)
        else:
            # lay deu trong lan ghe qua nay de toi da hoa do trai (baseline) goc nhin
            picks = np.linspace(0, len(v) - 1, args.n_per_visit).astype(int)
            selected.update(v[p] for p in picks)
    print(f'So khung uu tien (co vat muc tieu, da lay mau deu trong tung lan ghe qua): {len(selected)}')

    remaining_budget = max(args.target_total - len(selected), 0)
    if remaining_budget > 0:
        all_idx = np.linspace(0, n_total - 1, remaining_budget + len(selected)).astype(int)
        for idx in all_idx:
            if len(selected) >= args.target_total:
                break
            selected.add(int(idx))
    else:
        print(f'CANH BAO: so khung uu tien ({len(selected)}) da vuot ngan sach {args.target_total} - '
              f'se cat bot deu trong tung lan ghe qua')
        # neu vuot ngan sach, giam bot deu tu moi lan ghe qua
        selected = set()
        n_visits = len(visits)
        per_visit_budget = max(args.target_total // max(n_visits, 1), 1)
        for v in visits:
            picks = np.linspace(0, len(v) - 1, min(per_visit_budget, len(v))).astype(int)
            selected.update(v[p] for p in picks)

    final_indices = sorted(selected)[:args.target_total]
    print(f'TONG SO KHUNG CUOI CUNG DUOC CHON: {len(final_indices)}')

    os.makedirs(args.out_dir, exist_ok=True)
    for idx in final_indices:
        src = os.path.abspath(os.path.join(args.frames_dir, frame_files[idx]))
        dst = os.path.join(args.out_dir, frame_files[idx])
        if not os.path.exists(dst):
            os.symlink(src, dst)

    meta = {
        'n_total_frames': n_total,
        'n_hit_frames': len(hit_indices),
        'n_visits': len(visits),
        'selected_indices': final_indices,
        'target_classes': args.classes,
    }
    with open(os.path.join(args.out_dir, 'selection_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Da tao {len(final_indices)} symlink trong {args.out_dir}')


if __name__ == '__main__':
    main()
