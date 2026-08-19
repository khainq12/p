import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

# du lieu goc: (nhan, diem (x,y), baseline (m), t_trung_binh (s, tinh tu luc bat dau capture))
cap28 = {
    'A(53-81)': ((-1.045, -0.110), 3.61, 67.0),
    'B(11-32)': ((-1.534, -0.216), 1.42, 21.0),
    'C(0-4)':   ((-1.987, -0.710), 0.0, 2.0),
    'D(35-36)': ((-1.116, 0.922), 0.0, 35.5),
    'E(112-125)': ((-1.817, -2.835), 0.0, 118.5),
    'F(123-126)': ((-1.784, -2.802), 0.0, 124.5),
    'G(155-158)': ((-3.365, -3.105), 0.0, 156.5),
}
cap29 = {
    'A(0-9)':    ((1.278, -0.086), 0.86, 4.4),
    'B(29-41)':  ((1.232, -0.078), 0.89, 35.0),
    'C(80-86)':  ((-0.262, 0.140), 0.08, 83.0),
    'D(120-129)': ((-0.179, 0.119), 0.001, 124.5),
}

rows = []
for cap_name, cap in [('C28', cap28), ('C29', cap29)]:
    for (n1, (p1, b1, t1)), (n2, (p2, b2, t2)) in itertools.combinations(cap.items(), 2):
        d = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        min_b = min(b1, b2)
        dt = abs(t1 - t2)
        rows.append((min_b, d, dt, f'{cap_name}: {n1} vs {n2}'))

xs = np.array([r[0] for r in rows])
ys = np.array([r[1] for r in rows])
ts = np.array([r[2] for r in rows])

fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=200)

sizes = 45 + ts * 1.6
sc = ax.scatter(xs, ys, s=sizes, c=ts, cmap='viridis_r', edgecolors='#1a1a1a', linewidths=0.8, alpha=0.88, zorder=3)

ax.axvspan(0.85, xs.max() + 0.3, color='#4ade80', alpha=0.08, zorder=0)
ax.axvline(0.85, color='#4ade80', linestyle='--', linewidth=1.4, zorder=1, label='Ngưỡng baseline = 0,85 m')

ax.set_xlabel('Baseline nhỏ nhất giữa 2 lần quan sát (m)', fontsize=11)
ax.set_ylabel('Sai lệch vị trí giữa 2 lần quan sát (m)', fontsize=11)
ax.set_title('Sai lệch vị trí vs. Baseline tam giác hoá — chậu cây\n'
              f'(tất cả {len(rows)} cặp so sánh, capture 28 + 29)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.25, zorder=0)
ax.set_xlim(-0.15, xs.max() + 0.4)
ax.set_ylim(-0.1, ys.max() + 0.3)

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Khoảng cách thời gian giữa 2 lần (s)', fontsize=9)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig('/home/khai/semantic-mapping/fig_baseline_vs_error.png', dpi=200, bbox_inches='tight')
print('saved fig_baseline_vs_error.png, so cap du lieu:', len(rows))
for r in sorted(rows, key=lambda r: r[0]):
    print(f'  baseline={r[0]:.2f} sai_lech={r[1]:.3f} dt={r[2]:.0f}s  {r[3]}')
