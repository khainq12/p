import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

labels = ['Session 1', 'Session 2', 'Session 3', 'Session 4', 'Session 5', 'Session 6']
dist = [2.77, 2.76, 3.13, 3.08, 3.26, 3.12]
gt = 3.00
std = 0.19

angles_deg = [0, 60, 120, 180, 240, 300]
angles_rad = np.deg2rad(angles_deg)

GREEN = '#1a8f5e'
DARK = '#0d4a2e'

fig = plt.figure(figsize=(7.2, 7.0), dpi=220)
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

max_r = 4.0
ax.set_ylim(0, max_r)
ax.set_rticks([1, 2, 3])
ax.set_rlabel_position(200)
ax.set_yticklabels(['1 m', '2 m', '3 m'], fontsize=9.5, color='#555')
ax.set_xticklabels([])
ax.grid(True, alpha=0.3, linewidth=0.8)
ax.spines['polar'].set_alpha(0.4)

theta_full = np.linspace(0, 2*np.pi, 200)
ax.fill_between(theta_full, gt-std, gt+std, color=GREEN, alpha=0.10, zorder=1)
ax.plot(theta_full, [gt]*len(theta_full), '--', color='#222', linewidth=1.6, zorder=2)

ax.plot(0, 0, marker='o', markersize=10, color='#222', zorder=5)
ax.annotate('Plant pot\n(true position)', xy=(0, 0), xytext=(np.deg2rad(30), 0.95),
            textcoords='data', fontsize=9.5, ha='left', va='center', color='#222', fontweight='bold')

def label_align(deg):
    if deg in (0,):
        return 'center', 'bottom'
    if deg in (180,):
        return 'center', 'top'
    if deg < 180:
        return 'left', 'center'
    return 'right', 'center'

for a, deg, r, lab in zip(angles_rad, angles_deg, dist, labels):
    ax.plot([a], [r], marker='^', markersize=16, color=GREEN, markeredgecolor=DARK,
             markeredgewidth=1.3, zorder=4)
    err_pct = (r - gt) / gt * 100
    ha, va = label_align(deg)
    dx = 0.28 if deg not in (0, 180) else 0
    label_theta = a
    label_r = r + 0.5
    ax.annotate(f'{lab}: {r:.2f} m ({err_pct:+.1f}%)', xy=(a, r), xytext=(label_theta, label_r),
                textcoords='data', fontsize=9.5, ha=ha, va=va, color=DARK, fontweight='bold')

legend_elems = [
    plt.Line2D([0], [0], color='#222', linestyle='--', linewidth=1.6, label=f'Ground truth (measured): {gt:.2f} m'),
    plt.Rectangle((0,0),1,1, facecolor=GREEN, alpha=0.10, label=f'Standard deviation (±{std:.2f} m)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=GREEN, markeredgecolor=DARK,
               markersize=12, label='Estimated position (6 independent sessions)'),
]
ax.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, -0.06), fontsize=9.5, frameon=False, ncol=1)

ax.set_title('distance = radius; angle only separates the 6 points for readability (does not represent true bearing)',
              fontsize=10, color='#666', pad=14)

fig.tight_layout(rect=[0, 0.02, 1, 0.99])
fig.savefig('/tmp/claude-1000/-home-ubuntu-Desktop/7015d3ad-0542-4555-9746-b94c4b9a9420/scratchpad/fig_dartboard_v2.png',
            dpi=220, bbox_inches='tight', facecolor='white')
print('saved fig_dartboard_v2.png')
