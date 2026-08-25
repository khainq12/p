import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sessions = [1, 2, 3, 4, 5, 6]
dist = [2.77, 2.76, 3.13, 3.08, 3.26, 3.12]
gt = 3.00
std = 0.19

GREEN = '#1a8f5e'

fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=220)

ax.axhspan(gt - std, gt + std, color=GREEN, alpha=0.10, zorder=1)
ax.bar(sessions, dist, color=GREEN, edgecolor='#0d4a2e', linewidth=1.0, width=0.65, zorder=2)
ax.axhline(gt, color='#222', linestyle='--', linewidth=1.6, zorder=3, label=f'Ground truth: {gt:.2f} m')

for x, y in zip(sessions, dist):
    ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, 5),
                ha='center', fontsize=10.5)

ax.set_xlabel('Capture session', fontsize=11)
ax.set_ylabel('Estimated distance (m)', fontsize=11)
ax.set_xticks(sessions)
ax.set_ylim(0, 3.8)
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)

fig.tight_layout()
fig.savefig('/tmp/claude-1000/-home-ubuntu-Desktop/7015d3ad-0542-4555-9746-b94c4b9a9420/scratchpad/fig_results_bar_en.png',
            dpi=220, bbox_inches='tight', facecolor='white')
print('saved fig_results_bar_en.png')
