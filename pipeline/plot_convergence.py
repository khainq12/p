import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (capture, best distance estimate, motion quality tag)
data = [
    (29, 1.26, 'radial'),
    (32, 0.95, 'radial'),
    (33, 1.84, 'mixed'),
    (34, 2.05, 'mixed'),
    (35, 2.30, 'mixed'),
    (36, 2.97, 'arc'),
]

colors = {'radial': '#ef4444', 'mixed': '#f59e0b', 'arc': '#22c55e'}
labels = {'radial': 'Predominantly radial motion', 'mixed': 'Partial arc motion', 'arc': 'Clean arc motion'}

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)

seen = set()
for cap, dist, tag in data:
    lbl = labels[tag] if tag not in seen else None
    seen.add(tag)
    ax.scatter([cap], [dist], s=180, c=colors[tag], edgecolors='black', linewidths=1.3, zorder=5, label=lbl)

xs = [d[0] for d in data]
ys = [d[1] for d in data]
ax.plot(xs, ys, '-', color='#9ca3af', linewidth=1.3, zorder=2, alpha=0.7)

ax.axhline(3.00, color='#374151', linestyle='--', linewidth=1.5, zorder=1, label='Tape-measured ground truth: 3.00m')

for cap, dist, tag in data:
    ax.annotate(f'{dist:.2f}m', (cap, dist), textcoords='offset points', xytext=(0, 12), ha='center', fontsize=9)

ax.set_xlabel('Capture number (successive attempts, same physical target)', fontsize=11)
ax.set_ylabel('Estimated distance from start (m)', fontsize=11)
ax.set_title('Convergence toward ground truth as robot motion\nshifted from radial approach to arc motion', fontsize=12, fontweight='bold')
ax.set_xticks(xs)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 3.5)

fig.tight_layout()
fig.savefig('/home/khai/semantic-mapping/fig_convergence.png', dpi=200, bbox_inches='tight')
print('saved fig_convergence.png')
