#!/usr/bin/env python3
"""Generate figures for the Fragility paper using actual experimental data."""
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- Figure 1: Non-transitivity directed graph ----
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Triangle vertices
angles = np.array([90, -30, -150]) * np.pi / 180
r = 1.1
nodes = {
    'CfC+ACT\n(A)': (r * np.cos(angles[0]), r * np.sin(angles[0])),
    'Random':        (r * np.cos(angles[1]), r * np.sin(angles[1])),
    'GRU\n(C)':     (r * np.cos(angles[2]), r * np.sin(angles[2])),
}

# Edges: (source, target, label)
edges = [
    ('CfC+ACT\n(A)', 'Random', '96%'),
    ('Random', 'GRU\n(C)', '89%'),
    ('GRU\n(C)', 'CfC+ACT\n(A)', '100%'),
]

for src, dst, label in edges:
    x1, y1 = nodes[src]
    x2, y2 = nodes[dst]
    # Offset arrow slightly from center
    dx, dy = x2 - x1, y2 - y1
    d = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / d, dy / d
    # Shift start/end points for node radius
    r_node = 0.25
    xs = x1 + dx * r_node
    ys = y1 + dy * r_node
    xe = x2 - dx * r_node
    ye = y2 - dy * r_node
    ax.annotate('',
                xy=(xe, ye), xytext=(xs, ys),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#2c3e50'),
                zorder=3)
    # Label at midpoint with offset
    mx, my = (xs + xe) / 2, (ys + ye) / 2
    # Offset perpendicular to edge
    px, py = -dy, dx
    offset = 0.18
    ax.text(mx + px * offset, my + py * offset, label,
            fontsize=11, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#bdc3c7', alpha=0.9))

# Draw nodes
for name, (x, y) in nodes.items():
    color = {'CfC+ACT\n(A)': '#e74c3c', 'Random': '#95a5a6', 'GRU\n(C)': '#2ecc71'}[name]
    ax.plot(x, y, 'o', markersize=45, markeredgecolor='#2c3e50',
            markeredgewidth=2, markerfacecolor=color, zorder=4)
    ax.text(x, y, name, fontsize=9, ha='center', va='center',
            fontweight='bold', color='white', zorder=5)

ax.set_title('Rock-Paper-Scissors Dynamics\n(Non-Transitive Cycle)',
             fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('fig1_nontransitivity.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()
print("fig1_nontransitivity.pdf generated")

# ---- Figure 2: Transfer Learning Curves ----
with open('transfer_results.json') as f:
    data = json.load(f)

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

colors = {'transfer': '#e74c3c', 'baseline': '#3498db'}
labels = {'transfer': 'Transfer (TTT $\to$ C4)', 'baseline': 'From scratch'}

for entry in data:
    label = entry['label']
    games = [m['game'] for m in entry['metrics'] if m.get('note') is None]
    wrs = [m['wr'] * 100 for m in entry['metrics'] if m.get('note') is None]
    ax.plot(games, wrs, color=colors[label], linewidth=1.8, alpha=0.7, label=labels[label])
    # Rolling average for smoothness
    window = 5
    games_arr = np.array(games)
    wrs_arr = np.array(wrs)
    if len(wrs_arr) >= window:
        kernel = np.ones(window) / window
        wrs_smooth = np.convolve(wrs_arr, kernel, mode='valid')
        games_smooth = games_arr[window//2:-(window//2)] if window % 2 == 1 else games_arr[window//2:-(window//2-1)]
        ax.plot(games_smooth, wrs_smooth, color=colors[label], linewidth=2.5, alpha=1.0)

# Annotate peak and final
ax.annotate('Peak 70%', xy=(2600, 70), xytext=(2300, 82),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
            fontsize=10, fontweight='bold', color='#c0392b')
ax.annotate('Collapse\nto 23.3%', xy=(3000, 23.3), xytext=(3100, 45),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
            fontsize=10, fontweight='bold', color='#c0392b',
            ha='center')
ax.annotate('Baseline\n80.0%', xy=(3000, 80), xytext=(3100, 80),
            fontsize=10, fontweight='bold', color='#2980b9',
            ha='center', va='center')

ax.axhline(y=23.3, color='#e74c3c', linestyle=':', alpha=0.4, linewidth=1)
ax.axhline(y=80.0, color='#3498db', linestyle=':', alpha=0.4, linewidth=1)

ax.set_xlabel('Training Game', fontsize=12)
ax.set_ylabel('Win Rate (%)', fontsize=12)
ax.set_title('Transfer Learning: TTT $\\longrightarrow$ Connect Four',
             fontsize=13, fontweight='bold', pad=8)
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(0, 3500)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_transfer_curves.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()
print("fig2_transfer_curves.pdf generated")
