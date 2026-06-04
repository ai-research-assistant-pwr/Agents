import itertools

import matplotlib.pyplot as plt

# --- DATA ------------------------------------------------------------------
# Replace this with your actual data.
# Format: list of (method_name: str, relevancy_score: float, diversity_score: float)
# Both scores on a 0-4 scale.
data = [
    ("Baseline", 3.03, 2.8),
    ("BFS", 3.0, 2.78),
    ("Random Walk", 3.1, 3.0),
    ("Personalized PageRank", 3.15, 2.9),
    ("Agent", 3.125, 2.8),
    ("Agent PPR", 3.18, 2.8),
    ("Agent BFS", 3.06, 2.9),
    ("Agent RW", 3.15, 3.0),
    ("Agent PPR RW", 3.25, 2.8),
    ("Agent PPR BFS", 3.18, 2.8),
    ("Agent BFS RW", 2.96, 2.6),
]
# ---------------------------------------------------------------------------

methods = sorted({d[0] for d in data})
colors = {m: c for m, c in zip(methods, itertools.cycle(plt.cm.Set1.colors))}

plt.figure(figsize=(8, 6))

for method in methods:
    group = [d for d in data if d[0] == method]
    xs = [d[1] for d in group]
    ys = [d[2] for d in group]
    plt.scatter(
        xs,
        ys,
        c=[colors[method]],
        marker="o",
        label=method,
        s=40,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

plt.xlabel("Relevancy", fontsize=12)
plt.ylabel("Diversity", fontsize=12)
plt.xlim(2, 4)
plt.ylim(2, 4)
plt.xticks([2.0, 2.5, 3.0, 3.5, 4.0])
plt.yticks([2.0, 2.5, 3.0, 3.5, 4.0])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Method", fontsize=9, title_fontsize=10, loc="lower left")
plt.tight_layout()

output_path = "relevancy_diversity.png"
plt.savefig(output_path, dpi=150)
print(f"Saved to {output_path}")

plt.show()
