"""Figure 3.1: training coverage curves, drawn with the repository's plotting conventions
(src/extract_and_plot_coverage.py: its model colours and line styles, 25-point rolling mean, and its
coverage_curves_cache.json of logged training curves). The MAPPO Self-healing curve in the cache starts at
2.0M steps; its first 1.97M steps are taken from that run's RLlib progress.csv
(MAPPO_SelfHeal_comparison_progress.csv, column env_runners/custom_metrics/coverage_rate_mean), and the two
segments are joined by a straight line.
Usage (from the repository root):
  python3 analysis/repo_figures/make_fig31.py <training_logs_dir> <out_dir>"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import extract_and_plot_coverage as ep

logs, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
data = ep.load_data()

early = pd.read_csv(os.path.join(logs, "MAPPO_SelfHeal_comparison_progress.csv"))
early = early.dropna(subset=["env_runners/custom_metrics/coverage_rate_mean"])
k = ep.rllib_models["MAPPO SelfHeal"]["cache_key"]
s0 = early["num_env_steps_sampled"].to_numpy() / 1e6
v0 = early["env_runners/custom_metrics/coverage_rate_mean"].to_numpy()
if np.nanmax(v0) <= 1.0:
    v0 = v0 * 100
# each segment is smoothed on its own, then the two are joined by a straight line
seg = {k: (s0, ep.smooth_curve(v0, window=25))}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.6), sharey=True)
fig.suptitle("Multi-Agent Training Learning Curves: Mean Area Coverage Rate\n"
             "Standard 25×25 Environment ($p = 0.0005$) | 20M Environment Steps",
             fontsize=14.5, fontweight="bold", y=0.98)
for ax, models, title in ((ax1, ep.rllib_models, "RLlib Framework"), (ax2, ep.epymarl_models, "EPyMARL Suite")):
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=12)
    ax.set_xlabel("Environment Steps (Millions)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 20); ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    for name, info in models.items():
        d = data[info["cache_key"]]
        x, y = np.array(d["steps"]), ep.smooth_curve(np.array(d["vals"]), window=25)
        if info["cache_key"] in seg:
            x = np.concatenate([seg[info["cache_key"]][0], x]); y = np.concatenate([seg[info["cache_key"]][1], y])
        ax.plot(x, y,
                color=info["color"], linewidth=info["lw"], linestyle=info["ls"], label=name)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=10.2, framealpha=0.96, edgecolor="#cccccc")
ax1.set_ylabel("Mean Area Coverage Rate (%)", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(os.path.join(out, "fig_3_1_training_curves.png"), dpi=300)
print("wrote", os.path.join(out, "fig_3_1_training_curves.png"))
