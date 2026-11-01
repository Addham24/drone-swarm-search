import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE = os.path.join(SCRIPT_DIR, "coverage_curves_cache.json")
FALLBACK_CACHE = "/Users/AdhamMotawi/.gemini/antigravity-ide/brain/261d090f-67ca-4f3a-8220-dd983c502482/scratch/coverage_curves_cache.json"
CACHE_FILE = LOCAL_CACHE if os.path.exists(LOCAL_CACHE) else FALLBACK_CACHE

# Color Palette: Each algorithm family has ONE unique dedicated hue
# Solid (-) = Self-Healing | Dashed (--) = Vanilla
rllib_models = {
    "RSPO SelfHeal": {
        "cache_key": "RSPO SelfHeal (Ours)",
        "color": "#D35400",  # Strong burnt orange
        "lw": 2.5,
        "ls": "-"
    },
    "RSPO Vanilla": {
        "cache_key": "RSPO Vanilla (Ours)",
        "color": "#D35400",  # Same burnt orange
        "lw": 2.1,
        "ls": "--"
    },
    "MAPPO SelfHeal": {
        "cache_key": "MAPPO SelfHeal",
        "color": "#1E88E5",  # Vibrant Blue (same as MAPPO Vanilla)
        "lw": 2.3,
        "ls": "-"
    },
    "MAPPO Vanilla": {
        "cache_key": "MAPPO Vanilla",
        "color": "#1E88E5",  # Vibrant Blue
        "lw": 2.1,
        "ls": "--"
    },
    "I-DQN SelfHeal": {
        "cache_key": "I-DQN SelfHeal",
        "color": "#8E24AA",  # Amethyst Purple
        "lw": 2.2,
        "ls": "-"
    },
    "I-DQN Vanilla": {
        "cache_key": "I-DQN Vanilla",
        "color": "#8E24AA",  # Same amethyst purple
        "lw": 2.0,
        "ls": "--"
    }
}

epymarl_models = {
    "QMIX SelfHeal": {
        "cache_key": "QMIX SelfHeal",
        "color": "#2E7D32",  # Deep Forest Green
        "lw": 2.3,
        "ls": "-"
    },
    "QMIX Vanilla": {
        "cache_key": "QMIX Vanilla",
        "color": "#2E7D32",  # Same deep forest green
        "lw": 2.0,
        "ls": "--"
    },
    "COMA SelfHeal": {
        "cache_key": "COMA SelfHeal",
        "color": "#D32F2F",  # Crimson Red
        "lw": 2.2,
        "ls": "-"
    },
    "COMA Vanilla": {
        "cache_key": "COMA Vanilla",
        "color": "#D32F2F",  # Same crimson red
        "lw": 1.9,
        "ls": "--"
    },
    "MAA2C SelfHeal": {
        "cache_key": "MAA2C SelfHeal",
        "color": "#B78103",  # Deep Golden Amber
        "lw": 2.2,
        "ls": "-"
    },
    "MAA2C Vanilla": {
        "cache_key": "MAA2C Vanilla",
        "color": "#B78103",  # Same deep golden amber
        "lw": 1.9,
        "ls": "--"
    }
}

def load_data():
    with open(CACHE_FILE, "r") as f:
        return json.load(f)

def smooth_curve(vals, window=25):
    s = pd.Series(vals)
    return s.rolling(window=window, min_periods=1, center=False).mean().values

def main():
    data = load_data()
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12.5,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "figure.titlesize": 15
    })
    
    art_dir = "/Users/AdhamMotawi/.gemini/antigravity-ide/brain/261d090f-67ca-4f3a-8220-dd983c502482"
    desk_dir = "/Users/AdhamMotawi/Desktop/Results"
    
    # =========================================================================
    # 1. UNIFIED PLOT (ALL 12 MODELS ON ONE CANVAS - UNIQUE FAMILY COLORS)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(15.2, 8.2))
    
    plt.suptitle("Multi-Agent Training Learning Curves: Mean Area Coverage Rate",
                 fontsize=15.5, fontweight="bold", y=0.97)
    ax.set_title("Standard 25×25 Environment ($p = 0.0005$) | 20M Environment Steps",
                 fontsize=12, color="#444444", pad=12)
    
    ax.set_xlabel("Environment Steps (Millions)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Area Coverage Rate (%)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.55)
    
    rllib_handles = []
    rllib_labels = []
    for name, info in rllib_models.items():
        k = info["cache_key"]
        if k in data:
            steps = np.array(data[k]["steps"])
            vals = np.array(data[k]["vals"])
            v_smooth = smooth_curve(vals, window=25)
            line, = ax.plot(steps, v_smooth, color=info["color"], linewidth=info["lw"], linestyle=info["ls"])
            rllib_handles.append(line)
            rllib_labels.append(name)
            
    epymarl_handles = []
    epymarl_labels = []
    for name, info in epymarl_models.items():
        k = info["cache_key"]
        if k in data:
            steps = np.array(data[k]["steps"])
            vals = np.array(data[k]["vals"])
            v_smooth = smooth_curve(vals, window=25)
            line, = ax.plot(steps, v_smooth, color=info["color"], linewidth=info["lw"], linestyle=info["ls"])
            epymarl_handles.append(line)
            epymarl_labels.append(name)

    dummy_header = Line2D([], [], color="none")
    combined_handles = [dummy_header] + rllib_handles + [dummy_header] + epymarl_handles
    combined_labels = ["--- RLlib Framework ---"] + rllib_labels + ["--- EPyMARL Suite ---"] + epymarl_labels
    
    leg = ax.legend(combined_handles, combined_labels,
                    loc="center left", bbox_to_anchor=(1.015, 0.5),
                    fontsize=10.2, framealpha=0.96, edgecolor="#cccccc")
    
    for text in leg.get_texts():
        if "---" in text.get_text():
            text.set_fontweight("bold")
            text.set_color("#2c3e50")
            text.set_fontsize(11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    os.makedirs(desk_dir, exist_ok=True)
    out_comb_desk = os.path.join(desk_dir, "coverage_learning_curves_combined.png")
    out_comb_desk_fresh = os.path.join(desk_dir, "coverage_learning_curves_combined_identical_colors.png")
    plt.savefig(out_comb_desk, dpi=300)
    plt.savefig(out_comb_desk_fresh, dpi=300)

    if os.path.isdir(art_dir):
        plt.savefig(os.path.join(art_dir, "coverage_learning_curves_combined.png"), dpi=300)
        plt.savefig(os.path.join(art_dir, "coverage_learning_curves_combined_identical_colors.png"), dpi=300)
    
    plt.close()
    print("Saved updated unified plots to:\n -", out_comb_desk_fresh)

    # =========================================================================
    # 2. SPLIT PLOT (2 PANELS: "RLlib Framework" & "EPyMARL Suite")
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.8), sharey=True)
    fig.suptitle("Multi-Agent Training Learning Curves: Mean Area Coverage Rate\nStandard 25×25 Environment ($p = 0.0005$) | 20M Environment Steps",
                 fontsize=14.5, fontweight="bold", y=0.98)
    
    # Panel 1: RLlib Framework
    ax1.set_title("RLlib Framework", fontsize=13.5, fontweight="bold", pad=12)
    ax1.set_xlabel("Environment Steps (Millions)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Area Coverage Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    for name, info in rllib_models.items():
        k = info["cache_key"]
        if k in data:
            steps = np.array(data[k]["steps"])
            vals = np.array(data[k]["vals"])
            v_smooth = smooth_curve(vals, window=25)
            ax1.plot(steps, v_smooth, label=name, color=info["color"], linewidth=info["lw"], linestyle=info["ls"])
            
    ax1.legend(loc="lower right", fontsize=10.5, framealpha=0.95, edgecolor="#cccccc")
    
    # Panel 2: EPyMARL Suite
    ax2.set_title("EPyMARL Suite", fontsize=13.5, fontweight="bold", pad=12)
    ax2.set_xlabel("Environment Steps (Millions)", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    for name, info in epymarl_models.items():
        k = info["cache_key"]
        if k in data:
            steps = np.array(data[k]["steps"])
            vals = np.array(data[k]["vals"])
            v_smooth = smooth_curve(vals, window=25)
            ax2.plot(steps, v_smooth, label=name, color=info["color"], linewidth=info["lw"], linestyle=info["ls"])
            
    ax2.legend(loc="upper right", fontsize=10.5, framealpha=0.95, edgecolor="#cccccc")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_split_art = os.path.join(art_dir, "coverage_learning_curves_split.png")
    out_split_desk = os.path.join(desk_dir, "coverage_learning_curves_split.png")
    out_split_art_fresh = os.path.join(art_dir, "coverage_learning_curves_split_identical_colors.png")
    out_split_desk_fresh = os.path.join(desk_dir, "coverage_learning_curves_split_identical_colors.png")
    
    plt.savefig(out_split_art, dpi=300)
    plt.savefig(out_split_desk, dpi=300)
    plt.savefig(out_split_art_fresh, dpi=300)
    plt.savefig(out_split_desk_fresh, dpi=300)
    plt.close()
    print("Saved updated split plots to:\n -", out_split_art_fresh, "\n -", out_split_desk_fresh)

if __name__ == "__main__":
    main()
