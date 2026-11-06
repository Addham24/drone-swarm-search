#!/usr/bin/env python3
"""
Master Visualization Suite for Final Synchronized Benchmark (2D-final & 3D-final)
===================================================================================
Generates publication-quality RQ figures from the freshly evaluated final datasets:
  - RQ1: Self-Healing Ablation (2D, 3D, and Unified)
  - RQ3: Map Size & Urban Warfare Obstacle Generalisation (2D, 3D, and Unified)
  - RQ4: Model Leaderboards (2D, 3D, and Unified)

Targets:
  - ~/Desktop/Results/2D-final/
  - ~/Desktop/Results/3D-final/
  - ~/Desktop/Results/ (Unified thesis figures)
"""

import os
import sys
import shutil

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Rebind data paths to 2D-final and 3D-final
import generate_rq_visualizations as grq
grq.BASE_2D = os.path.expanduser("~/Desktop/Results/2D-final/Coverage")
grq.BASE_3D = os.path.expanduser("~/Desktop/Results/3D-final/Coverage")
grq.OUT_2D = os.path.expanduser("~/Desktop/Results/2D-final")
grq.OUT_3D = os.path.expanduser("~/Desktop/Results/3D-final")
grq.OUT_GEN = os.path.expanduser("~/Desktop/Results")

import generate_rq1_visualizations as grq1
grq1.OUT_2D = os.path.expanduser("~/Desktop/Results/2D-final")
grq1.OUT_3D = os.path.expanduser("~/Desktop/Results/3D-final")
grq1.OUT_GEN = os.path.expanduser("~/Desktop/Results")

import generate_rq3_v2_coverage as grq3_v2
grq3_v2.OUT_2D = os.path.expanduser("~/Desktop/Results/2D-final")
grq3_v2.OUT_3D = os.path.expanduser("~/Desktop/Results/3D-final")
grq3_v2.OUT_GEN = os.path.expanduser("~/Desktop/Results")


def generate_all_final_rq_figures():
    print(f"\n{'='*80}")
    print("  GENERATING ALL RESEARCH QUESTION (RQ) FIGURES FROM FINAL RESULTS")
    print("  Source 2D: ~/Desktop/Results/2D-final/Coverage")
    print("  Source 3D: ~/Desktop/Results/3D-final/Coverage")
    print(f"{'='*80}\n")

    # 1. RQ1: Self-Healing Ablation
    print("  [▶] Generating RQ1 Self-Healing Ablation plots...")
    p_rq1_2d = grq1.generate_rq1_dashboard("2D")
    p_rq1_3d = grq1.generate_rq1_dashboard("3D")
    p_rq1_uni = grq1.generate_rq1_unified()
    print("      [✓] RQ1 2D, 3D, and Unified generated.")

    # 2. RQ3: Generalisation (Standard vs Urban Obstacles, 25 vs 50)
    print("  [▶] Generating RQ3 Generalisation heatmaps and dashboards...")
    p_rq3_2d = grq.generate_rq3_dashboard("2D")
    p_rq3_3d = grq.generate_rq3_dashboard("3D")
    p_rq3_uni = grq.generate_rq3_unified()
    p_rq3_v2_2d = grq3_v2.generate_rq3_v2_dashboard("2D")
    p_rq3_v2_3d = grq3_v2.generate_rq3_v2_dashboard("3D")
    p_rq3_v2_uni = grq3_v2.generate_rq3_v2_unified()
    print("      [✓] RQ3 2D, 3D, and Unified generated.")

    # 3. RQ4: Leaderboard
    print("  [▶] Generating RQ4 Model Leaderboard rankings...")
    p_rq4_2d = grq.generate_rq4_leaderboard("2D")
    p_rq4_3d = grq.generate_rq4_leaderboard("3D")
    p_rq4_uni = grq.generate_rq4_unified()
    print("      [✓] RQ4 2D, 3D, and Unified generated.")

    all_plots = [
        p_rq1_2d, p_rq1_3d, p_rq1_uni,
        p_rq3_2d, p_rq3_3d, p_rq3_uni,
        p_rq3_v2_2d, p_rq3_v2_3d, p_rq3_v2_uni,
        p_rq4_2d, p_rq4_3d, p_rq4_uni
    ]

    grq.copy_to_artifacts([p for p in all_plots if p and os.path.exists(p)])
    print(f"\n{'='*80}")
    print("  ALL RESEARCH QUESTION FIGURES SUCCESSFULLY GENERATED!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    generate_all_final_rq_figures()
