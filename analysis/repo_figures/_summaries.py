"""Shared helper: rebuild the per-scenario summary files the repository's plotting scripts read,
from the per-seed evaluation files, with SMEI computed per episode and then averaged (thesis §2.10)."""
import os
import pandas as pd

MAPS = ["25x25_Standard", "25x25_Obstacles", "50x50_Standard", "50x50_Obstacles"]
FAULTS = ["No_Faults_0.0", "Fault_0.00025", "Active_Faults_0.0005", "Fault_0.00075", "Fault_0.001"]


def dim_dir(root, dim):
    for cand in (f"{dim}-final", f"{dim}-final-v3"):
        if os.path.isdir(os.path.join(root, cand)):
            return os.path.join(root, cand)
    raise FileNotFoundError(f"no {dim}-final folder under {root}")


def build(root, tmp):
    """Write <tmp>/<dim>/Coverage/<map>/<fault>/coverage_<dim>_summary.csv for 2D and 3D."""
    for dim in ("2D", "3D"):
        for m in MAPS:
            for f in FAULTS:
                src = os.path.join(dim_dir(root, dim), "Coverage", m, f,
                                   f"coverage_{dim.lower()}_raw_seed_by_seed.csv")
                d = pd.read_csv(src)
                c = (d["coverage_pct"] / 100).clip(lower=0.001)
                s = (d["survival_rate"] / 100).clip(lower=0.001)
                b = (d["avg_battery"] / 100).clip(lower=0.001)
                d["smei"] = 100 * c ** 0.4 * s ** 0.3 * b ** 0.3
                g = d.groupby("Model", sort=False).agg(coverage_rate=("coverage_pct", "mean"),
                                                        survival_rate=("survival_rate", "mean"),
                                                        smei=("smei", "mean")).reset_index()
                out = os.path.join(tmp, dim, "Coverage", m, f)
                os.makedirs(out, exist_ok=True)
                g.to_csv(os.path.join(out, f"coverage_{dim.lower()}_summary.csv"), index=False)
    return os.path.join(tmp, "2D", "Coverage"), os.path.join(tmp, "3D", "Coverage")
