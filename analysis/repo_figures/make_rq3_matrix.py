"""Figures C.1 and C.2: runs the repository's RQ3 plotting script (src/generate_rq_visualizations.py,
generate_rq3_dashboard) on the final per-seed files, so every cell is a scenario mean over 100 episodes
with SMEI computed per episode.
Usage (from the repository root):
  python3 analysis/repo_figures/make_rq3_matrix.py <results_root> <out_dir>"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _summaries
import generate_rq_visualizations as rq

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tmp = tempfile.mkdtemp()
rq.BASE_2D, rq.BASE_3D = _summaries.build(root, tmp)
rq.OUT_2D = rq.OUT_3D = rq.OUT_GEN = out
for dim, name in (("2D", "fig_C1_rq3_matrix_2d.png"), ("3D", "fig_C2_rq3_matrix_3d.png")):
    p = rq.generate_rq3_dashboard(dim)
    os.replace(p, os.path.join(out, name))
    print("wrote", os.path.join(out, name))
