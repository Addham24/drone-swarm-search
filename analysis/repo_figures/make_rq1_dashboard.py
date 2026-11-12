"""Figure 3.2b: runs the repository's RQ1 plotting script (src/generate_rq1_visualizations.py,
generate_rq1_unified) on the final per-seed files.
Usage (from the repository root):
  python3 analysis/repo_figures/make_rq1_dashboard.py <results_root> <out_dir>"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _summaries
import generate_rq_visualizations as rq
import generate_rq1_visualizations as rq1

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tmp = tempfile.mkdtemp()
rq.BASE_2D, rq.BASE_3D = _summaries.build(root, tmp)
rq1.OUT_2D = rq1.OUT_3D = rq1.OUT_GEN = out
path = rq1.generate_rq1_unified()
os.replace(path, os.path.join(out, "fig_rq1_dashboard.png"))
print("wrote", os.path.join(out, "fig_rq1_dashboard.png"))
