# Analysis scripts (RSPO honours thesis)

These R scripts recompute every statistic, table value and Chapter 3 figure in the thesis
from the per-seed evaluation files. They use base R only (tested with R 4.3.3) and need no packages.

## Input

A results folder written by the evaluation pipeline (`src/run_master_final_benchmarks.py`), containing

```
<results_root>/2D-final/Coverage/<map>/<fault>/coverage_2d_raw_seed_by_seed.csv
<results_root>/3D-final/Coverage/<map>/<fault>/coverage_3d_raw_seed_by_seed.csv
```

for the four maps (`25x25_Standard`, `25x25_Obstacles`, `50x50_Standard`, `50x50_Obstacles`) and five
fault folders (`No_Faults_0.0`, `Fault_0.00025`, `Active_Faults_0.0005`, `Fault_0.00075`, `Fault_0.001`).
Each file holds 100 seeds (1000-1099) for each of the 12 models. The folders may also be named
`2D-final-v3/` and `3D-final-v3/`; the scripts accept either.

The thesis results are the v3 run (open maps from v2, byte-identical; obstacle maps re-run at
commit 213be59 after the obstacle-wrapper fix). See Appendix E of the thesis.

## Run

From the repository root, in this order:

```bash
Rscript analysis/analysis.R      <results_root> analysis/out   # statistics (~1 min)
Rscript analysis/episodes.R      <results_root> analysis/out   # per-episode table
Rscript analysis/figures.R       analysis/out                  # Figures 3.2-3.5
Rscript analysis/supplementary.R analysis/out > analysis/out/supplementary.txt
```

| Script | Produces | Thesis |
|---|---|---|
| `analysis.R` | `scenario_summary.csv` (means, 95% bootstrap CIs, B = 10,000, seed 20261103); `rq1_*.csv` (Wilcoxon rank-sum, rank-biserial, Holm); `rq2_*.csv` (SMEI retention, OLS slopes with t-intervals, scenario leaders); `rq3_*.csv`; `rq4_leaderboard.csv`; `rq4_smei_sensitivity.csv` (120 alternative weightings) | §2.10-2.11, Chapter 3 tables |
| `episodes.R` | `episodes.csv`: all 48,000 episodes with per-episode SMEI | input to the two scripts below |
| `figures.R` | `fig_rq1_deltas.png`, `fig_rq2_smei_degradation.png`, `fig_rq3_coverage_heatmap.png`, `fig_rq4_leaderboard.png` | Figures 3.2-3.5 |
| `supplementary.R` | fault-only survival ceiling and gaps, RSPO-MAPPO head-to-head counts (Holm across the 40 scenarios per pair), weighting summary, top three per scenario, obstacle cost, 3D charging steps | Table 3.4, §3.5-3.8 |

The 40 scenario dashboards (Figures 3.6-3.13) are written by the evaluation pipeline itself
(`src/generate_all_4_dashboards.py`); their captions use the top-three and ceiling values printed by `supplementary.R`.

## Notes

- SMEI is computed per episode, SMEI_e = 100 x C^0.4 x S^0.3 x B^0.3 (each term floored at 0.001), and
  then averaged per scenario (§2.10). `analysis.R` checks the logged SMEI against this formula.
- Where both arms give the same value on every seed (greedy models at p = 0), the rank-sum test is
  undefined and the comparison is counted as not significant (p = 1).
- Charging steps are logged by the 3D runner only; the 2D columns are left empty.
