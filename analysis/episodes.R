# Builds out/episodes.csv: one row per evaluation episode (48,000 rows) with per-episode SMEI.
# Usage:  Rscript analysis/episodes.R <results_root> <out_dir>
args <- commandArgs(TRUE)
if (length(args) < 2) stop("usage: Rscript episodes.R <results_root> <out_dir>")
root <- args[1]; out <- args[2]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
dimdir <- function(dim) { for (cand in c(paste0(dim, "-final"), paste0(dim, "-final-v3")))
  if (dir.exists(file.path(root, cand))) return(cand); stop("no ", dim, "-final folder under ", root) }
maps <- c("25x25_Standard","25x25_Obstacles","50x50_Standard","50x50_Obstacles")
faults <- c(No_Faults_0.0=0, Fault_0.00025=0.00025, Active_Faults_0.0005=0.0005, Fault_0.00075=0.00075, Fault_0.001=0.001)
r <- list(); for (dim in c("2D","3D")) for (m in maps) for (f in names(faults)) {
  d <- read.csv(file.path(root, dimdir(dim), "Coverage", m, f, sprintf("coverage_%s_raw_seed_by_seed.csv", tolower(dim))))
  if (!"charging_steps" %in% names(d)) d$charging_steps <- NA   # 3D runner only
  d$smei <- 100*pmax(d$coverage_pct/100,.001)^.4*pmax(d$survival_rate/100,.001)^.3*pmax(d$avg_battery/100,.001)^.3
  r[[length(r)+1]] <- data.frame(dim=dim, map=m, p=faults[[f]], d[,c("seed","Model","coverage_pct","survival_rate","avg_battery","smei","crash_events","charging_steps")]) }
x <- do.call(rbind, r); write.csv(x, file.path(out, "episodes.csv"), row.names=FALSE)
cat("episodes.csv:", nrow(x), "rows\n")
