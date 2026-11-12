# RSPO thesis: statistical analysis behind every number in Chapter 3 (§2.11)
# R 4.3.3, base R only.
#
# Usage:  Rscript analysis/analysis.R <results_root> <out_dir>
#   <results_root> holds 2D-final/ and 3D-final/ (or 2D-final-v3/ and 3D-final-v3/), each with
#   Coverage/<map>/<fault>/coverage_<2d|3d>_raw_seed_by_seed.csv written by the evaluation pipeline.
#
# Outputs (CSV, in <out_dir>):
#   scenario_summary.csv     per scenario x model means, 95% bootstrap CIs, deterministic flag
#   rq1_p0005_pooled.csv     RQ1 headline: self-healing minus vanilla at p = 0.0005, pooled over maps
#   rq1_by_scenario.csv      RQ1 per scenario with Holm-adjusted rank-sum p-values
#   rq1_wins.csv             RQ1 win / significant-win counts
#   rq2_degradation.csv      RQ2 SMEI retention (p = 0.001 vs 0) and OLS slopes with t-intervals
#   rq2_leaders.csv          SMEI leader per scenario
#   rq3_*.csv                RQ3 generalisation summaries
#   rq4_leaderboard.csv      RQ4 leaderboard (per-episode SMEI averaged)
#   rq4_smei_sensitivity.csv RQ4 ranking stability over 120 alternative SMEI weightings

set.seed(20261103)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript analysis.R <results_root> <out_dir>")
root <- args[1]; out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
B <- 10000

dimdir <- function(dim) {
  for (cand in c(paste0(dim, "-final"), paste0(dim, "-final-v3")))
    if (dir.exists(file.path(root, cand))) return(cand)
  stop("no ", dim, "-final folder under ", root)
}

maps   <- c("25x25_Standard", "25x25_Obstacles", "50x50_Standard", "50x50_Obstacles")
faults <- c(No_Faults_0.0 = 0, Fault_0.00025 = 0.00025, Active_Faults_0.0005 = 0.0005,
            Fault_0.00075 = 0.00075, Fault_0.001 = 0.001)
models <- c("RSPO Vanilla","RSPO SelfHeal","MAPPO Vanilla","MAPPO SelfHeal","QMIX Vanilla","QMIX SelfHeal",
            "MAA2C Vanilla","MAA2C SelfHeal","COMA Vanilla","COMA SelfHeal","I-DQN Vanilla","I-DQN SelfHeal")

# ---- load ----
rows <- list()
for (dim in c("2D","3D")) for (m in maps) for (f in names(faults)) {
  fn <- file.path(root, dimdir(dim), "Coverage", m, f,
                  sprintf("coverage_%s_raw_seed_by_seed.csv", tolower(dim)))
  d <- read.csv(fn, check.names = FALSE)
  if (!"charging_steps" %in% names(d)) d$charging_steps <- NA   # logged by the 3D runner only
  if (!"smei" %in% names(d)) d$smei <- NA
  d <- d[, c("seed","coverage_pct","survival_rate","avg_battery","smei","crash_events","charging_steps","Model")]
  d$dim <- dim; d$map <- m; d$p <- faults[[f]]
  rows[[length(rows)+1]] <- d
}
x <- do.call(rbind, rows)
x$Model <- factor(x$Model, levels = models)
x$alg  <- sub(" (Vanilla|SelfHeal)$", "", as.character(x$Model))
x$cond <- ifelse(grepl("SelfHeal$", x$Model), "SelfHeal", "Vanilla")
cat("rows:", nrow(x), " cells:", nrow(unique(x[,c("dim","map","p","Model")])), "\n")
stopifnot(all(table(x$dim, x$map, x$p, x$Model) == 100))

# per-episode SMEI (§2.10); recomputed from components and checked against the logged value
smei_fn <- function(C,S,Bt, w=c(.4,.3,.3)) 100 * pmax(C/100,.001)^w[1] * pmax(S/100,.001)^w[2] * pmax(Bt/100,.001)^w[3]
x$smei_chk <- smei_fn(x$coverage_pct, x$survival_rate, x$avg_battery)
cat("max |logged SMEI - recomputed|:", max(abs(x$smei - x$smei_chk), na.rm=TRUE), "\n")
x$smei[is.na(x$smei)] <- x$smei_chk[is.na(x$smei)]

boot_ci <- function(v) {
  if (sd(v) == 0) return(c(mean(v), mean(v)))
  mm <- colMeans(matrix(sample(v, length(v)*B, TRUE), nrow = length(v)))
  unname(quantile(mm, c(.025,.975)))
}

# ---- 1. scenario summaries with bootstrap CIs ----
key <- unique(x[, c("dim","map","p","Model")])
S <- do.call(rbind, lapply(seq_len(nrow(key)), function(i) {
  k <- key[i,]; d <- x[x$dim==k$dim & x$map==k$map & x$p==k$p & x$Model==k$Model, ]
  cc <- boot_ci(d$coverage_pct); sc <- boot_ci(d$survival_rate); mc <- boot_ci(d$smei)
  data.frame(k, coverage=mean(d$coverage_pct), cov_lo=cc[1], cov_hi=cc[2], cov_sd=sd(d$coverage_pct),
             survival=mean(d$survival_rate), surv_lo=sc[1], surv_hi=sc[2],
             battery=mean(d$avg_battery), smei=mean(d$smei), smei_lo=mc[1], smei_hi=mc[2],
             crashes=mean(d$crash_events), charging=mean(d$charging_steps),
             deterministic = sd(d$coverage_pct)==0 && sd(d$survival_rate)==0)
}))
write.csv(S, file.path(out, "scenario_summary.csv"), row.names = FALSE)
cat("deterministic cells:", sum(S$deterministic), "of", nrow(S), "\n")
print(table(S$deterministic, S$p))

# ---- 2. RQ1: SelfHeal - Vanilla ----
rb <- function(a, b) { w <- suppressWarnings(wilcox.test(a, b, exact = FALSE));
  list(p = w$p.value, r = 2*w$statistic/(length(a)*length(b)) - 1) }   # rank-biserial, a vs b
diff_ci <- function(a, b) { if (sd(a)==0 && sd(b)==0) return(c(mean(a)-mean(b), mean(a)-mean(b)))
  ma <- colMeans(matrix(sample(a, length(a)*B, TRUE), nrow=length(a)))
  mb <- colMeans(matrix(sample(b, length(b)*B, TRUE), nrow=length(b)))
  unname(quantile(ma-mb, c(.025,.975))) }
# (a) headline: p = 0.0005, pooled over the 4 maps (400 episodes per model)
rq1 <- list()
for (dim in c("2D","3D")) for (a in unique(x$alg)) for (met in c("coverage_pct","survival_rate","smei")) {
  sh <- x[x$dim==dim & x$alg==a & x$cond=="SelfHeal" & x$p==0.0005, met]
  va <- x[x$dim==dim & x$alg==a & x$cond=="Vanilla"  & x$p==0.0005, met]
  t <- rb(sh, va); ci <- diff_ci(sh, va)
  rq1[[length(rq1)+1]] <- data.frame(dim=dim, alg=a, metric=met, vanilla=mean(va), selfheal=mean(sh),
                                     delta=mean(sh)-mean(va), lo=ci[1], hi=ci[2], p=t$p, r_rb=unname(t$r))
}
rq1 <- do.call(rbind, rq1)
rq1$p_holm <- ave(rq1$p, rq1$metric, FUN = function(p) p.adjust(p, "holm"))
write.csv(rq1, file.path(out, "rq1_p0005_pooled.csv"), row.names = FALSE)
# (b) all 20 scenarios: per-scenario deltas and tests (Holm within alg x dim x metric; deterministic ties p = 1)
rq1s <- list()
for (dim in c("2D","3D")) for (a in unique(x$alg)) for (m in maps) for (pp in faults) for (met in c("coverage_pct","survival_rate","smei")) {
  sh <- x[x$dim==dim & x$alg==a & x$cond=="SelfHeal" & x$map==m & x$p==pp, met]
  va <- x[x$dim==dim & x$alg==a & x$cond=="Vanilla"  & x$map==m & x$p==pp, met]
  pv <- if (sd(sh)==0 && sd(va)==0) NA else rb(sh, va)$p
  rq1s[[length(rq1s)+1]] <- data.frame(dim=dim, alg=a, map=m, p=pp, metric=met, delta=mean(sh)-mean(va), pval=pv)
}
rq1s <- do.call(rbind, rq1s)
rq1s$p_holm <- ave(ifelse(is.na(rq1s$pval), 1, rq1s$pval), rq1s$dim, rq1s$alg, rq1s$metric, FUN=function(p) p.adjust(p,"holm"))
write.csv(rq1s, file.path(out, "rq1_by_scenario.csv"), row.names = FALSE)
wins <- aggregate(cbind(win = delta > 0, sigwin = delta > 0 & p_holm < .05, sigloss = delta < 0 & p_holm < .05, mean_delta = delta) ~ dim + alg + metric, rq1s, function(v) if (is.logical(v)) sum(v) else mean(v))
write.csv(wins, file.path(out, "rq1_wins.csv"), row.names = FALSE)

# ---- 3. RQ2: degradation ----
rq2 <- list()
for (dim in c("2D","3D")) for (mo in models) for (m in maps) {
  d <- x[x$dim==dim & x$Model==mo & x$map==m, ]
  s0 <- d$smei[d$p==0]; s1 <- d$smei[d$p==0.001]
  rr <- if (mean(s0) > 0) mean(s1)/mean(s0) else NA
  if (sd(s0)==0 && sd(s1)==0) rci <- c(rr, rr) else {     # bootstrap the ratio of means
    b0 <- colMeans(matrix(sample(s0, 100*B, TRUE), 100)); b1 <- colMeans(matrix(sample(s1, 100*B, TRUE), 100))
    rci <- unname(quantile(b1/b0, c(.025,.975))) }
  fit <- lm(smei ~ I(p*1000), data = d)                     # OLS slope per +0.001 in p, t-interval
  cf <- confint(fit)[2,]
  rq2[[length(rq2)+1]] <- data.frame(dim=dim, Model=mo, map=m, smei_p0=mean(s0), smei_p001=mean(s1),
     retention=rr, ret_lo=rci[1], ret_hi=rci[2], slope=coef(fit)[2], slope_lo=cf[1], slope_hi=cf[2],
     cov_p0=mean(d$coverage_pct[d$p==0]), cov_p001=mean(d$coverage_pct[d$p==0.001]),
     surv_p0=mean(d$survival_rate[d$p==0]), surv_p001=mean(d$survival_rate[d$p==0.001]))
}
rq2 <- do.call(rbind, rq2); write.csv(rq2, file.path(out, "rq2_degradation.csv"), row.names = FALSE)
lead <- do.call(rbind, lapply(split(S, list(S$dim, S$map, S$p), drop=TRUE), function(d) {
  o <- d[order(-d$smei),]; data.frame(dim=o$dim[1], map=o$map[1], p=o$p[1], first=as.character(o$Model[1]),
  first_smei=o$smei[1], second=as.character(o$Model[2]), second_smei=o$smei[2], gap=o$smei[1]-o$smei[2],
  ci_overlap = o$smei_lo[1] <= o$smei_hi[2]) }))
write.csv(lead, file.path(out, "rq2_leaders.csv"), row.names = FALSE)

# ---- 4. RQ3: generalisation ----
g <- aggregate(cbind(coverage, survival, smei) ~ dim + Model, S, mean)
gmin <- aggregate(coverage ~ dim + Model, S, min); names(gmin)[3] <- "worst_coverage"
g <- merge(g, gmin)
S$size <- substr(S$map,1,5); S$obs <- grepl("Obstacles", S$map)
sz <- aggregate(coverage ~ dim + Model + size, S, mean); sz <- reshape(sz, idvar=c("dim","Model"), timevar="size", direction="wide")
sz$size_cost <- sz$`coverage.25x25` - sz$`coverage.50x50`
fc <- aggregate(coverage ~ dim + Model + p, S, mean); fc <- fc[fc$p %in% c(0, .001),]
fc <- reshape(fc, idvar=c("dim","Model"), timevar="p", direction="wide"); fc$fault_cost <- fc$coverage.0 - fc$coverage.0.001
g <- merge(merge(g, sz[,c("dim","Model","size_cost")]), fc[,c("dim","Model","fault_cost")])
write.csv(g[order(g$dim, -g$smei),], file.path(out, "rq3_generalisation.csv"), row.names = FALSE)
cov_by_map <- aggregate(coverage ~ dim + Model + map, S, mean)
write.csv(cov_by_map, file.path(out, "rq3_coverage_by_map.csv"), row.names = FALSE)
lead_cov <- do.call(rbind, lapply(split(S, list(S$dim, S$map, S$p), drop=TRUE), function(d) {
  o <- d[order(-d$coverage),]; data.frame(dim=o$dim[1], map=o$map[1], p=o$p[1], first=as.character(o$Model[1]), cov=o$coverage[1], second=as.character(o$Model[2]), cov2=o$coverage[2]) }))
write.csv(lead_cov, file.path(out, "rq3_coverage_leaders.csv"), row.names = FALSE)

# ---- 5. RQ4: leaderboard + SMEI weight sensitivity ----
lb <- aggregate(cbind(smei, coverage, survival, battery, crashes) ~ dim + Model, S, mean)
lb <- lb[order(lb$dim, -lb$smei),]; lb$rank <- ave(-lb$smei, lb$dim, FUN = rank)
write.csv(lb, file.path(out, "rq4_leaderboard.csv"), row.names = FALSE)
grid <- expand.grid(wc = seq(.1,.8,.05), ws = seq(.1,.8,.05)); grid$wb <- 1 - grid$wc - grid$ws
grid <- grid[grid$wb >= .1 - 1e-9,]
sens <- list()
for (dim in c("2D","3D")) {
  xd <- x[x$dim==dim,]
  base <- tapply(xd$smei, xd$Model, mean)
  for (i in seq_len(nrow(grid))) {
    w <- unlist(grid[i,1:3]); s <- tapply(smei_fn(xd$coverage_pct, xd$survival_rate, xd$avg_battery, w), xd$Model, mean)
    kt <- cor(base, s, method = "kendall")
    sens[[length(sens)+1]] <- data.frame(dim=dim, wc=w[1], ws=w[2], wb=w[3], kendall=kt,
                                         top=names(which.max(s)), rspo_sh_rank=rank(-s)["RSPO SelfHeal"], rspo_v_rank=rank(-s)["RSPO Vanilla"])
  }
}
sens <- do.call(rbind, sens); write.csv(sens, file.path(out, "rq4_smei_sensitivity.csv"), row.names = FALSE)
print(aggregate(cbind(kendall, rspo_sh_rank, rspo_v_rank) ~ dim, sens, function(v) c(min=min(v), median=median(v), max=max(v))))
print(table(sens$dim, sens$top))
for (met in c("coverage","survival")) { cat("\nRank by", met, "\n"); print(lb[order(lb$dim, -lb[[met]]), c("dim","Model",met)]) }
cat("\nDONE\n")
