# Supplementary numbers quoted in Chapter 3: fault-only survival ceiling and gaps (Table 3.4),
# head-to-head scenario wins, SMEI weight-sensitivity summary, per-scenario top three (dashboard
# captions, §3.7-3.8), obstacle cost, SMEI-of-means check and 3D charging steps (§3.6).
# Usage:  Rscript analysis/supplementary.R <out_dir>     (after analysis.R and episodes.R)
args <- commandArgs(TRUE)
if (length(args) < 1) stop("usage: Rscript supplementary.R <out_dir>")
out <- args[1]
x <- read.csv(file.path(out, "episodes.csv")); S <- read.csv(file.path(out, "scenario_summary.csv"))
maps <- c("25x25_Standard","25x25_Obstacles","50x50_Standard","50x50_Obstacles"); ps <- c(0,0.00025,0.0005,0.00075,0.001)
r0 <- function(v) sprintf("%d", floor(v + 0.5 + 1e-9))          # half-up rounding, as in the thesis
ab <- function(m) sub(" SelfHeal"," SH", sub(" Vanilla"," V", m))

# fault-only survival ceiling: 100 (1 - p)^T, T = 750 (25x25) or 1,500 (50x50)
ceil <- function(p) mean(100*(1-p)^c(750,750,1500,1500))
cat("\n== Fault-only ceiling and survival gaps (Table 3.4) ==\n")
for (d in c("2D","3D")) { z <- S[S$dim==d,]; t <- tapply(z$survival, list(z$p, z$Model), mean)
  g <- sapply(rownames(t), function(p) ceil(as.numeric(p))) - t
  cat(d, "ceiling:", round(sapply(as.numeric(rownames(t)), ceil), 1), "\n")
  print(round(g, 1)) }

cat("\n== RSPO-MAPPO head-to-head on SMEI (Holm across the 40 scenarios per pair; deterministic ties p = 1) ==\n")
x$type <- ifelse(grepl("Obst", x$map), "Obs", "Std")
pairs <- list(c("RSPO SelfHeal","MAPPO SelfHeal"), c("RSPO Vanilla","MAPPO Vanilla"), c("MAPPO SelfHeal","MAPPO Vanilla"), c("RSPO SelfHeal","RSPO Vanilla"))
for (pr in pairs) { r <- do.call(rbind, lapply(split(x, paste(x$dim,x$map,x$p)), function(s) { a <- s$smei[s$Model==pr[1]]; b <- s$smei[s$Model==pr[2]]
  data.frame(dim=s$dim[1], type=s$type[1], diff=mean(a)-mean(b), pv=if (sd(a)==0 && sd(b)==0) 1 else suppressWarnings(wilcox.test(a,b)$p.value)) }))
  r$ph <- p.adjust(r$pv, "holm")
  cat("\n", pr[1], "vs", pr[2], ": wins", sum(r$diff>0), "sig", sum(r$diff>0 & r$ph<.05), "| losses", sum(r$diff<0), "sig", sum(r$diff<0 & r$ph<.05), "\n")
  r$win <- r$diff > 0; r$sigw <- r$diff > 0 & r$ph < .05; r$sigl <- r$diff < 0 & r$ph < .05
  print(merge(aggregate(cbind(win, sigw, sigl) ~ dim + type, r, sum), aggregate(cbind(mean_diff = diff) ~ dim + type, r, function(v) round(mean(v), 1)))) }

cat("\n== SMEI weight sensitivity ==\n")
K <- read.csv(file.path(out, "rq4_smei_sensitivity.csv"))
for (d in c("2D","3D")) { k <- K[K$dim==d,]; cat(d, "\n"); print(table(k$top)); print(round(quantile(k$kendall, c(0,.5)), 2)) }

cat("\n== Top three by SMEI per scenario (dashboard captions) ==\n")
for (d in c("2D","3D")) for (m in maps) for (p in ps) { z <- S[S$dim==d & S$map==m & S$p==p,]; z <- z[order(-z$smei),][1:3,]
  cat(sprintf("%s %s p=%s: %s | ceiling %s%%\n", d, m, p,
      paste(sprintf("%s %.1f / %s / %.1f", ab(z$Model), z$coverage, r0(z$survival), z$smei), collapse="; "),
      r0(100*(1-p)^ifelse(grepl("^25", m), 750, 1500)))) }

cat("\n== Obstacle cost (coverage and survival, standard vs obstacles) ==\n")
for (d in c("2D","3D")) for (m in c("RSPO Vanilla","RSPO SelfHeal","MAPPO Vanilla","MAPPO SelfHeal")) { z <- S[S$dim==d & S$Model==m,]
  a <- round(tapply(z$coverage, z$map, mean), 1); b <- round(tapply(z$survival, z$map, mean), 1)
  cat(d, m, "coverage:", paste(names(a), a, collapse="  "), "| survival:", paste(names(b), b, collapse="  "), "\n") }

cat("\n== Per-episode SMEI vs SMEI of scenario means: same ranking? ==\n")
for (d in c("2D","3D")) { z <- S[S$dim==d,]
  mm <- 100*pmax(z$coverage/100,.001)^.4*pmax(z$survival/100,.001)^.3*pmax(z$battery/100,.001)^.3
  a <- tapply(mm, z$Model, mean); b <- tapply(z$smei, z$Model, mean)
  cat(d, "rank identical:", identical(names(sort(a)), names(sort(b))), "\n") }

cat("\n== 3D charging steps per episode, RSPO at p = 0.0005 (3D runner only) ==\n")
c3 <- aggregate(charging_steps ~ Model + map + p, x[x$dim=="3D",], mean)
print(range(c3$charging_steps[grepl("RSPO", c3$Model) & c3$p==5e-4]))
cat("\nDONE\n")
