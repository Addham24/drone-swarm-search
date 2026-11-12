# Figures 3.2-3.5 (R 4.3.3, base graphics). Reads the CSVs written by analysis.R and episodes.R.
# Usage:  Rscript analysis/figures.R <out_dir>
args <- commandArgs(TRUE)
if (length(args) < 1) stop("usage: Rscript figures.R <out_dir>")
out <- args[1]
set.seed(7)
col <- c(RSPO="#D35400", MAPPO="#1E88E5", QMIX="#2E7D32", COMA="#D32F2F", MAA2C="#B78103", `I-DQN`="#8E24AA")
alg  <- function(m) sub(" (Vanilla|SelfHeal)$", "", m)
isSH <- function(m) grepl("SelfHeal$", m)
ink <- "#1f1f1f"; quiet <- "#6b6b6b"; grid <- "#e6e6e3"; grey <- "#c9c9c4"
maps <- c("25x25_Standard","25x25_Obstacles","50x50_Standard","50x50_Obstacles")
ps <- c(0,0.00025,0.0005,0.00075,0.001)

# ---------- Figure 3.2: RQ1 pooled self-healing minus vanilla at p = 0.0005 ----------
R <- read.csv(file.path(out,"rq1_p0005_pooled.csv"))
algs <- c("RSPO","MAPPO","QMIX","MAA2C","COMA","I-DQN")
png(file.path(out,"fig_rq1_deltas.png"), width=3200, height=1700, res=300, type="cairo")
par(mfrow=c(1,4), mar=c(3.4,1,2.4,0.8), oma=c(0,4.5,1.2,0), family="sans", col.axis=quiet, fg=quiet, las=1, mgp=c(2,0.5,0), tcl=-0.25)
for (d in c("2D","3D")) for (mt in c("coverage_pct","survival_rate")) {
  z <- R[R$dim==d & R$metric==mt,]; z <- z[match(algs,z$alg),]
  plot(NA, xlim=c(-30,30), ylim=c(0.5,6.5), yaxt="n", xaxt="n", xlab="", ylab="", bty="n")
  abline(v=seq(-30,30,10), col=grid, lwd=.8); abline(v=0, col=quiet, lwd=1)
  axis(1, at=seq(-30,30,10), lwd=0, cex.axis=.8)
  for (i in 1:6) { y <- 7-i; sig <- !is.na(z$p_holm[i]) && z$p_holm[i] < .05
    if (is.na(z$p_holm[i])) { text(0, y, "both arms 0%", cex=.7, col=quiet, pos=4); next }
    segments(z$lo[i], y, z$hi[i], y, col=col[[algs[i]]], lwd=1.8)
    points(z$delta[i], y, pch=ifelse(sig,19,21), bg="white", col=col[[algs[i]]], cex=1.3, lwd=1.6)
    text(z$delta[i], y+0.32, sprintf("%+.1f%s", z$delta[i], ifelse(sig,"*","")), cex=.72, col=ink) }
  if (d=="2D" && mt=="coverage_pct") axis(2, at=6:1, labels=algs, lwd=0, cex.axis=.95, col.axis=ink, outer=FALSE, line=-0.2, xpd=NA, pos=-33, hadj=1)
  mtext(sprintf("%s: %s", d, ifelse(mt=="coverage_pct","coverage","survival")), side=3, line=.6, adj=0, cex=.75, font=2, col=ink)
  mtext("Self-healing minus vanilla (pp)", side=1, line=2.1, cex=.55, col=quiet)
}
mtext("Filled = Holm-adjusted p < 0.05 (also marked *); open = not significant. Bars: 95% bootstrap CI. Pooled over 4 maps at p = 0.0005, 400 episodes per arm.", side=3, outer=TRUE, line=0, cex=.6, col=quiet, adj=0)
dev.off()

# ---------- Figure 3.3: RQ2 SMEI degradation ----------
S <- read.csv(file.path(out, "scenario_summary.csv"))
mlab <- c("25x25 standard","25x25 obstacles","50x50 standard","50x50 obstacles")
hl <- c("RSPO SelfHeal","RSPO Vanilla","MAPPO SelfHeal","MAPPO Vanilla","QMIX SelfHeal")
allm <- unique(S$Model)
png(file.path(out, "fig_rq2_smei_degradation.png"), width = 3200, height = 1900, res = 300, type = "cairo")
par(mfrow = c(2,4), mar = c(3.2,3.2,2.2,0.6), oma = c(0,1.2,3.2,0), mgp = c(2,0.55,0), tcl = -0.25,
    family = "sans", col.axis = quiet, col.lab = ink, fg = quiet, las = 1, cex.axis = 0.8)
for (d in c("2D","3D")) for (j in seq_along(maps)) {
  plot(NA, xlim = c(0, 0.001), ylim = c(0, 90), xaxt = "n", yaxt = "n", xlab = "", ylab = "", bty = "n")
  abline(h = seq(0, 90, 15), col = grid, lwd = 0.8)
  axis(1, at = ps, labels = c("0",".00025",".0005",".00075",".001"), lwd = 0, lwd.ticks = 0.6, cex.axis = 0.7)
  axis(2, at = seq(0, 90, 30), lwd = 0, cex.axis = 0.75)
  mtext(sprintf("%s: %s", d, mlab[j]), side = 3, line = 0.5, cex = 0.72, col = ink, font = 2, adj = 0)
  for (m in setdiff(allm, hl)) { z <- S[S$dim==d & S$map==maps[j] & S$Model==m,]; z <- z[order(z$p),]
    lines(z$p, z$smei, col = grey, lwd = 0.9, lty = ifelse(isSH(m),1,2)) }
  for (m in rev(hl)) { z <- S[S$dim==d & S$map==maps[j] & S$Model==m,]; z <- z[order(z$p),]
    lines(z$p, z$smei, col = col[[alg(m)]], lwd = 2, lty = ifelse(isSH(m),1,2))
    points(z$p, z$smei, pch = ifelse(isSH(m),19,21), bg = "white", col = col[[alg(m)]], cex = 0.75, lwd = 1.3) }
  if (j == 1) mtext("Mean SMEI (%)", side = 2, line = 2.1, cex = 0.65, col = ink, las = 0)
  if (d == "3D") mtext("Fault probability p (per drone, per step)", side = 1, line = 2.0, cex = 0.55, col = quiet)
}
par(fig = c(0,1,0,1), oma = c(0,0,0,0), mar = c(0,0,0,0), new = TRUE); plot.new()
legend("top", inset = 0.005, ncol = 6, bty = "n", cex = 0.72, text.col = ink, text.width = c(0.11,0.09,0.12,0.1,0.11,0.1),
       legend = c("RSPO self-healing","RSPO vanilla","MAPPO self-healing","MAPPO vanilla","QMIX self-healing","Other 7 models"),
       col = c(col["RSPO"],col["RSPO"],col["MAPPO"],col["MAPPO"],col["QMIX"],grey),
       lty = c(1,2,1,2,1,1), lwd = c(2,2,2,2,2,1), pch = c(19,21,19,21,19,NA), pt.bg = "white", seg.len = 2.2)
dev.off()

# ---------- Figure 3.4: coverage heat map, model x map (mean over fault rates) ----------
C <- read.csv(file.path(out,"rq3_coverage_by_map.csv"))
mlab2 <- c("25 std","25 obs","50 std","50 obs")
L <- read.csv(file.path(out,"rq4_leaderboard.csv"))
pal <- colorRampPalette(c("#f4f1ea","#f2c6a0","#d9773a","#8a3a0f"))(101)
png(file.path(out,"fig_rq3_coverage_heatmap.png"), width=3200, height=1800, res=300, type="cairo")
par(mfrow=c(1,2), mar=c(3,8.5,2.4,1), oma=c(0,0,0,0), family="sans", fg=quiet, las=1)
for (d in c("2D","3D")) {
  mods <- L$Model[L$dim==d][order(L$rank[L$dim==d])]
  plot(NA, xlim=c(.5,4.5), ylim=c(.5,12.5), xaxt="n", yaxt="n", xlab="", ylab="", bty="n")
  for (i in seq_along(mods)) for (j in 1:4) { v <- C$coverage[C$dim==d & C$Model==mods[i] & C$map==maps[j]]; y <- 13-i
    rect(j-.48, y-.46, j+.48, y+.46, col=pal[round(v)+1], border=NA)
    text(j, y, sprintf("%.1f", v), cex=.72, col=ifelse(v>55,"white",ink)) }
  axis(2, at=12:1, labels=sub("SelfHeal","SH",sub("Vanilla","V",mods)), lwd=0, cex.axis=.8, col.axis=ink)
  axis(3, at=1:4, labels=mlab2, lwd=0, cex.axis=.8, col.axis=ink, line=-.6)
  mtext(sprintf("%s: mean coverage (%%) by map, rows ranked by overall SMEI", d), side=1, line=.8, adj=0, cex=.72, font=2, col=ink)
}
dev.off()

# ---------- Figure 3.5: RQ4 leaderboard + coverage vs survival ----------
x <- read.csv(file.path(out, "episodes.csv"))
png(file.path(out, "fig_rq4_leaderboard.png"), width = 3200, height = 2600, res = 300, type = "cairo")
layout(matrix(1:4, 2, byrow = TRUE), widths = c(1.15, 1))
par(oma = c(0,0,1.5,0), family = "sans", col.axis = quiet, fg = quiet, las = 1, mgp = c(2,0.5,0), tcl = -0.25)
for (d in c("2D","3D")) {
  xd <- x[x$dim == d,]
  mm <- tapply(xd$smei, xd$Model, mean)
  ci <- t(sapply(names(mm), function(m) { v <- xd$smei[xd$Model==m]
        bm <- colMeans(matrix(sample(v, length(v)*2000, TRUE), nrow = length(v))); quantile(bm, c(.025,.975)) }))
  o <- order(mm); n <- length(mm)
  par(mar = c(3,8.2,2.2,1))
  plot(NA, xlim = c(0, 70), ylim = c(0.5, n+0.5), yaxt = "n", xaxt = "n", xlab = "", ylab = "", bty = "n")
  abline(v = seq(0,70,10), col = grid, lwd = 0.8); axis(1, at = seq(0,70,10), lwd = 0, cex.axis = 0.75)
  for (i in seq_len(n)) { m <- names(mm)[o[i]]; cc <- col[[alg(m)]]
    segments(ci[m,1], i, ci[m,2], i, col = cc, lwd = 1.6)
    points(mm[m], i, pch = ifelse(isSH(m),19,21), bg = "white", col = cc, cex = 1.15, lwd = 1.6)
    text(ci[m,2] + 1.2, i, sprintf("%.1f", mm[m]), adj = 0, cex = 0.68, col = ink) }
  axis(2, at = seq_len(n), labels = sub("SelfHeal","self-healing", sub("Vanilla","vanilla", names(mm)[o])), lwd = 0, cex.axis = 0.72, col.axis = ink)
  mtext(sprintf("%s: mean SMEI over 20 scenarios (95%% CI)", d), side = 3, line = 0.6, adj = 0, cex = 0.75, font = 2, col = ink)
  if (d == "3D") mtext("Mean SMEI (%)", side = 1, line = 1.9, cex = 0.6, col = quiet)
  cv <- tapply(xd$coverage_pct, xd$Model, mean); sv <- tapply(xd$survival_rate, xd$Model, mean)
  par(mar = c(3,3.2,2.2,1))
  plot(NA, xlim = c(0,100), ylim = c(0,60), xaxt="n", yaxt="n", xlab="", ylab="", bty="n")
  abline(v = seq(0,100,20), h = seq(0,60,20), col = grid, lwd = 0.8)
  axis(1, at = seq(0,100,20), lwd = 0, cex.axis = 0.75); axis(2, at = seq(0,60,20), lwd = 0, cex.axis = 0.75)
  for (m in names(cv)) points(cv[m], sv[m], pch = ifelse(isSH(m),19,21), bg = "white", col = col[[alg(m)]], cex = 1.2, lwd = 1.6)
  lab <- names(cv)[cv > 60 | sv > 5]
  pos <- c("RSPO Vanilla"=3, "RSPO SelfHeal"=1, "MAPPO SelfHeal"=2, "MAPPO Vanilla"=1, "QMIX Vanilla"=3, "QMIX SelfHeal"=4, "MAA2C SelfHeal"=4, "COMA SelfHeal"=4, "COMA Vanilla"=4)
  if (d == "2D") { pos[["MAPPO Vanilla"]] <- 3; pos[["MAPPO SelfHeal"]] <- 1; pos[["RSPO SelfHeal"]] <- 4 }
  for (m in lab) text(cv[m], sv[m], sub(" SelfHeal"," SH", sub(" Vanilla"," V", m)), pos = if (m %in% names(pos)) pos[[m]] else 3, offset = 0.6, cex = 0.6, col = ink)
  mtext(sprintf("%s: mean coverage vs mean survival", d), side = 3, line = 0.6, adj = 0, cex = 0.75, font = 2, col = ink)
  mtext("Survival (%)", side = 2, line = 2.0, cex = 0.6, col = quiet, las = 0)
  if (d == "3D") mtext("Coverage (%)", side = 1, line = 1.9, cex = 0.6, col = quiet)
}
par(fig = c(0,1,0,1), oma = c(0,0,0,0), mar = c(0,0,0,0), new = TRUE); plot.new()
legend("top", horiz = TRUE, bty = "n", cex = 0.72, text.col = ink, legend = c(names(col), "filled = self-healing", "open = vanilla"),
       col = c(col, ink, ink), pch = c(rep(19,6), 19, 21), pt.bg = "white")
dev.off()
cat("figures done\n")
