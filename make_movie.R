require(lattice)
require(latticeExtra)
require(gridExtra)

plotit = function(mat, idx) {
   plot1 = cloud(
      mat,
      panel.3d.cloud = panel.3dbars,
      xlab = NULL,
      ylab = NULL,
      zlab = NULL,
      scales = list(arrows=FALSE, draw=FALSE),
      par.box = list(col=NA),
      col.facet = 'grey90',
      zlim = c(0,90)
   )
   fixit = function(z) ifelse(z == 0, 3, z)
   dat = data.frame(x = 4-ceiling(idx / 3), y = fixit(idx %% 3))
   plot2 = xyplot(
      y ~ x,
      data = dat,
      xlim = c(.5,3.5),
      ylim = c(.5,3.5),
      xlab = NULL,
      ylab = NULL,
      scales = list(at=-1),
      abline=list(h=c(1.5,2.5,3.5), v=c(1.5,2.5,3.5)),
      par.settings = simpleTheme(pch=19, col="black", cex=2)
   )
   grid.arrange(plot1, plot2, ncol=2)
}

mat = matrix(0, nrow=3, ncol=3)
colnames(mat) = rownames(mat) = 1:3

trans = list(
   c(2,4),
   c(1,3,5),
   c(2,6),
   c(1,5,7),
   c(2,4,6,8),
   c(3,5,9),
   c(4,8),
   c(5,7,9),
   c(6,8)
)

set.seed(123)
idx = sample(9, size=1)

for (iter in 1:512) {
   idx = sample(trans[[idx]], size=1)
   mat[idx] = mat[idx] + 1
   fname = paste("snapshot_", sprintf("%03d", iter), ".png", sep="")
   print(fname)

   png(height=256, width=512, fname)
   par(mfrow=c(1,2))
   plotit(mat, idx)
   dev.off()

}
