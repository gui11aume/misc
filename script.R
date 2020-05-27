plus = function(x) { ifelse(x < 0, 0, x) }

COL = colorRampPalette(c("red", "black"))(8)
k = c(1,2,3,4,5,10,20,50)

pdf("Largest_fragment.pdf")
plot(c(0,1), c(0,1), type="n", xlab="", ylab="Cumulative probability")
rect(xleft=-1, xright=2, ybottom=-1, ytop=2, border=NA, col="grey96")
abline(h=seq(from=0, to=1, by=.2), lwd=2, col="white")
abline(v=seq(from=0, to=1, by=.2), lwd=2, col="white")
for (i in 1:8) {
   x = seq(from=0, to=1, by=.005)
   y = rep(0, length(x))
   for (j in 0:(k[i]+1)) {
      y = y + choose(k[i]+1,j)*(-1)^j*(plus(1-j*x))^k[i]
   }
   lines(x, y, type='l', lwd=2, col=COL[i])
   if (i > 5) {
      lines(x, exp(-exp(-(k[i]+1)*x+log(k[i]+1))), lwd=1, col=COL[i])
   }
}
legend(legend=c("1", "2", "3", "4", "5", "10", "20", "50"),
   x="bottomright", inset=0.01, bg="white", lwd=2, col=COL)
dev.off()
