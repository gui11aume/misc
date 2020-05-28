library(rgl)
set.seed(123)
X = matrix(rnorm(600), ncol=3)
X[1:100,] = X[1:100,] + 3
X = -X
cols <- rep(c("grey40", "red"), each=100)
plot3d(X, col=cols, box=NA, size=6.5)
# This is ugly. NA makes the drawing of the box fail.
# We do not want the box to mess up the animation.
# A that point you can play with the window, rotate
# the cloud with the mouse etc.
dir.create("animation")
for (i in 1:90) {
  view3d(userMatrix=rotationMatrix(2*pi * i/90, 1, -1, -1))
  rgl.snapshot(filename=paste("animation/frame-",
                                  sprintf("%03d", i), ".png", sep=""))
}
