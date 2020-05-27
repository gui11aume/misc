barcirc <- function(height, ref, names, draw.circle=TRUE, ...) {
# Circular barplot.

   # A bit of maths.
   angle <- 2*pi / length(height)

   # Dispatch dot arguments.
   dotargs = list(...)
   valid_plot_args <- c("main")
   plotargs <- dotargs[names(dotargs) %in% valid_plot_args]
   dotargs <- dotargs[!names(dotargs) %in% valid_plot_args]
   plotargs[["type"]] <- "n"
   plotargs[["bty"]] <- "n"
   plotargs[["xaxt"]] <- "n"
   plotargs[["yaxt"]] <- "n"
   plotargs[["xlab"]] <- ""
   plotargs[["ylab"]] <- ""
   plotargs[["x"]] <- c(-1.2,1.2)
   plotargs[["y"]] <- c(-1.2,1.2)

   # Do the plot.
   do.call(what=plot, args=plotargs);

   # Manually recycle the parameters passed to 'polygon()'.
   # NB: 'polargs.i' is a list of lists of arguments.
   polargs <- rep(
      do.call(
         what = mapply,
         args = c(list(FUN=list, SIMPLIFY=FALSE), dotargs)
      ),
      length.out = length(height)
   )

   for (i in 1:length(height)) {
      nsteps <- angle / .01
      steps <- seq(from=(i-1)*angle, to=i*angle, length.out=nsteps)
      x <- c(0, height[i]*cos(steps))
      y <- c(0, height[i]*sin(steps))
      do.call(what=polygon, args=c(list(x=x, y=y), polargs[[i]]))
      if (!missing(ref)) {
         #polargs[[i]][['border']] <- NA
         #polargs[[i]][['col']] <- NA
         #polargs[[i]][['col']] <-
         #   rgb(t(col2rgb(polargs[[i]][['col']])), alpha=.9,
         #   maxColorValue = 255)
         x <- ref[i]*cos(steps)
         y <- ref[i]*sin(steps)
         polargs[[i]][['lwd']] <- 2
         polargs[[i]][['border']] <- 'black'
         do.call(what=polygon, args=c(list(x=x, y=y), polargs[[i]]))
      }
   }
   if (draw.circle) {
      polygon(x=cos((0:99) * 2*pi / 100), y=sin((0:99) * 2*pi / 100))
   }
   if (!missing(names)) {
      for (i in 1:length(names)) {
         text(1.2*cos((i-.5)*angle), 1.2*sin((i-.5)*angle),
            labels=names[i], srt=i*360/length(names))
      }
   }
}
