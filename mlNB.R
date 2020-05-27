mlNB <- function(x, tol=1e-6) {
# Maximum likelihood estimation of negative binomial parameters.

   n <- length(x)
   m <- mean(x, na.rm=TRUE)
   # The function 'tabulate()' starts counting from 1.
   # Include 0s by adding 1 before tabulation.
   tab <- tabulate(x+1L)
   u <- 0:(length(tab)-1L)

   # 0 only slow down the computation.
   u <- u[tab > 0]
   tab <- tab[tab > 0]

   # Function to solve and its derivative.
   f <- function(a) {
      return(sum(tab*digamma(a+u)) - n*digamma(a) + n*log(a/(m+a)))
   }
   df <- function(a) {
      return(sum(tab*trigamma(a+u)) - n*trigamma(a) + n*m/(a*(m+a)))
   }
   
   # Step 1. Bisect to find lower and upper bound.
   a <- 1
   if (f(a) < 0) {
      a <- a / 2
      while(f(a) < 0) { a <- a / 2 }
      a. <- a
      .a <- a * 2
   }
   else {
      a <- a * 2
      while(f(a) >= 0) { a <- a * 2 }
      a. <- a / 2
      .a <- a
   }

   # In some cases, like when there are only 0s and 1s in the sample,
   # there is not enough information to estimate 'a'. If the estimate
   # becomes larger than a "reasonable" value, return NA.
   if (a. > 1024) { return(c(a=NA, p=NA)) }

   # Step 2. Hybrid of Newton-Raphson and bisection method.
   a.new <- a + .a + 2*tol
   while (abs(a.new - a) > tol) {
      a <- ifelse(a.new < a. || a.new > .a, (a. + .a) / 2, a.new)
      fa <- f(a)
      if (fa < 0) { .a <- a }
      else { a. <- a }
      a.new <- a - fa / df(a)
   }

   return(c(a=a, p=a/(m+a)))

}

mlNB2 <- function(x, y, tol=1e-6) {
# Maximum likelihood estimation of negative binomial parameters.

   k <- length(x)
   n <- length(y)
   mx <- mean(x, na.rm=TRUE)
   my <- mean(y, na.rm=TRUE)
   tab <- tabulate(c(x,y)+1L)
   u <- 0:(length(tab)-1L)

   # 0 only slow down the computation.
   u <- u[tab > 0]
   tab <- tab[tab > 0]

   # Function to solve and its derivative.
   f <- function(a) {
      return(sum(tab*digamma(a+u)) - (k+n)*digamma(a) -
         k*log(1+mx/a) - n*log(1+my/a))
   }
   df <- function(a) {
      return(sum(tab*trigamma(a+u)) - (k+n)*trigamma(a) +
          k*mx/(a*(mx+a)) + n*my/(a*(my+a)))
   }
   
   # Step 1. Bisect to find lower and upper bound.
   a <- 1
   if (f(a) < 0) {
      a <- a / 2
      while(f(a) < 0) { a <- a / 2 }
      a. <- a
      .a <- a * 2
   }
   else {
      a <- a * 2
      while(f(a) >= 0) { a <- a * 2 }
      a. <- a / 2
      .a <- a
   }

   # In some cases, like when there are only 0s and 1s in the sample,
   # there is not enough information to estimate 'a'. If the estimate
   # becomes larger than a "reasonable" value, return NA.
   if (a. > 1024) { return(c(a=NA, p=NA)) }

   # Step 2. Hybrid of Newton-Raphson and bisection method.
   a.new <- a + .a + 2*tol
   while (abs(a.new - a) > tol) {
      a <- ifelse(a.new < a. || a.new > .a, (a. + .a) / 2, a.new)
      fa <- f(a)
      if (fa < 0) { .a <- a }
      else { a. <- a }
      a.new <- a - fa / df(a)
   }

   return(c(a=a, p=a/(mx+a), q=a/(my+a)))

}
