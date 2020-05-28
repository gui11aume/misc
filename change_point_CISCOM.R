r <- function(x, n) {
   h <- log(n)^(3/2)/n
   t <- log((1-h)^2/h^2)
   return(x * exp(-x^2/2) / 2.506628274631000685702 * (t + (4-t)/x^2))
}

xlogx <- function(x) {
   x[x == 0] <- 1
   return(x*log(x))
}

do_it_with_negative_binomial <- function(x) {
   tab <- tabulate(x+1L)
   u <- 0:(length(tab)-1L)
   n <- length(x)
   ap <- mlNB(x)

   a <- ap[1]
   p <- ap[2]
   ln <- sum(tab*lgamma(a+u)) - n*lgamma(a) + a*n*log(p) + sum(x)*log(1-p)

   lambda <- rep(NA, n)
   for (k in 1:(n-1)) {
      apq <- mlNB2(x[1:k], x[(k+1):n])
      a <- apq[1]
      p <- apq[2]
      q <- apq[3]
      lk <- sum(tab*lgamma(a+u)) - n*lgamma(a) +
         a*k*log(p) + sum(x[1:k])*log(1-p) +
         a*(n-k)*log(q) + sum(x[(k+1):n])*log(1-q)
      lambda[k] <- 2*lk - 2*ln
   }

   return(r(max(sqrt(lambda), na.rm=TRUE), n))
}

do_it_with_poisson <- function(x) {

   n <- length(x)
   lambda_0 = mean(x)
   lambda_1 = cumsum(x[-n]) / 1:(n-1) 
   lambda_2 = rev(cumsum(rev(x[-1])) / 1:(n-1))

   l0 = -n*lambda_0 + sum(x)*log(lambda_0)
   l1 = -(1:(n-1)) * (lambda_1 - xlogx(lambda_1))
   l2 = -((n-1):1) * (lambda_2 - xlogx(lambda_2))

   return(sqrt(max(2*l1 +2*l2 - 2*l0)))

}

source("mlNB.R")

pvalues <- list()
for (term in dir("terms")) {
   x <- scan(paste("terms", term, sep="/"), what=0, quiet=TRUE)
   term <- sub(".txt", "", term)
   pval <- do_it_with_negative_binomial(x)
   if (is.na(pval)) {
      pval <- do_it_with_poisson(x)
   }
   cat(paste(term, pval, "\n"), file=stderr())
   pvalues[term] <- pval
}
