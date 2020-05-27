# Params.
T = 1000
tau = 1000
theta0 = 0.4
theta1 = 0.2
phi = (log(1-theta1)-log(1-theta0)) / (log(theta0)-log(theta1))

# Random series.
#set.seed(123)
R = 0 + c(runif(tau) < theta0, runif(T-tau) < theta1)
R_ = rev(R)

Rbar = cumsum(R) / (1:T)
Rbar_ = cumsum(R_) / (1:T)

xlogx = function(x) {
   x[x == 0] = 1
   return(x*log(x))
}

# Likelihood
#Lt = diffinv(R*log(theta0)+(1-R)*log(1-theta0)) +
#      rev(diffinv(R_*log(theta1)+(1-R_)*log(1-theta1)))
#plot(Lt, type='l')
#Xt = diffinv(R*log(theta0/theta1) + (1-R)*log((1-theta0)/(1-theta1)))
Xt = (1:T) * (xlogx(Rbar) + xlogx(1-Rbar)) +
      rev((1:T) * (xlogx(Rbar_) + xlogx(1-Rbar_)))
plot(Xt-Xt[T], type='l')
#t = which.max(Xt)
#theta0_hat = Rbar[t]
#theta1_hat = Rbar_[T-t]
#tmp = sort(c(Rbar[t], Rbar_[T-t]))
#theta0_hat = tmp[2]
#theta1_hat = tmp[1]
#phi_hat = log((1-theta0_hat)/(1-theta1_hat)) / log((theta1_hat)/(theta0_hat))
#K = 1000
#p00 = exp(-sum(pbinom(q=floor((1:K)*phi_hat/(1+phi_hat)), size=1:K, prob=theta0_hat, lower.tail=TRUE)/(1:K)))
#print(phi_hat)
#print(p00)
#Q = matrix(rep(0, K^2), ncol=K)
#Q[1,2:K] = p00*(1-theta0_hat)^(1:(K-1))
#Q[2,1] = p00*theta0_hat
#for (m in 2:K) {
#   for (l in 2:K) {
#      if ((l-1) > phi_hat*(m-1) + 1) next
#      if ((l-1) < phi_hat*(m-1) - phi_hat) {
#         Q[l,m] = theta0_hat*Q[l-1,m] + (1-theta0_hat)*Q[l,m-1]
#      } else {
#         Q[l,m] = theta0_hat*Q[l-1,m]
#      }
#   }
#}
#print(sum(Q))
# 6.906944
