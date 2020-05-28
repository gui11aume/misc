tsne = function (X, initial_config = NULL,
    perplexity = 100, max_iter = 1000, min_cost = 0, epoch_callback = NULL, 
    whiten = TRUE, epoch = 100) {

    X = as.matrix(X)
    X = X - min(X)
    X = X/max(X)

    k = 2
    initial_dims = 3
    n = nrow(X)

    #momentum = 0.5
    momentum = 0.5
    final_momentum = 0.8
    mom_switch_iter = 250
    #epsilon = 500
    epsilon = 1
    min_gain = 0.01
    initial_P_gain = 4
    eps = 2^(-52)

    set.seed(123)
    ydata = matrix(rnorm(k * n), n) / 5
    P = tsne:::.x2p(X, perplexity, 1e-05)$P
    P = 0.5 * (P + t(P))
    P[P < eps] <- eps
    P = P/sum(P)
    P = P * initial_P_gain
    grads = matrix(0, nrow(ydata), ncol(ydata))
    incs = matrix(0, nrow(ydata), ncol(ydata))
    gains = matrix(1, nrow(ydata), ncol(ydata))

    R1 = colorRampPalette(c("white", "grey40"))(11)
    R2 = colorRampPalette(c("white", "red"))(11)

    for (iter in 1:260) {
        sum_ydata = apply(ydata^2, 1, sum)
        num = 1/(1 + sum_ydata + sweep(-2 * ydata %*% t(ydata), 
            2, -t(sum_ydata)))
        diag(num) = 0
        Q = num/sum(num)
        if (any(is.nan(num))) 
            message("NaN in grad. descent")
        Q[Q < eps] = eps
        stiffnesses = 4 * (P - Q) * num
        for (i in 1:n) {
            grads[i, ] = apply(sweep(-ydata, 2, -ydata[i, ]) * 
                stiffnesses[, i], 2, sum)
        }
        gains = ((gains + 0.2) * abs(sign(grads) != sign(incs)) + 
            gains * 0.8 * abs(sign(grads) == sign(incs)))
        gains[gains < min_gain] = min_gain
        incs = momentum * incs - epsilon * (gains * grads)
        ydata = ydata + incs
        ydata = sweep(ydata, 2, apply(ydata, 2, mean))
        if (iter == mom_switch_iter) 
            momentum = final_momentum
        if (iter == 100 && is.null(initial_config)) 
            P = P/4
        if (iter < 11) {
           cols=rep(c(R1[iter], R2[iter]), each=100)
        } else if (iter < 251) {
           cols=rep(c("grey40", "red"), each=100)
        } else {
           cols=rep(c(R1[261-iter], R2[261-iter]), each=100)
        }
        png(paste(c("shot_", sprintf("%03d", iter), ".png"), collapse=""),
            width=256, height=256)
        par(mar=c(0,0,0,0))
        plot(ydata, xlim=c(-4,4), ylim=c(-4,4), pch=19, cex=.9,
             bty="n", xaxt="n", yaxt="n", col=cols)
        dev.off()
    }
    ydata
}

