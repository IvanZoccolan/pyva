
#Calibrates a CIR process to the A62F projected tables

library(data.table)

lx = fread("IPS55F.csv", sep=",", header=FALSE, dec=".")
names(lx) <- c("t", "lx")

age <- 65

len <- length(lx$t)

s <- data.table(t = 0:(len-age-1), s = lx[t>=age, lx] / lx[t==65, lx])

s_spline <-  splinefun(x=s$t, y=s$s)
library(ggplot2)

p <- ggplot(s, aes(x=t, y=s)) + geom_point()
p <- p + ggtitle("Survival function")
p


# Gompertz model 

surv_gompertz <- function(t, theta, mu0) {
  
  exp(-(mu0/theta)*(exp(theta*t) -1))
  
}

# Fitting the Gompertz model 

rse_gompertz <- function(par) {
  sqrt(sum((surv_gompertz(s$t, theta=par[1], mu0=par[2]) - s$s)^2))
}

res <- optim(par=c(0.1, 0.1), fn=rse_gompertz)

fitted_gompertz <- function(t) surv_gompertz(t, theta=res$par[1], mu0=res$par[2])

p <- ggplot(s, aes(x=t, y=s)) + geom_point()
p <- p + ggplot2::geom_function(fun = fitted_gompertz, colour="red")
p

gompertz <- function(t, theta=res$par[1], mu0=res$par[2]) {
  mu0*exp(theta*t)
}


# CIR model 

A <- function(t, alpha, theta, sigma) {
  
  gamma <- sqrt(theta^2 + 2*sigma^2)
  n <- 2*gamma * exp(0.5 * (gamma - theta)*t)
  d <- (gamma - theta) * (exp(gamma*t) - 1) + 2*gamma
  res <- (n/d)^((2*alpha)/(sigma^2))
  res
}

B <- function(t, alpha, theta, sigma){
  gamma <- sqrt(theta^2 + 2*sigma^2)
  n <- 2*(exp(gamma*t) - 1)
  d <- (gamma - theta)*(exp(gamma*t) - 1) + 2*gamma
  res <- n/d
  res
}

surv_cir <- function(t, alpha, theta, sigma, mu0) {
  
  A(t, alpha, theta, sigma) * exp(-B(t, alpha, theta, sigma)*mu0)
  
}

# Fitting the CIR model 

rse_cir <- function(par) {
  
  if (par[1] < 0 | par[2] < 0 | par[3] < 0 | par[4] < 0){
    # model paramters must be positive
    res <- Inf
  } else if (2*par[1]-par[3]^2 <= 1E-4) {
    # this condition guarantees we never reach negative values
    res <- Inf
    
  } else if (par[3] < 1E-2) {
    res <- Inf
  } else {
    res <-  sqrt(sum((surv_cir(s$t, alpha=par[1], theta=par[2], sigma=par[3], mu0=par[4]) - s$s)^2))
  }
 res
}

library(DEoptim)

res_cir <- DEoptim(fn=rse_cir, lower=c(0, 0, 0, 0), upper = c(1, 1, 1, 1), control=list(itermax=3000))

best_par <- res_cir$optim$bestmem
fitted_cir <- function(t) surv_cir(t, alpha=best_par[1], theta=best_par[2], sigma=best_par[3], mu0=best_par[4])

# Graphically compare the fitting of the two models 
p <- ggplot(s, aes(x=t, y=s)) + geom_point()
p <- p + ggplot2::geom_function(fun = fitted_cir, colour="blue")
p <- p + geom_function(fun=fitted_gompertz, colour="red")
p


# Simulate the CIR Stochastic process
set.seed(1223)

a <- best_par[1]
k <- best_par[2]
sig <- best_par[3]
m0 <- best_par[4]

dt <- 0.1
n_paths <- 1000
ncols <- (len - age)/dt+1
mu <- matrix(0, nrow=n_paths, ncol=ncols)
mu[, 1] <- m0

W <- matrix(0, nrow=n_paths, ncol=ncols)
for (i in 1:n_paths) {
  W[i, ] <- rnorm(ncols)
}

# Simulate the CIR paths by means of an Eulero scheme
for (j in 2:ncols) {
  mu[ ,j] <- mu[ ,j-1] + (a + k*mu[ ,j-1])*dt + sig*sqrt(mu[ , j-1])*sqrt(dt)*W[, j]
}

esp <- rexp(n_paths)
death <- rep(0, n_paths)
for (i in 1:n_paths) {
  death[i] <- min(which(cumsum(mu[i, ]*dt) > esp[i])) / (1/dt)
}

death[death == Inf] <- len - age

mean(death)


# Plot some stochastic intensity sample paths along with the deterministic gompertz 
# intensity fitted to the same data. 

y_lim <- ceiling(max(mu[, ncols], na.rm=TRUE))

x_grid <- seq(0, len - age, dt)
gompertz_curve <- gompertz(x_grid)


plot(x=x_grid, y=mu[1, ], type="l", xlab="residual life time", col="yellow", ylab="mu", ylim=c(0, y_lim), main="Sample CIR paths with Gompertz (red)")

for (i in 2:n_paths) {
  lines(x=x_grid, y=mu[i, ], col="yellow")
}

lines(x=x_grid, y=gompertz_curve, lwd=4, col="red")
