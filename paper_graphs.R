
library(data.table)
library(ggplot2)

wd <- "D:\\Python\\pyva\\"
setwd(wd)

df <- fread("fees_paper.csv")

df.long <- melt(df, id.vars="fee", measure.vars=c("Static", "Mixed", "Dynamic"))
names(df.long) <- c("Fee", "Approach", "Value")

p <- ggplot(df.long, aes(x=Fee, y=Value, colour=Approach)) + geom_line(aes(linetype=Approach))
p <- p + theme_bw() + geom_hline(yintercept = 100, linetype = "twodash")
p <- p + scale_x_continuous(breaks = scales::extended_breaks(n=16))
p <- p + labs(y=quote(V[0]), x=quote(phi1))
p


ggsave("fee.pdf", p, width=8, height = 5)

######

df <- fread("fees_penalties.csv", header = TRUE)
names(df)[[1]] <- "Fee"

df.long <- melt(df, id.vars="Fee", measure.vars = c("0.0", "0.02", "0.04"))
names(df.long) <- c("Fee", "Penalty (k)", "Value")

p <- ggplot(df.long, aes(x=Fee, y=Value, colour=`Penalty (k)`)) + geom_line(aes(linetype=`Penalty (k)`))
p <- p  + theme_bw() + geom_hline(yintercept = 100, linetype = "twodash")
p <- p + scale_x_continuous(breaks = scales::extended_breaks(n=10))
p <- p + labs(y=quote(V[0]), x=quote(phi1))
p

ggsave("fee_penalty_2.pdf", p, width=8, height = 5)

######

df <- fread("fees_rollups.csv", header = TRUE)
names(df)[[1]] <- "Fee"

df.long <- melt(df, id.vars="Fee", measure.vars = c("0.04", "0.06", "0.08"))
names(df.long) <- c("Fee", "Rollup (b)", "Value")

p <- ggplot(df.long, aes(x=Fee, y=Value, colour=`Rollup (b)`)) + geom_line(aes(linetype=`Rollup (b)`))
p <- p  + theme_bw() + geom_hline(yintercept = 100, linetype = "twodash")
p <- p + scale_x_continuous(breaks = scales::extended_breaks(n=10))
p <- p + labs(y=quote(V[0]), x=quote(phi1))
p

ggsave("fee_rollup.pdf", p, width=8, height = 5)


######

df <- fread("fees_g_rates.csv", header = TRUE)
names(df)[[1]] <- "Fee"

df.long <- melt(df, id.vars="Fee", measure.vars = c("0.04", "0.05", "0.06"))
names(df.long) <- c("Fee", "Withdrawal (g)", "Value")

p <- ggplot(df.long, aes(x=Fee, y=Value, colour=`Withdrawal (g)`)) + geom_line(aes(linetype=`Withdrawal (g)`))
p <- p  + theme_bw() + geom_hline(yintercept = 100, linetype = "twodash")
p <- p + scale_x_continuous(breaks = scales::extended_breaks(n=10))
p <- p + labs(y=quote(V[0]), x=quote(phi1))
p

ggsave("fee_g_rates.pdf", p, width=8, height = 5)
