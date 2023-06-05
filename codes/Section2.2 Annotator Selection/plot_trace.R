data = read.csv("/Users/helenology/Desktop/光华/ 论文/4-Crowdsourcing/codes/simulation/trace.csv")
data = data[, -1]

library(latex2exp)

plot(data$sigma, data$trace, type='o', 
     mgp=c(1.5, 0.5, 0),
     cex=0.5,
     xlab = TeX("$\\sigma_{02}$"),
     ylab = TeX("$tr(\\widetilde{\\Sigma}_\\beta^{-1})$"),
     cex.lab = 1.3, 
     cex.axis = 1.3)

