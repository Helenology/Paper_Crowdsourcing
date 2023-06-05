data = read.csv("/Users/helenology/Desktop/光华/ 论文/4-Crowdsourcing/codes/Section2.4 Reinforced Labeling/two_rounds_props.csv")
data = data[, -1]

library(latex2exp)

plot(data$pi0, data$p2_large_case, type = "l", lty=1,
     xlab = TeX("$p_{i0}$"),
     ylab = "Confidence",
     ylim = c(0.5, 1),
     mgp=c(2, 0.5, 0),
     )
# plot(data$pi0, data$p2_equal_case, #type='o,
#      # mgp=c(2, 0.5, 0),
#      # cex=0.5,
     # xlab = TeX("$p_{i0}$"),
     # ylab = "Confidence",
#      # cex.lab = 1.3, 
#      # cex.axis = 1.3,
#      )
lines(data$pi0, data$p2_equal_case, lty = 4)
lines(data$pi0, data$pi0, lty=2)

legend("bottomright", legend = c("Second-Round 1", "Second-Round 2", "First-Round"),
       lty = c(1, 4, 2))

