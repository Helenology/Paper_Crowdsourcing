
# names = 1:6)
# ylab = "log(RMSE)"#,
# names=c(0, 1)


# write.csv(median_RMSE, "/Users/helenology/Desktop/median_RMSE.csv")

mean_RMSE = ddply(data, .(alpha, subset_ratio), function(x){
  mean(x$os_beta_mse)
}); mean_RMSE

median_RMSE = ddply(data, .(alpha, subset_ratio), function(x){
  median(x$os_beta_mse)
}); median_RMSE

# library(ggplot2)

#ggplot(dat) +
#  geom_boxplot(aes(y = MSE, x = factor(alpha), fill=factor(subset_ratio))) 
#  # geom_boxplot(aes(y = MSE, fill=factor(variable)))
