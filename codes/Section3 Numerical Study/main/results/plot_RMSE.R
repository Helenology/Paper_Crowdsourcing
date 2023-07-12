library(stringr)
library(plyr)

# 设置工作目录
setwd("/Users/helenology/Desktop/光华/\ 论文/4-Crowdsourcing/codes/Section3 Numerical Study/main/results/")

# 获取所有csv文件路径
csv_files <- list.files(pattern = "*.csv")

# 读取所有csv文件
all_csv_data <- lapply(csv_files, read.csv)

data = data.frame()
for(i in 1:length(all_csv_data)){
  csv_data = all_csv_data[[i]]
  # csv_data$file_name = csv_files[i]
  data = rbind(data, csv_data)
}

# 正则匹配
# data$alpha = str_extract(data$X, pattern = "alpha.{3}")
# data$r = str_extract(data$X, pattern = "r(\\d|\\.)+")

data$os_mse = data$os_beta_mse + data$os_sigma_mse
data$inr_mse = data$inr_beta_mse + data$inr_sigma_mse

library(reshape2)
dat = data[, c("alpha", "subset_ratio", "os_mse", "inr_mse")]
dat = melt(dat, id.vars = c("alpha", "subset_ratio"),
           value.name = "MSE")

# alpha + subset_ratio

# without log
boxplot(MSE ~ alpha + subset_ratio + variable, data = dat,
        xlab = "")
# with log
boxplot(log(MSE) ~ alpha + subset_ratio + variable, data = dat,
        xlab = "")
# names = 1:6)
        # ylab = "log(RMSE)"#,
        # names=c(0, 1)

mean_RMSE = ddply(data, .(alpha, subset_ratio), function(x){
  mean(x$os_beta_mse)
}); mean_RMSE

median_RMSE = ddply(data, .(alpha, subset_ratio), function(x){
  median(x$os_beta_mse)
}); median_RMSE

library(ggplot2)

#ggplot(dat) +
#  geom_boxplot(aes(y = MSE, x = factor(alpha), fill=factor(subset_ratio))) 
#  # geom_boxplot(aes(y = MSE, fill=factor(variable)))


# write.csv(median_RMSE, "/Users/helenology/Desktop/median_RMSE.csv")
