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
  csv_data[, 1] = csv_files[i]
  data = rbind(data, csv_data)
}

# 正则匹配
data$alpha = str_extract(data$X, pattern = "alpha.{3}")
data$r = str_extract(data$X, pattern = "r(\\d|\\.)+")


# alpha + r
boxplot(RMSE ~ alpha + r, data = data,
        xlab = "")
        # ylab = "log(RMSE)"#,
        # names=c(0, 1)

mean_RMSE = ddply(data, .(alpha, r), function(x){
  mean(x$RMSE)
}); mean_RMSE

median_RMSE = ddply(data, .(alpha, r), function(x){
  median(x$RMSE)
}); median_RMSE

write.csv(median_RMSE, "/Users/helenology/Desktop/median_RMSE.csv")
