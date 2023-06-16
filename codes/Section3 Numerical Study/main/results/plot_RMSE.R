library(stringr)

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
data$X = str_extract(data$X, pattern = "N\\d+")


boxplot(RMSE ~ X, data = data,
        xlab = "",
        ylab = "RMSE"#,
        # names=c(0, 1)
        )
