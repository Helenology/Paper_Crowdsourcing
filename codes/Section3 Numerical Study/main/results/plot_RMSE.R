library(stringr)
library(reshape2)
library(plyr)
library(ggplot2)

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

# MSE
par(las = 1)
data$os_mse = data$os_beta_mse + data$os_sigma_mse
data$inr_mse = data$inr_beta_mse + data$inr_sigma_mse
dat = data[, c("alpha", "subset_ratio", "os_mse", "inr_mse")]
dat = melt(dat, id.vars = c("alpha", "subset_ratio"),
           value.name = "MSE")
dat$variable = substr(dat$variable, 1, 3)
dat$variable = gsub("_", "", dat$variable)
alphas = unique(dat$alpha); alphas
subset_ratios = unique(dat$subset_ratio); subset_ratios
subset_ratios
for(alpha in alphas){
  for(subset_ratio in subset_ratios){
    aaa = dat[(dat$alpha == alpha) & (dat$subset_ratio == subset_ratio), ]
    boxplot(log(MSE) ~ variable, data = aaa,
            xlab = "",
            # names = c("INR", "OS"),
            main = paste0("alpha=", alpha, " subset=", subset_ratio))
  }
}

####################################
############### beta ###############
####################################
# dat = data[, c("alpha", "subset_ratio", "os_beta_mse", "inr_beta_mse")]
# dat = melt(dat, id.vars = c("alpha", "subset_ratio"),
#            value.name = "MSE")
# dat$variable = substr(dat$variable, 1, 3)
# dat$variable = gsub("_", "", dat$variable)

for(each_data in all_csv_data){
  alpha = each_data$alpha[1]
  subset_ratio = each_data$subset_ratio[1]
  dat = each_data[, c("alpha", "subset_ratio", "os_beta_mse", "inr_beta_mse")]
  dat = melt(dat, id.vars = c("alpha", "subset_ratio"),
             value.name = "MSE")
  dat$variable = substr(dat$variable, 1, 3)
  dat$variable = gsub("_", "", dat$variable)
  
  boxplot(log(MSE) ~ variable, data = dat,
          xlab ="",
          # names = c("INR", "OS"),
          main = paste0("beta MSE: ", "alpha=", alpha, " subset=", subset_ratio))
  
}

############################################
############### beta + sigma ###############
############################################

for(each_data in all_csv_data){
  each_data$os_mse = each_data$os_beta_mse + each_data$os_sigma_mse
  each_data$inr_mse = each_data$inr_beta_mse + each_data$inr_sigma_mse
  alpha = each_data$alpha[1]
  subset_ratio = each_data$subset_ratio[1]
  dat = each_data[, c("alpha", "subset_ratio", "os_mse", "inr_mse")]
  dat = melt(dat, id.vars = c("alpha", "subset_ratio"),
             value.name = "MSE")
  dat$variable = substr(dat$variable, 1, 3)
  dat$variable = gsub("_", "", dat$variable)
  
  boxplot(log(MSE) ~ variable, data = dat,
          xlab ="",
          # names = c("INR", "OS"),
          main = paste0("all MSE: ", "alpha=", alpha, " subset=", subset_ratio))
  
}


# without log
boxplot(MSE ~ alpha + subset_ratio + variable, data = dat,
        xlab = "")
# with log
boxplot(log(MSE) ~ alpha + subset_ratio + variable, data = dat,
        xlab = "")


ggplot(data = dat) +
  geom_boxplot(aes(x = factor(subset_ratio) + factor(alpha), y = log(MSE), fill = variable))

a = data[data$inr_beta_mse == max(data$inr_beta_mse), ]




# sigma
dat = data[, c("alpha", "subset_ratio", "os_sigma_mse", "inr_sigma_mse")]
dat = melt(dat, id.vars = c("alpha", "subset_ratio"),
           value.name = "MSE")


# without log
boxplot(MSE ~ alpha + subset_ratio + variable, data = dat,
        xlab = "")
# with log
boxplot(log(MSE) ~ alpha + subset_ratio + variable, data = dat,
        xlab = "")

b = data[data$inr_sigma_mse == max(data$inr_sigma_mse), ]
