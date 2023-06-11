library(ggplot2)

data = read.csv("/Users/helenology/Desktop/test.csv")
data = data[, -1]
names(data) = c('seed', 'sigma_3', 'MSE', 'ACC')
data$sigma_3 = factor(data$sigma_3)


ggplot(data = data, aes(y = log(MSE), x = sigma_3)) +
  geom_boxplot()

ggplot(data = data, aes(y = ACC, x = sigma_3)) +
  geom_boxplot()
