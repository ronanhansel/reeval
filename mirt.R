library(mirt)

data <- read.csv("clean_data/base_matrix.csv", row.names=1)
print(data)
model <- mirt(data, 1, '3PL')
summary(model)
plot(model)