# install.packages("mirt")
# install.packages("ggplot2")
# install.packages("dplyr")

library(mirt)
library(ggplot2)
library(dplyr)

data <- read.csv("./clean_data/perturb2_matrix.csv", row.names=1)

items_to_remove <- which(apply(data, 2, function(x) length(unique(x)) == 1))
cleaned_data <- data[, -items_to_remove]

model <- mirt(cleaned_data, 1, '3PL')
coef(model)
coefficient <- coef(model)
write.csv(coefficient, "./model_coef/perturb2_coef.csv")

# summary(model)
# plot(model)

