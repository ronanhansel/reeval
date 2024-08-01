# options(repos = c(CRAN = "https://cloud.r-project.org"))
# install.packages("tidyverse")
# install.packages("devtools")
# library(devtools)
# install_github("masurp/ggmirt")  

library(tidyverse)
library(mirt)
library(ggmirt)

data <- sim_irt(1000, 1000, discrimination = .25, seed = 42)

items_to_remove <- which(apply(data, 2, function(x) length(unique(x)) == 1))
if(length(items_to_remove) > 0) {
    data <- data[, -items_to_remove]
    }

model <- mirt(data, 1, '3PL')
coefficient <- coef(model)
write.csv(coefficient, "coef.csv")