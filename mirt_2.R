# options(repos = c(CRAN = "https://cloud.r-project.org"))
# install.packages("tidyverse")
# install.packages("devtools")
# library(devtools)
# install_github("masurp/ggmirt")  

library(tidyverse)
library(mirt)
library(ggmirt)

d <- read.csv("./clean_data/divided_base_matrix.csv", row.names=1)
items_to_remove <- which(apply(d, 2, function(x) length(unique(x)) == 1))
d <- d[, -items_to_remove]

unimodel <- 'F1 = 1-35'

fit3PL <- mirt(data = d, 
               model = unimodel,  # alternatively, we could also just specify model = 1 in this case
               itemtype = "3PL", 
               verbose = FALSE)
fit3PL

# summary(fit3PL)

params3PL <- coef(fit3PL, IRTpars = TRUE, simplify = TRUE)
round(params3PL$items, 2)

tracePlot(fit3PL)