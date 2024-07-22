# install.packages("mirt")
# install.packages("ggplot2")
# install.packages("dplyr")

library(mirt)
library(ggplot2)
library(dplyr)

input_files <- c("./clean_data/divided_base_matrix.csv", 
                 "./clean_data/divided_perturb1_matrix.csv", 
                 "./clean_data/divided_perturb2_matrix.csv")

output_files <- c("./theta/divided_base_theta_1PL.csv", 
                       "./theta/divided_perturb1_theta_1PL.csv", 
                       "./theta/divided_perturb2_theta_1PL.csv")

for (i in 1:length(input_files)) {
  data <- read.csv(input_files[i], row.names=1)
  
  items_to_remove <- which(apply(data, 2, function(x) length(unique(x)) == 1))
  if(length(items_to_remove) > 0) {
    data <- data[, -items_to_remove]
  }

    model <- mirt(data, 1, 'Rasch')
    theta <- fscores(model)
    write.csv(theta, output_files[i])
    fscores(model)
}
