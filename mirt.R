# install.packages("mirt")
# install.packages("ggplot2")
# install.packages("dplyr")

library(mirt)
library(ggplot2)
library(dplyr)

input_files <- c("./clean_data/divided_base_matrix.csv", 
                 "./clean_data/divided_perturb1_matrix.csv", 
                 "./clean_data/divided_perturb2_matrix.csv")

output_files_base <- c("./model_coef/divided_base_coef_1PL.csv", 
                       "./model_coef/divided_base_coef_2PL.csv", 
                       "./model_coef/divided_base_coef_3PL.csv")

output_files_perturb1 <- c("./model_coef/divided_perturb1_coef_1PL.csv", 
                           "./model_coef/divided_perturb1_coef_2PL.csv", 
                           "./model_coef/divided_perturb1_coef_3PL.csv")

output_files_perturb2 <- c("./model_coef/divided_perturb2_coef_1PL.csv", 
                           "./model_coef/divided_perturb2_coef_2PL.csv", 
                           "./model_coef/divided_perturb2_coef_3PL.csv")

output_files_list <- list(output_files_base, output_files_perturb1, output_files_perturb2)

models <- c('Rasch', '2PL', '3PL')

for (i in 1:length(input_files)) {
  data <- read.csv(input_files[i], row.names=1)
  
  items_to_remove <- which(apply(data, 2, function(x) length(unique(x)) == 1))
  if(length(items_to_remove) > 0) {
    data <- data[, -items_to_remove]
  }

  for (j in 1:length(models)) {
    model <- mirt(data, 1, models[j])
    coefficient <- coef(model)
    write.csv(coefficient, output_files_list[[i]][j])
    fscores(model)
  }
}


