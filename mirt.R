library(mirt)
library(ggplot2)
library(dplyr)

input_files <- c("data/real/response_matrix/base_matrix.csv", 
                 "data/real/response_matrix/perturb1_matrix.csv", 
                 "data/real/response_matrix/perturb2_matrix.csv")

output_files_base_Z <- c("data/real/irt_reslult/Z/base_1PL_Z.csv", 
                         "data/real/irt_reslult/Z/base_2PL_Z.csv", 
                         "data/real/irt_reslult/Z/base_3PL_Z.csv")
output_files_perturb1_Z <- c("data/real/irt_reslult/Z/perturb1_1PL_Z.csv", 
                             "data/real/irt_reslult/Z/perturb1_2PL_Z.csv", 
                             "data/real/irt_reslult/Z/perturb1_3PL_Z.csv")
output_files_perturb2_Z <- c("data/real/irt_reslult/Z/perturb2_1PL_Z.csv", 
                             "data/real/irt_reslult/Z/perturb2_2PL_Z.csv", 
                             "data/real/irt_reslult/Z/perturb2_3PL_Z.csv")
output_files_list_Z <- list(output_files_base_Z, output_files_perturb1_Z, output_files_perturb2_Z)

output_files_base_theta <- c("data/real/irt_reslult/theta/base_1PL_theta.csv", 
                             "data/real/irt_reslult/theta/base_2PL_theta.csv", 
                             "data/real/irt_reslult/theta/base_3PL_theta.csv")
output_files_perturb1_theta <- c("data/real/irt_reslult/theta/perturb1_1PL_theta.csv", 
                                 "data/real/irt_reslult/theta/perturb1_2PL_theta.csv", 
                                 "data/real/irt_reslult/theta/perturb1_3PL_theta.csv")
output_files_perturb2_theta <- c("data/real/irt_reslult/theta/perturb2_1PL_theta.csv", 
                                 "data/real/irt_reslult/theta/perturb2_2PL_theta.csv", 
                                 "data/real/irt_reslult/theta/perturb2_3PL_theta.csv")
output_files_list_theta <- list(output_files_base_theta, output_files_perturb1_theta, output_files_perturb2_theta)

models <- c('Rasch', '2PL', '3PL')

for (i in 1:length(input_files)) {
  data <- read.csv(input_files[i], row.names=1)
  items_to_remove <- which(apply(data, 2, function(x) length(unique(x)) == 1))
  if(length(items_to_remove) > 0) {
    data <- data[, -items_to_remove]
  }

  for (j in 1:length(models)) {
    model <- mirt(data, 1, models[j])
    Z <- coef(model)
    write.csv(Z, output_files_list_Z[[i]][j])
    
    theta <- fscores(model)
    write.csv(theta, output_files_list_theta[[i]][j])
  }
}
