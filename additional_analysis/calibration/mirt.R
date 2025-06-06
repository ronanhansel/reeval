library(mirt)
data <- read.csv("Subset.csv", header = FALSE)
data[data == -1] <- NA
# (model <- mirt(data, 1, SE = TRUE, technical=list(NCYCLES=5000), method = 'MHRM'))
(model <- mirt(data, 1, itemtype = 'Rasch', technical=list(NCYCLES=5000), method = 'EM'))
# model <- mirt(data, 1, itemtype = "Rasch")

fscores_out <- fscores(model, method="MAP", full.scores = TRUE, full.scores.SE=TRUE)

abilities <- fscores_out[,"F1"]
abilities_se <- fscores_out[,"SE_F1"]

diff = coef(model, simplify=TRUE, IRTpars = TRUE)
difficulty <- diff$items   # Extract item difficulty parameters
write.table(difficulty, file = "item_difficulty.csv", 
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(abilities, file = "ability_MAP.csv", 
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(abilities_se, file = "ability_SE.csv", 
            sep = ",", row.names = FALSE, col.names = FALSE)