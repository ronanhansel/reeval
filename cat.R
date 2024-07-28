library(catR)

item.bank <- read.csv("path/to/item_bank.csv")
resp

theta= rep(0,np) #intial starting value
test <- list(method = 'EAP', itemSelect = method, infoType = "Fisher")
final <- list(method = 'EAP')
stop<-list(rule="length",thr=stoplen)
start <- list(nrItems=skipitems,theta = theta[i],startSelect="MFI", seed=123) #try startSelect bOpt
res <- randomCAT(itemBank = item.bank,
                    responses = as.numeric(resp[i,]),
                    start = start,
                    test = test,
                    final = final,
                    stop = stop)
