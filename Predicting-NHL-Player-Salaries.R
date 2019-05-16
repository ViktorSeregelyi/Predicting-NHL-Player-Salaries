library(mlbench)
library(caret)
library(glmnet)
library(neuralnet)
library(party)

# Predicting NHL player salaries

# data from http://www.hockeyabstract.com
setwd("C:\\Users\\Viktor\\Desktop")
dat = read.csv("NHL_2016-17_Cleaned_v2.csv", header = T)
dat = na.omit(dat)
dat$IPP. = as.numeric(dat$IPP.)

#Summary Statistics
subdat = dat[,c(1,5,6,12:14,17:20,24,27,36,38,42,50,57,145,152)]
sumstat = sapply(subdat, mean)

sumstat = data.frame("Statistic" = colnames(subdat),
                     "Mean" = sapply(subdat, mean), 
                     "Median" = sapply(subdat, median), 
                     "Standard Deviation" = sapply(subdat, sd),
                     "Minimum" = sapply(subdat, min),
                     "Maximum" = sapply(subdat, max))
sumstat

#Divide into training and test sets
set.seed(42)
idx = createDataPartition(subdat$Salary, p = 0.8, list = FALSE)
dat_trn = subdat[idx, ]
dat_tst = subdat[-idx, ]

# do feature selection using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=5)
results <- rfe(dat_trn[,c(1:18)], dat_trn[,19],
               sizes=c(1:10), rfeControl=control)
print(results)
# the chosen features
predictors(results)
plot(results, type=c("g", "o"))

#Create subset of dataset using the optimal selection of features
dat_trn = dat_trn[,c(19, 1, 11, 6, 4, 14, 13, 7, 15, 17)]
dat_tst = dat_tst[,c(19, 1, 11, 6, 4, 14, 13, 7, 15, 17)]

dat_trn$Age <- scales::rescale(dat_trn$Age, to=c(0,1))
dat_trn$TOI.GP <- scales::rescale(dat_trn$TOI.GP, to=c(0,1))
dat_trn$A <- scales::rescale(dat_trn$A, to=c(0,1))
dat_trn$iCF <- scales::rescale(dat_trn$iCF, to=c(0,1))
dat_trn$iFF <- scales::rescale(dat_trn$iFF, to=c(0,1))
dat_trn$PTS <- scales::rescale(dat_trn$PTS, to=c(0,1))
dat_trn$iBLK <- scales::rescale(dat_trn$iBLK, to=c(0,1))
dat_trn$GP <- scales::rescale(dat_trn$GP, to=c(0,1))
dat_trn$ixG <- scales::rescale(dat_trn$ixG, to=c(0,1))

dat_tst$Age <- scales::rescale(dat_tst$Age, to=c(0,1))
dat_tst$TOI.GP <- scales::rescale(dat_tst$TOI.GP, to=c(0,1))
dat_tst$A <- scales::rescale(dat_tst$A, to=c(0,1))
dat_tst$iCF <- scales::rescale(dat_tst$iCF, to=c(0,1))
dat_tst$iFF <- scales::rescale(dat_tst$iFF, to=c(0,1))
dat_tst$PTS <- scales::rescale(dat_tst$PTS, to=c(0,1))
dat_tst$iBLK <- scales::rescale(dat_tst$iBLK, to=c(0,1))
dat_tst$GP <- scales::rescale(dat_tst$GP, to=c(0,1))
dat_tst$ixG <- scales::rescale(dat_tst$ixG, to=c(0,1))

x_train = dat_trn[,c(2:10)]
x_test = dat_tst[,c(2:10)]

y_train = dat_trn[,c(1)]
y_test = dat_tst[,c(1)]


cv_5 = trainControl(method = "cv", number = 5)
oob = trainControl(method = "oob", number = 5)



# The models
linGrid = expand.grid(C = c(2^ (-5:5)))
SVM_Lin = train(x_train, y_train,
                method="svmLinear", tuneGrid= linGrid,
                trControl = cv_5)
SVM_Lin
sqrt((mean((predict(SVM_Lin, x_test) - y_test)^2)))


radGrid = expand.grid(sigma= 2^c(-25, -20, -15,-10, -5, 0), C= 2^c(0:5))
SVM_Rad = train(x_train, y_train,
                method="svmRadial", tuneGrid= radGrid,
                trControl = cv_5)
SVM_Rad
sqrt((mean((predict(SVM_Rad, x_test) - y_test)^2)))


k_set = data.frame(k=c(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25))
knn = train(x_train, y_train,
            method="knn", tuneGrid= k_set,
            trControl = cv_5)
knn
sqrt((mean((predict(knn, x_test) - y_test)^2)))


rf = train(x_train, y_train,
           method="rf", trControl = oob)
rf
sqrt((mean((predict(rf, x_test) - y_test)^2)))


nnet = train(x_train, y_train,
             method = "brnn",
             trControl = cv_5)
nnet
sqrt((mean((predict(nnet, x_test) - y_test)^2)))


glmboost = train(x_train, y_train,
                 method="glmboost",
                 trControl = cv_5)
glmboost
sqrt((mean((predict(glmboost, x_test) - y_test)^2)))

# Results
Model = c()
Model[1] = "Linear Grid SMV"
Model[2] = "Radial Grid SVM"
Model[3] = "KNN"
Model[4] = "Random Forest"
Model[5] = "Bayesian Regularized Neural Network"
Model[6] = "Boosted Generalized Linear Model"

CV.RMSE = c()
CV.RMSE[1] = min(SVM_Lin$results$RMSE)
CV.RMSE[2] = min(SVM_Rad$results$RMSE)
CV.RMSE[3] = min(knn$results$RMSE)
CV.RMSE[4] = min(rf$results$RMSE)
CV.RMSE[5] = min(nnet$results$RMSE)
CV.RMSE[6] = min(glmboost$results$RMSE)

TestRMSE = c()
TestRMSE[1] = sqrt((mean((predict(SVM_Lin, x_test) - y_test)^2)))
TestRMSE[2] = sqrt((mean((predict(SVM_Rad, x_test) - y_test)^2)))
TestRMSE[3] = sqrt((mean((predict(knn, x_test) - y_test)^2)))
TestRMSE[4] = sqrt((mean((predict(rf, x_test) - y_test)^2)))
TestRMSE[5] = sqrt((mean((predict(nnet, x_test) - y_test)^2)))
TestRMSE[6] = sqrt((mean((predict(glmboost, x_test) - y_test)^2)))

frame = data.frame("Model" = c(Model),
                   "CV Train RMSE" = c(CV.RMSE),
                   "Test RMSE" = c(TestRMSE))
frame

# feature importance
varImp(knn)
