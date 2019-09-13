if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if (!require(dslabs)) install.packages('dslabs')
if (!require(lubridate)) install.packages('lubridate')
if (!require(pdftools)) install.packages('pdftools')
if (!require(rpart)) install.packages('rpart')
if (!require(matrixStats)) install.packages('matrixStats')
if (!require(knitr)) install.packages('knitr')
if (!require(ranger)) install.packages('ranger')
if (!require(rsample)) install.packages('rsample')
if (!require(randomForest)) install.packages('randomForest')
if (!require(randomForest)) install.packages('randomForest')
if (!require(lattice)) install.packages('lattice')

library(caret)
library(dslabs)
library(lubridate)
library(tidyverse)
library(pdftools)
library(rpart)
library(matrixStats)
library(knitr)
library(data.table)
library(rsample)
library(ranger)
library(randomForest)
library(rfUtilities)

#install.packages("ellipse")
################################
# Create training set, validation set
################################


falldetectdf <- read.csv("Data/falldeteciton.csv")

## explore dataset, confirm structure, etc.

head(falldetectdf)
str(falldetectdf)

## recoding activity into bivariate fall/non-fall
## ACTIVITY CODING: 0- Standing 1- Walking 2- Sitting 3- Falling 4- Cramps 5- Running
actdistro <- prop.table(table(falldetectdf$ACTIVITY)) *100
cbind(freq = table(falldetectdf$ACTIVITY), Activity_distro = actdistro)

## FALL_CODED Coding: 0 Non-fall, 1 - Fall

falldetectdf$FALL_CODED <- recode(falldetectdf$ACTIVITY, 
                                  '0' = '0',
                                  '1' = '0',
                                  '2' = '0',
                                  '3' = '1', 
                                  '4' = '0',
                                  '5' = '0')


## convert FALL CODED to factor variable

falldetectdf$FALL_CODED <- as.factor(falldetectdf$FALL_CODED)

levels(falldetectdf$FALL_CODED)


### creating Validation and Training set, Validation set will be 10% of falldetectdf

set.seed(1, sample.kind= "Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead

test_index <- createDataPartition(y = falldetectdf$FALL_CODED, times = 1, p = 0.2, list = FALSE)
train_df <- falldetectdf[-test_index,]  ##  train set
validation <- falldetectdf[test_index,]  ##  test/validation set 


### TRAINING DATA DESCRIPTIVES & CLEANING ####

## understand the dataset dimensions, attibutes
dim(train_df)
str(train_df)
head(train_df)

sapply(train_df, class)


#### Acticity Distributions & Summary Descriptives###
Activity_distro <- prop.table(table(train_df$FALL_CODED)) *100
cbind(freq = table(train_df$FALL_CODED), Activity_distro = Activity_distro)

summary(train_df)

#### Visualizations ####
## univariate box and whisker plots
x <- train_df[,2:7]
y <- train_df[,8]

par(mfrow = c(2,3))
for (i in 1:6){
  boxplot(x[,i], main = names(x)[i])
}


plot(y)

### multivariate plot looking at interactions 

#scatterplot matrix
ellipse <- featurePlot(x = x, y = y, plot = "ellipse")


par(mfrow = c(1,1))

#boxplot matrix
featurePlot(x=x, y=y, plot = "box")

## density plots
scales <- list(x = list(relation ="free"), 
               y = list(relation = "free"))
featurePlot(x=x, 
            y=y, 
            plot="density", 
            scales = scales)

par(mfrow = c(1,1))

## removing Activity column from train so it does not confound the results
train2 <- train_df[,2:ncol(train_df)]

saveRDS(train2, "Data/trimmedTrainData.rds")

##model evaluation

##setting up 10 fold CV
control <- trainControl(method = "cv", number = 10, p = .8)
metric <- "Accuracy"  ## ratio of correctly predicted

#1: linear model
set.seed(1, sample.kind= "Rounding")
fit.lda2 <- train(FALL_CODED~., 
                  data=train2,
                  method = "lda", 
                  metric = metric, 
                  trControl = control)

#2: nonlinear model 
set.seed(1, sample.kind= "Rounding")
fit.cart2 <- train(FALL_CODED~. , 
                   data=train2, 
                   method = "rpart",
                   metric = metric, 
                   trControl = control)

#3 knn
set.seed(1, sample.kind= "Rounding")
fit.knn2 <- train(FALL_CODED~., 
                  data=train2, 
                  method = "knn", 
                  metric = metric, 
                  trControl = control) 



#4: rf
set.seed(1, sample.kind= "Rounding")
fit.rf2 <- train(FALL_CODED~ . , 
                 data=train2,
                 method = "rf", 
                 metric = metric, 
                 trControl = control)

### picking best mtry ###

mtryplot <- ggplot(fit.rf2)

ggsave("Results/mtryplot.png", mtryplot)   #saving to pull into .pdf
fit.rf2$bestTune


fit.rf2_2 <- randomForest(FALL_CODED~., data=train2, 
                          mtry = fit.rf2$bestTune
)
importance(fit.rf2_2)

rf_cm <- confusionMatrix(fit.rf2_2$predicted, train2$FALL_CODED)

RF_CMTable <- data_frame(Accuracy = rf_cm$overall[["Accuracy"]], 
                      Kappa = rf_cm$overall[["Kappa"]],
                      Sensitivity =  rf_cm$byClass[["Sensitivity"]], 
                      Specificity =  rf_cm$byClass[["Specificity"]])

saveRDS(RF_CMTable, "Results/RFModelConfusionMatrix.rds") #saving to pull into .pdf
RF_CMTable %>% knitr::kable()

### taking importance into account in the model
fit.rf2_3 <- randomForest(FALL_CODED~., data=train2,
                                       mtry= fit.rf2$bestTune,
                                       importance = TRUE)
fit.rf2_3

confusionMatrix(fit.rf2_3$predicted, train2$FALL_CODED)



results <- resamples(list(lda=fit.lda2, cart=fit.cart2, knn=fit.knn2, rf=fit.rf2)) 
summary(results)


## creating results table
accTable <- as.data.frame(accTable)

accTable1 <- data_frame(Model = "Linear(lda)",
                       Accuracy = mean(results$values[,"lda~Accuracy"]), 
                       Kappa = mean(results$values[,"lda~Kappa"]))

accTable2 <- data_frame(Model = "Non-Linear",
                                   Accuracy = mean(results$values[,"cart~Accuracy"]), 
                                   Kappa = mean(results$values[,"cart~Kappa"]))


accTable3 <-data_frame(Model = "KNN",
                                   Accuracy = mean(results$values[,"knn~Accuracy"]), 
                                   Kappa = mean(results$values[,"knn~Kappa"]))

accTable4 <-data_frame(Model = "Random Forest",
                                   Accuracy = mean(results$values[,"rf~Accuracy"]), 
                                   Kappa = mean(results$values[,"rf~Kappa"]))
accTable <- rbind(accTable1,accTable2, accTable3, accTable4 )

rm(accTable1, accTable2, accTable3, accTable4)

saveRDS(accTable, "Results/AccuracyTable.rds") #saving to pull into .pdf
accTable %>% knitr::kable()

# compare accuracy of models
dotplot(results)

print(fit.rf2)


## final model 
## trimming the activity column from the dataset so it does not confound results
falldetectdf_trimmed<- falldetectdf[,2:ncol(falldetectdf)]
saveRDS(falldetectdf_trimmed, "Data/falldetectedTrimmedDataset.rds") #saving to pull into .pdf

## running final model
set.seed(1, sample.kind = "Rounding")
fit.rfFINAL <- randomForest(FALL_CODED~., data=falldetectdf_trimmed,
                          mtry = fit.rf2$bestTune$mtry)
cm <- confusionMatrix(fit.rfFINAL$predicted, falldetectdf_trimmed$FALL_CODED)
CMTable <- data_frame(Accuracy = cm$overall[["Accuracy"]], 
                      Kappa = cm$overall[["Kappa"]],
                      Sensitivity =  cm$byClass[["Sensitivity"]], 
                      Specificity =  cm$byClass[["Specificity"]])

saveRDS(CMTable, "Results/FinalModelConfusionMatrix.rds") #saving to pull into .pdf
CMTable %>% knitr::kable()
