## -------------------------------------------------------------------------------------------------##
## Installing required libraries

##install.packages("reshape2")
library(reshape2)

library(e1071)

##install.packages('caret')
library(caret)

##install.packages("corrplot")
library(corrplot)

##install.packages("ggplot2")
library(ggplot2)

##install.packages('dplyr')
library(dplyr)

require(MASS)

##install.packages('clusterGeneration')
require(clusterGeneration)

##install.packages('logistf')
library(logistf)

##install.packages('pscl')
library(pscl)

##install.packages('arm')
library(arm)

library(Rfast)

library(standardize)

library(glmnet)

##install.packages('InformationValue')
library(InformationValue)

##install.packages('mlbench')
library(mlbench)

##install.packages('rattle')
library(rattle)                 # Fancy tree plot

library(rpart.plot)

library(parallel)

library(rpart)

library(randomForest)

library(pROC)

## -------------------------------------------------------------------------------------------------##

#### Set working directory
setwd("/Users/sysadmin/Documents/Project/Submission Files/")

## -------------------------------------------------------------------------------------------------##

## Loading dataset
data  = read.csv("Vinay_BoTNet_IoT.csv")

## -------------------------------------------------------------------------------------------------##
##Pre-processing and Data Analyzing Steps

## Understanding the basic structure of dataset
dim(data)                                             ###(rows & columns)
head(data)                                            ###Displaying top 5 rows in the dataset
tail(data)                                            ###Displaying last 5 rows in the dataset
names(data)                                           ###Display all the names of the column
length(unique(data$ID))                               ###Check for duplicate values or rows
sum(is.na(data))                                      ###To check number of total null values in data frame
apply(data,2,function(x) sum(is.na(x)))               ###To check column wise null values
str(data)                                             ###Checking the dataset structure
summary(data)                                         ###Summary of the Dataset


## -------------------------------------------------------------------------------------------------##


plot.new()   #What is this for 
frame()      #What is this for

## -------------------------------------------------------------------------------------------------##
## Understanding the target variable

counts <- table(data$label)
print(counts)
#png("LabelDistribution.png")
barplot(counts,main='Label Distribution')

## -------------------------------------------------------------------------------------------------##
## Plotting the first column to understand distribution

png("DataDistribution.png")
plot(data$MI_dir_L5_weight,data$label)

## -------------------------------------------------------------------------------------------------##
## Lets seperate the data into 2 sets based on Labels - 0  and 1

data0 <- data[data$label == 0, ]
data1 <- data[data$label == 1, ]

## -------------------------------------------------------------------------------------------------##

## Plotting benign Vs Junk traffic distribution for each column
## Since there are a lot of columns, will run a loop and compare how both the label wise data is distributed
for (i in 1:ncol(data)) {
  png(file = paste("Comparetraffic - Column", i, ".jpg", sep=""))
  par(mfrow=c(1,2))
  plot(data0[, i],main=colnames(data0[i]),col='green')
  plot(data1[, i],main=colnames(data1[i]),col='red')
  dev.off()
}

## -------------------------------------------------------------------------------------------------##

## Lets look at the datatypes and the range of values in each column
str(data)

## -------------------------------------------------------------------------------------------------##

## PCA to reduce dimensionality
## seperating target variable
data_pca <- data.frame(data)
str(data_pca)
#splitting into train and test for pca
require(caTools)
set.seed(101)
sample = sample.split(data_pca$ID, SplitRatio = .75)
train = subset(data_pca, sample == TRUE)
test = subset(data_pca, sample == FALSE)

## Removing target and identifier variables from train and test dataset
train_pca <- train
train_pca$ID <- NULL
train_pca$label <- NULL

## Saving test label into predictions dataframe 
predictions_df <- data.frame(test$label)

test_pca <- test
test_pca$label <- NULL
test_pca$ID <- NULL

## PCA for dimensionality reduction
prin_comp <- prcomp(train_pca, scale. = T)
names(prin_comp)

#mean of variables
prin_comp$center

#standard deviation of variables
prin_comp$scale

#principal component loading vectors
prin_comp$rotation
prin_comp$rotation[1:5,1:4]
dim(prin_comp$x)
png("PCA.png")
biplot(prin_comp, scale = 0)

## Checking the stddev and variance
std_dev <- prin_comp$sdev
pr_var <- std_dev^2

#check variance of first 10 components
pr_var[1:10]

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:21]

#scree plot
png("PCA_SCREE.png")
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")


#cumulative scree plot
png("PCA_SCREE_CUM.png")
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#Creating Training set with principal components
dim(prin_comp$x)
train.data <- data.frame(label = train$label, prin_comp$x)

dim(train.data)

#Selecting first 20 PCAs and adding target variable
train.data <- train.data[,1:21]
train.data$label <- train$label
train.data$label<-as.factor(train.data$label)
dim(train.data)
colnames(train.data)

#Convering test data into PCA
test.data <- predict(prin_comp, newdata = test_pca)
test.data <- as.data.frame(test.data)
colnames(test.data)
test.data <- test.data[,1:20]
test.data$label <- test$label
test.data$label<-as.factor(test.data$label)
dim(test.data)
colnames(test.data)

## -------------------------------------------------------------------------------------------------##

## Modelling with logistic regression
set.seed(1)
system.time(model_lg <- train(label ~., data=train.data, method='glm',
                              tuneGrid=expand.grid(parameter=c(1,10,100, 1000))))    ## Building logistic model and tuning with parameters(1,10,100,1000)
model_lg
pred_train_lg <-predict(model_lg,train.data,type='raw')
pred_train_lg <- as.numeric(pred_train_lg)
pred_train_lg[pred_train_lg=='1']<- 0
pred_train_lg[pred_train_lg=='2']<- 1
pred_train_lg

pred_test_lg <- predict(model_lg,test.data,type='raw')
pred_test_lg <- as.numeric(pred_test_lg)
pred_test_lg[pred_test_lg=='1']<- 0
pred_test_lg[pred_test_lg=='2']<- 1
pred_test_lg

str(predictions_df)

#Confusion matrix
confusionMatrix(test$label,pred_test_lg)


# Accuracy
acc_train_lg<-1-misClassError(train$label,pred_train_lg)
acc_test_lg<-1-misClassError(test$label,pred_test_lg)
results_df <- data.frame("Model"="Logistic","Train Accuracy"=acc_train_lg,"Test Accuracy"=acc_test_lg)
results_df

## Plotting ROC/ AUC curves
roc1 = roc(test$label,pred_test_lg)
plot(roc1, main='ROC curve for Logistic Regression')


## -------------------------------------------------------------------------------------------------##

## Modelling with Decision tree
library(rpart)
system.time(model_dt <- rpart(label~., data = train.data))
model_dt
train.data$label
pred_train_dt <- predict(model_dt, train.data,type = 'class')
pred_train_dt <- as.numeric(pred_train_lg)
pred_test_dt <- predict(model_dt, test.data,type='class')
pred_test_dt <- as.numeric(pred_test_lg)
predictions_df$DecisionTree <- pred_test_dt
predictions_df$Logistic <- pred_test_lg

str(pred_test_dt)
confusionMatrix(test$label,pred_test_dt)

# Accuracy
acc_train_dt<-1-misClassError(train.data$label,pred_train_dt)
acc_test_dt<-1-misClassError(test.data$label,pred_test_dt)
add_dt <- data.frame("Model"='DecisionTree',"Train Accuracy"=acc_train_dt, "Test Accuracy"=acc_test_dt)
results_df <- rbind(results_df,add_dt)
results_df

## Plotting ROC/ AUC curves
roc2 = roc(test$label,pred_test_dt)
plot(roc2, main='ROC curve for Decision Tree')

## -------------------------------------------------------------------------------------------------##

## Modelling with RandomForest
##install.packages('NMF')
library(NMF)
library(rpart)
library(randomForest)

model_rf <- randomForest(label~.,train.data, importance = TRUE)
model_rf
pred_train_rf <- predict(model_rf,train.data,type='response')
pred_train_rf <- as.numeric(pred_train_rf)
pred_train_rf[pred_train_rf=='1']<- 0
pred_train_rf[pred_train_rf=='2']<- 1
pred_train_rf

pred_test_rf <- predict(model_rf, test.data,type='response')
pred_test_rf <- as.numeric(pred_test_rf)
pred_test_rf[pred_test_rf=='1']<- 0
pred_test_rf[pred_test_rf=='2']<- 1
pred_test_rf

predictions_df$RandomForest <- pred_test_rf

confusionMatrix(test.data$label,pred_test_rf)

# Accuracy
acc_train_rf<-1-misClassError(train.data$label,pred_train_rf)
acc_test_rf<-1-misClassError(test.data$label,pred_test_rf)
add_rf <- data.frame("Model"='RandomForest',"Train Accuracy"=acc_train_rf,"Test Accuracy"=acc_test_rf)
results_df <- rbind(results_df,add_rf)
results_df

## Plotting ROC/ AUC curves
roc3 = roc(test$label,pred_test_rf)
plot(roc3, main='ROC curve for RandomForest')


## -------------------------------------------------------------------------------------------------##

## Ensembling

## Majority voting

str(predictions_df)

predictions_df$ensembleVoting = as.factor(ifelse(predictions_df$Logistic==1 & predictions_df$DecisionTree==1,1,ifelse(predictions_df$RandomForest==1 & predictions_df$Logistic==1,1,ifelse(predictions_df$RandomForest==1 & predictions_df$DecisionTree==1,1,0))))
predictions_df
str(predictions_df)
predictions_df$ensembleVoting <- as.numeric(predictions_df$ensembleVoting)
predictions_df$ensembleVoting[predictions_df$ensembleVoting=='1']<- 0
predictions_df$ensembleVoting[predictions_df$ensembleVoting=='2']<- 1
predictions_df


confusionMatrix(predictions_df$test.label,predictions_df$ensembleVoting)

# Accuracy
acc_test_ev<-1-misClassError(test.data$label,predictions_df$ensembleVoting)
add_ev <- data.frame("Model"='Ensemble',"Train Accuracy"=1, "Test Accuracy"=acc_test_ev)
results_df <- rbind(results_df,add_ev)
results_df
library(pROC)

## Plotting ROC/ AUC curves
roc4 = roc(test$label,predictions_df$ensembleVoting)
plot(roc4, main='ROC curve for Ensemble')

## -------------------------------------------------------------------------------------------------##

results_df
predictions_df
