## Installing libraries
##install.packages("reshape2")
library(reshape2)

##install.packages("ggplot2")
library(ggplot2)

##install.packages("corrplot")
library(corrplot)

##install.packages('dplyr')
library(dplyr)

require(MASS)

##install.packages('clusterGeneration')
require(clusterGeneration)

##install.packages('logistf')
library(logistf)

##install.packages('pscl')
library(pscl)

##install.packages('caret')
library(caret)

##install.packages('arm')
library(arm)

## -------------------------------------------------------------------------------------------------##

##Load Data 
data  = read.csv("Vinay_BoTNet_IoT.csv")

## -------------------------------------------------------------------------------------------------##

##Pre-processing and Data Analyzing Steps
###Check for Dataset Dimentions
dim(data) #(rows & columns)

###Displaying top 5 and last 5 rows in the dataset
head(data,5)
tail(data,5)

###Display all the names of the column
names(data)

###Check for duplicate values or rows
length(unique(data$ID))

###Check for any missing values in the dataset
sum(is.na(data))  #To check number of total null values in data frame
apply(data,2,function(x) sum(is.na(x))) #To check column wise null values

###Checking the dataset structure
str(data)

###Summary of the Dataset
summary(data)

###Plotting label distribution for class imbalance
counts <- table(data$label)
jpeg("LabelDistribution.jpg")
barplot(counts,main='Label Distribution')
dev.off()

### Since the data seems to have outliers, calculating cooks distance
mod <- lm(label ~ ., data=data)
cooksd <- cooks.distance(mod)
jpeg("Cookesdistance.jpg")
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")

cooksd>4*mean(cooksd, na.rm=T)

###for the following rows, 31 273 487 549 701 751 959. Hence they are being dropped from the dataset
data <- data[-c(31,273,487,549,701,751,959),]

###Now to find the anomali in the data. I will be creating new datasets with label 1 and 0 to plot
#Splitting the data based on  
data0 <- data[data$label == 0, ]
data1 <- data[data$label == 1, ]

### Plotting traffic distribution for each column

for (i in 1:ncol(data)) {
  png(file = paste("Comparetraffic", i, ".jpg", sep=""))
  par(mfrow=c(1,2))
  plot(data0[, i],main=colnames(data0[i]))
  plot(data1[, i],main=colnames(data1[i]))
  dev.off()
}

### Seperating dataset into stream wise
MI <- data[ , grepl("MI_dir",names(data))]
MI <- cbind(MI,data$ID,data$label)
head(MI)

### Seperating dataset into stream wise
H <- data[ , grepl("H_",names(data))]
H <- cbind(H,data$ID,data$label)
head(H)

### Seperating dataset into stream wise
HH <- data[ , grepl("HH_",names(data))]
HH <- cbind(HH,data$ID,data$label)
head(HH)

### Seperating dataset into stream wise
HH_jit <- data[ , grepl("HH_jit",names(data))]
HH_jit <- cbind(HH_jit,data$ID,data$label)
head(HH_jit)

### Seperating dataset into stream wise
HpHp <- data[ , grepl("HpHp_",names(data))]
HpHp <- cbind(HpHp,data$ID,data$label)
head(HpHp)

###Checking Corelation stream wise
MI_cor <- cor(MI)
jpeg("MI_Correlogram.jpg")
corrplot(MI_cor, method='color')
dev.off()

H_cor <- cor(H)
jpeg("H_Correlogram.jpg")
corrplot(H_cor,method='color')
dev.off()

HH_cor <- cor(HH)
jpeg("HH_Correlogram.jpg")
corrplot(HH_cor,method='color')
dev.off()

HH_jit_cor <- cor(HH_jit)
jpeg("HH_jit_Correlogram.jpg")
corrplot(HH_jit_cor,method='color')
dev.off()

HpHp_cor <- cor(HpHp)
jpeg("HpHp_Correlogram.jpg")
corrplot(HpHp_cor,method='color')
dev.off()

## -------------------------------------------------------------------------------------------------##

###Changing to target to categorical
data$label <- as.factor(data$label)
str(data$label)
###Seperating y from dataset
y <- data.frame(label=data[,c(117)])
y
data$label <- NULL
data$ID<- NULL

## -------------------------------------------------------------------------------------------------##

###Standardizing the data
##install.packages("standardize")
library(standardize)
scaled_df <- scale(data, center = TRUE, scale = TRUE)
scaled_df
scaled_df <- data.frame(cbind(scaled_df,y))
scaled_df
scaled_df$label <- as.factor(scaled_df$label)

## -------------------------------------------------------------------------------------------------##


###Train-Test Split
require(caTools)
set.seed(101)
sample = sample.split(scaled_df$label, SplitRatio = .75)
train = subset(scaled_df, sample == TRUE)
test  = subset(scaled_df, sample == FALSE)

head(train)
head(test)


## -------------------------------------------------------------------------------------------------##

###Modeling with all the columns

model1 <- glm(label ~.,family='binomial',data=train,maxit = 100)
summary(model1)


model2 <- logistf(label ~., data=train,firth=FALSE,pl=TRUE)
summary(model2)


fitted.results <- model2$predict
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != train$label)
print(paste('Accuracy',1-misClasificError))

## -------------------------------------------------------------------------------------------------##


