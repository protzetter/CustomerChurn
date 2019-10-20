# This is the R code customer churn detection and prediction
# experiment with Naive Bayes
# Patrick Rotzetter, protzetter@buewin.ch, October 2019

# load required libraries
library(dplyr)
library(e1071)

#Read training and test files

trainData<-read.csv("Churn_Modelling.csv",header=TRUE,sep = ",", na.strings = "#DIV/0!")

# the first 3 columns are not required for prediction purposes

trainDataCheck<-select(trainData,c(4:14))


# split training and test data

set.seed(101) # Set Seed so that same sample can be reproduced in future also

# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(trainDataCheck), size = floor(.8*nrow(trainDataCheck)), replace = F)
train <- trainDataCheck[sample, ]
test  <- trainDataCheck[-sample, ]

#split train data and prediction
yTrain<-select(train,Exited)
yTrain$Exited<-as.factor(yTrain$Exited)
train<-select(train,-Exited)
yTest<-select(test,Exited)
test<-select(test,-Exited)

# train$Geographyy<-as.factor(train$Geography)
# train$Gender<-as.factor(train$Gender)

train<-select(train,c(CreditScore,Gender,Geography,Age,Tenure,Balance,NumOfProducts,EstimatedSalary,IsActiveMember,HasCrCard))
test<-select(test,c(CreditScore,Gender,Geography,Age,Tenure,Balance,NumOfProducts,EstimatedSalary,IsActiveMember,HasCrCard))



# create Naive Bayes classifier

classifier<- naiveBayes(train,yTrain)
