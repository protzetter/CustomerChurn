# This is the R code customer churn detection and prediction
# Patrick Rotzetter, August 2017
library(keras)
library(dplyr)

#Read training and test files

trainData<-read.csv("Churn_Modelling.csv",header=TRUE,sep = ",", na.strings = "#DIV/0!")

# the first 3 columns are not required for prediction purposes

trainDataCheck<-select(trainData,c(4:14))


# split training and test data

set.seed(101) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(trainDataCheck), size = floor(.75*nrow(trainDataCheck)), replace = F)
train <- trainDataCheck[sample, ]
test  <- trainDataCheck[-sample, ]

#split train data and prediction
yTrain<-select(train,Exited)
train<-select(train,-Exited)
yTest<-select(test,Exited)
test<-select(test,-Exited)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 5, activation = 'relu',input_shape=(c(10))) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 5, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  train, yTrain, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

