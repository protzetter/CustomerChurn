# This is the R code customer churn detection and prediction
# Patrick Rotzetter, August 2017
library(caret)
library(xgboost)
library(keras)
library(dplyr)
library(forcats )
library(yardstick)
library(tidyquant)
library(lime)
library(pROC)	
library(automl)
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
train<-select(train,-Exited)
yTest<-select(test,Exited)
test<-select(test,-Exited)

# train$Geographyy<-as.factor(train$Geography)
# train$Gender<-as.factor(train$Gender)

train<-select(train,c(CreditScore,Gender,Geography,Age,Tenure,Balance,NumOfProducts,EstimatedSalary,IsActiveMember,HasCrCard))
test<-select(test,c(CreditScore,Gender,Geography,Age,Tenure,Balance,NumOfProducts,EstimatedSalary,IsActiveMember,HasCrCard))

#trainMatrix<-data.matrix(train[yTrain==0,])
trainMatrix<-as.matrix(as.data.frame(lapply(train[yTrain==0,], as.numeric)))
#testMatrix<-data.matrix(test[yTest==0,])
testMatrix<-as.matrix(as.data.frame(lapply(test[yTest==0,], as.numeric)))
#fullTrainMatrix<-data.matrix(train)
fullTrainMatrix<-as.matrix(as.data.frame(lapply(train, as.numeric)))
#fullTestMatrix<-data.matrix(test)
fullTestMatrix<-as.matrix(as.data.frame(lapply(test, as.numeric)))


#automl experiment
fullTrainMatrix<-data.matrix(train)
fullTrainMatrix<-scale(fullTrainMatrix, scale = TRUE)
fullTestMatrix<-data.matrix(test)
yTrainMatrix<-as.matrix(yTrain,dimnames=NULL)
yTestMatrix<-as.matrix(yTest,dimnames=NULL)
amlmodel = automl_train_manual(
  Xref = fullTrainMatrix, Yref = yTrainMatrix,
  hpar = list(
    modexec = 'trainwpso',
    layersshape = c(10, 0),
    layersacttype = c('relu', 'softmax'),
    layersdropoprob = c(0, 0.5),
    numiterations = 50,
    psopartpopsize = 50
  )
)

res <- cbind(yTestMatrix, automl_predict(model = amlmodel, X = fullTestMatrix))
colnames(res) <- c('actual', 'predict')
head(res)
confusionMatrix(as.factor(res[,1]),as.factor(res[,2]),positive = '1')
# trainMatrix<-cbind(trainMatrix,geocat[,1:3])
# trainMatrix<-cbind(trainMatrix,gendercat[,1:2])
trainMatrix<-scale(trainMatrix, scale = TRUE)
testMatrix<-scale(testMatrix, scale = TRUE)
fullTrainMatrix<-scale(fullTrainMatrix, scale = TRUE)
fullTestMatrix<-scale(fullTestMatrix, scale = TRUE)
y_mean <- attributes(test)$'scaled:center' #the mean
y_std <- attributes(test)$'scaled:scale'   #the standard deviation
yTrainMatrix<-as.matrix(yTrain,dimnames=NULL)



#autoencoder model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 32, activation = "tanh", input_shape = c(10)) %>%
#  layer_dense(units = 128, activation = "tanh", input_shape = c(10), kernel_regularizer = regularizer_l1(l = 0.01)) %>%
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 32, activation = "tanh") %>%
  layer_dense(units = c(10))

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)


history <- model %>% fit(
  trainMatrix, trainMatrix, 
  epochs = 50, batch_size = 32,
  validation_data = list(testMatrix, testMatrix)
)


#Measure MSE for the model

pred_train<-model %>% predict(trainMatrix)
mse_train <- apply((trainMatrix - pred_train)^2, 1, sum)

pred_test<-model %>% predict(fullTestMatrix)
mse_test <- apply((fullTestMatrix - pred_test)^2, 1, sum)


library(Metrics)
auc(as.matrix(yTrain), mse_train)
auc(as.matrix(yTest), mse_test)



possible_k <- seq(0, 3, length.out = 100)
precision <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(mse_test > k)
  sum(predicted_class == 1 & yTest$Exited == 1)/sum(predicted_class)
})

qplot(possible_k, precision, geom = "line") + labs(x = "Threshold", y = "Precision")


recall <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(mse_test > k)
  sum(predicted_class == 1 & yTest$Exited == 1)/sum(yTest)
})
qplot(possible_k, recall, geom = "line") + labs(x = "Threshold", y = "Recall")


predicted_class <- as.numeric(mse_test >0.025)
confusionMatrix(as.factor(predicted_class),as.factor(yTest$Exited),positive = '1')



# #standard model for classification
# 
# model <- keras_model_sequential() 
# model %>% 
#   layer_dense(units = 16, activation = 'relu',input_shape=(c(10)),kernel_initializer = 'uniform') %>% 
#   layer_dropout(rate = 0.1) %>% 
#   layer_dense(units = 8, activation = 'relu',kernel_initializer = 'uniform') %>%
#   layer_dropout(rate = 0.1) %>%
#   layer_dense(units = 1, activation = 'sigmoid',kernel_initializer = 'uniform')
# 
# model %>% compile(
#   loss = 'binary_crossentropy',
#   optimizer = optimizer_adam(),
#   metrics = "accuracy"
# )
# 
# 
# history <- model %>% fit(
#   trainMatrix, yTrainMatrix, 
#   epochs = 100, batch_size = 32, validation_split = .2, class_weight = list("0"=1, "1"=4)
# )
# 
# 
# 
# 
# predictions<-model %>% predict_classes(testMatrix)

predicted_class <- as.data.frame(as.integer(mse_test > .004))
colnames(predicted_class)<-c('Exited')

confusionMatrix(as.factor(predicted_class$Exited),as.factor(yTest$Exited),positive = '1')

# Plot the model loss of the training data
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")

# Plot the model loss of the test data
lines(history$metrics$val_loss, col="green")

# Add legend
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the accuracy of the training data 
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")

# Plot the accuracy of the validation data
lines(history$metrics$val_acc, col="green")

# Add Legend
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))




# Setup lime::model_type() function for keras
model_type.keras.models.Sequential <- function(x, ...) {
  return("classification")
}


# Setup lime::predict_model() function for keras
predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}



# Test our predict_model() function
predict_model(x = model, newdata = testMatrix, type = 'raw') %>%
  tibble::as_tibble()



# Run lime() on training set
explainer <- lime::lime(
  x              = as.data.frame(trainMatrix), 
  model          = model, 
  bin_continuous = FALSE)

explanation <- lime::explain(
  as.data.frame(testMatrix)[10:20,], 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 4,
  kernel_width = 0.5)


plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")





plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Heatmap",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")





#-------------Basic Training using XGBoost in caret Library-----------------
# Set up control parameters for caret::train
# Here we use 10-fold cross-validation, repeating twice, and using random search for tuning hyper-parameters.
library(caret)
library(xgboost)

train <- trainDataCheck[sample, ]
test  <- trainDataCheck[-sample, ]
train$Exited<-as.factor(train$Exited)
test$Exited<-as.factor(test$Exited)
# levels(train$Exited)<-c("No","Yes")
# levels(test$Exited)<-c("No","Yes")

# train a xgbTree model using caret::train
set.seed(1009)

#fitControl <- trainControl(method = "cv", number = 10, search = "random",sampling="up",classProbs = TRUE,summaryFunction=twoClassSummary)
fitControl <- trainControl(method = "cv", number = 10, search = "random")
modelXGB <- train(Exited~., data = train, method = "xgbTree", trControl = fitControl)

set.seed(1009)
fitControl <- trainControl(method = "cv", number = 10, sampling="down",search = "random")
modelXGBUp <- train(Exited~., data = train, method = "xgbTree", trControl = fitControl)

set.seed(1009)
fitControl <- trainControl(method = "cv", number = 10, search = "random")
modelGBM <- train(Exited~., data = train, method = "gbm", trControl = fitControl)

set.seed(1009)
fitControl <- trainControl(method = "cv", number = 10, sampling="up",search = "random")
modelGBMUp <- train(Exited~., data = train, method = "gbm", trControl = fitControl)


# See model results
print(model)

#Look at the confusion matrix
predictions<-predict(modelXGBUp,test)
confusionMatrix(predictions,test$Exited,positive = '1')

#Draw the ROC curve 
model.probs <- predict(model,test,type="prob")

model.ROC <- roc(predictor=model.probs$`0`,
               response=test$Exited,
               levels=rev(levels(factor(test$Exited))))
model.ROC$auc
#Area under the curve: 0.8731
plot.roc(model.ROC,main="GBM ROC",col="green",add=TRUE)

rValues <- resamples(list(xgb=modelXGB,xgbup=modelXGBUp, gbm=modelGBM, gbmup=modelGBMUp))
rValues$values
summary(rValues)

bwplot(rValues,metric="ROC",main="gbm vs xgbtree with and without upsampling")	



# playing with som
library(kohonen)
library(dplyr)
train <- trainDataCheck[sample, ]
train<-select(train,-Exited)
trainM<-data.matrix(train)
trainM<-scale(trainM)
set.seed(20)

som_grid <- somgrid(xdim =10, ydim=10, topo="hexagonal")
som_model <- som(trainM, 
                 grid=som_grid, 
                 rlen=100, 
                 alpha=c(0.05,0.01), 
                 keep.data = TRUE)

plot(som_model, type="changes")

plot(som_model, type="count")

plot(som_model, type="dist.neighbours")
plot(som_model, type="codes")

plot(som_model, type="quality")



coolBlueHotRed <- function(n, alpha = 1) {rainbow(n, end=4/6, alpha=alpha)[n:1]}
par(mfrow=c(3,5))
variableN<-11
for (i in 1:variableN) {
  var <-i #define the variable to plot 
  if (i!=11 && i!=8 && i!= 9 && i!=4 && i!=8 ) {
    var_unscaled <- aggregate(as.numeric(train[,var]), by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2] 
  }
  else {
    if ( i==4) {
    var_unscaled <- aggregate(as.numeric(train[,var]), by=list(som_model$unit.classif), FUN=sum, simplify=TRUE)[,2]
    }
    else{
      var_unscaled <- aggregate(as.numeric(train[,var]), by=list(som_model$unit.classif), FUN=sum, simplify=TRUE)[,2]
    }
  }
  plot(som_model, type = "property", property=var_unscaled, main=names(train)[var], palette.name=coolBlueHotRed)
  
  # plot(som_model, type = "property", property = getCodes(som_model)[,i], main=colnames(getCodes(som_model))[i], palette.name=coolBlueHotRed)
}


var=2
mydata <- getCodes(som_model) 
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var)) 
for (i in 2:10) {
  wss[i] <- sum(kmeans(mydata, centers=i)$withinss)
}
plot(wss)

# feature selection with random forest

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(train[,1:10], train[,11], sizes=c(1:10), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))
