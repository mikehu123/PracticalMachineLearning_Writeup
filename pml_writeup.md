# Practical Machine Learning Class Project Writeup

## Summary

The goal of this project is to build a machine learning prediction model using the data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways, then use the prediction model to predict 20 different test cases of the manner in which the participants did the exercise.

## Data Processing

### Prepare training and cross validation data
We first read in the raw training data csv file, clean up the data by removing columns of all NAs and columns unrelated to accelerometers such as X|user_name|timestamp|new_window|num_window, then partiton data into training data (70%) and cross validation data (30%).


```r
require(caret)
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
require(randomForest)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
trainRaw <- read.csv("pml-training.csv", na.strings = c("NA", ""))
trainNoNA <- trainRaw[, which(colSums(data.frame(is.na(trainRaw))) == 0)]
cols2Remove <- grep("X|user_name|timestamp|new_window|num_window", names(trainNoNA))
trainClean <- trainNoNA[, -cols2Remove]
inTrain <- createDataPartition(y = trainClean$classe, p = 0.7, list = FALSE)
trainData <- trainClean[inTrain, ]
cvData <- trainClean[-inTrain, ]
```



```r
dim(trainData)
```

```
## [1] 13737    53
```

```r
dim(cvData)
```

```
## [1] 5885   53
```

### Prepare testing data
We read in the raw testing data csv file, clean up the data by removing columns of all NAs and columns unrelated to accelerometers such as X|user_name|timestamp|new_window|num_window|problem_id.


```r
testRaw <- read.csv("pml-testing.csv", na.strings = c("NA", ""))
testNoNA <- testRaw[, which(colSums(data.frame(is.na(testRaw))) == 0)]
cols2RemoveTest <- grep("X|user_name|timestamp|new_window|num_window|problem_id", 
    names(testNoNA))
testData <- testNoNA[, -cols2RemoveTest]
```



```r
dim(testData)
```

```
## [1] 20 52
```


## Predication Model Training, Cross Validation and Out of Sample Error Rate

### Model training

We trained 3 different predication models: rpart, gbm and randomForest. 


```r
fitRPART <- train(classe ~ ., method = "rpart", data = trainData)
```

```
## Loading required package: rpart
```

```r
fitRF <- randomForest(classe ~ ., data = trainData)
# fitGBM <- train(classe ~.,method='gbm', data = trainData2, verbose=FALSE)
```


### Model error rate estimation

gbm estimate of error rate is (1-0.957) = 4.3%
rpart estimate of error rate is (1-0.516) = 48.4%
randomForest OOB estimate of error rate is 0.62%


```r
fitRPART
```

```
## CART 
## 
## 13737 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.04  0.5       0.4    0.06         0.1     
##   0.06  0.4       0.2    0.06         0.1     
##   0.1   0.3       0.06   0.04         0.06    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04.
```

```r
fitRF
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = trainData) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.42%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3905    1    0    0    0    0.000256
## B   12 2640    6    0    0    0.006772
## C    0    8 2385    3    0    0.004591
## D    0    0   20 2231    1    0.009325
## E    0    0    1    6 2518    0.002772
```

```r
# fitGBM
```

### Cross Validation and Out of Sample Error Rate


```r
cvPredictionsRPART <- predict(fitRPART, cvData)
error_rate_oos_RPART <- 1 - sum(cvData$classe == cvPredictionsRPART)/length(cvData$classe)
error_rate_oos_RPART
```

```
## [1] 0.5116
```

```r

# cvPredictionsGBM <- predict(fitGBM,cvData) error_rate_oos_GBM <-
# 1-sum(cvData$classe==cvPredictionsGBM)/length(cvData$classe)

cvPredictionsRF <- predict(fitRF, cvData)
error_rate_oos_RF <- 1 - sum(cvData$classe == cvPredictionsRF)/length(cvData$classe)
error_rate_oos_RF
```

```
## [1] 0.006117
```

Obviously randomForest provides the best accuracy hence was chosen to predict the test data set.

## Predict Test Cases

Finally we used the above-trainined randomForest model on the above-cleaned test data to predict the 20 test cases. The results have been submitted and the accuracy rate was 100%.


```r
testPredictions <- predict(fitRF, testData)
```


===========================================================================
