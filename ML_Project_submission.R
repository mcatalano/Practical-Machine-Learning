#Load required packages and set seed for reproducibility
library(corrplot)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
set.seed(333)

#Store data files
big_train <- read.csv('pml-training.csv')
test_class <- read.csv('pml-testing.csv')

#Remove columns with NA or "" in 1st row
first_row <- big_train[1, ]
first_row[first_row == ""] <- NA
NA_names <- names(first_row[ ,is.na(first_row)])
new_train <- big_train[ ,!names(big_train) %in% NA_names]

#Subset data to all reasonable predictors
trimmed_train <- new_train[8:60]

#Split data into training and validation sets
part <- createDataPartition(y=trimmed_train$classe, p=0.6, list=FALSE)
training <- trimmed_train[part, ]
testing <- trimmed_train[-part, ]

#Test for near zero variance
nzv <- nearZeroVar(training, saveMetrics=TRUE)
nzv

#PCA on training set minus classe variable
princomp <- prcomp(training[ ,1:52], scale=TRUE)

#Summary shows 80% of variance captured by 12 PCs
summary(princomp)

#Plot variances for each component
screeplot(princomp, type='lines', col=2)

#Biplot shows observation scores (points) and feature loadings (arrows) 
biplot(princomp)

#Plot PC1 vs PC2 colored by activity class.
pcp <- qplot(x=princomp$x[ ,1], 
             y=princomp$x[ ,2], 
             color=training$classe,
             xlab='PC1',
             ylab='PC2')
pcp

#Train a random forest model on training data
rf.mod <- randomForest(classe ~.,
                       data=training,
                       type='class',
                       importance=TRUE,
                       ntree=50)
rf.mod

#Examine importance of variables
varImpPlot(rf.mod)

#Visualize important variables with feature plot
trellis.par.set(caretTheme())
fp <- featurePlot(x=training[ ,c(1,3,39,41)],
                  y=training$classe,
                  plot='pairs',
                  auto.key = list(columns = 5),)
fp

#Predict outcomes on testing set using RF model
rf.pred <- predict(rf.mod, newdata=testing)

#Evaluate model accuracy with confusion matrix
rf.cm <- confusionMatrix(rf.pred, testing$classe)
rf.cm

#Predict outcomes for the orignal test set file
rf.test <- predict(rf.mod, newdata=test_class)


# #Train support vector machine model
# svm.mod <- svm(training$classe ~.,
#                  data=training,
#                  cross=10)
# 
# #Predict outcomes on validation set
# pred.val <- predict(svm.mod, newdata=testing)
# 
# #Evaluate accuracy on validation set
# cm.val <- confusionMatrix(pred.val, testing$classe)
# 
# #Predict outcomes on training set
# pred.train <- predict(svm.mod, newdata=training)
# 
# #Evaluate accuracy on training set
# cm.train <- confusionMatrix(pred.train, training$classe)
# 
# #Predict outcomes on test set
# pred.test <- predict(svm.mod, newdata=test_class[ ,!names(big_train) %in% NA_names])



#Store predicted values as a vector of answers
answers <- as.vector(rf.test)
answers

#Write predicted values to submission text files
write_files <- function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
    }
}

write_files(answers)




