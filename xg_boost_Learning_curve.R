# 1
####### Call libraries #################
rm(list = ls()) ; gc()
library(ggplot2)
library(caret)          # For dummyVar and for data partitioning
library(xgboost)        # For regression tree
library(Matrix)         # For sparse matrix
library(Metrics)        # for rmse()


# 2
# Read files and process
setwd("C:\\Users\\user\\Desktop\\claim")
# Read input file
tr<-read.csv(file.choose(), header=TRUE)
names(tr)
head(tr)
str(tr)

#2.1 Removing ID 
tr$Member_ID <- NULL
tr$BENEFIT_CD <- NULL

# 3. Create partition
trIndex <- createDataPartition(tr$COB_SAVINGS_AMT, p = 0.7, list = F)
train<-tr[trIndex,]
test<-tr[-trIndex,]


## 4
sparse_matrix <- sparse.model.matrix( ~., data = train)
summary(sparse_matrix)
colnames(sparse_matrix)
sparse_matrix@Dimnames
sparse_matrix[1,1]

# 5. Model now
model <- xgboost(data = sparse_matrix,
                 label = train$COB_SAVINGS_AMT,
                 max.depth = 4,
                 eta = 1,      # set 0, 0.5 and 1
                 nthread = 2,
                 nround = 100,  
                 objective = "reg:linear")


model

# 6. Convert test to sparse matrix
test_new <- read.csv(file.choose(),header = T)

test <- test_new

test$Member_ID <- NULL
test$BENEFIT_CD <- NULL
sparse_matrix_test <- sparse.model.matrix( ~., data = test)
summary(sparse_matrix_test)
colnames(sparse_matrix_test)
sparse_matrix_test@Dimnames

# 7 Make predictions
pred <- predict(model, sparse_matrix_test)
pred

# 8. Check RMSE
rmse(pred,test$COB_SAVINGS_AMT)

#Check the match and unmatch
test_new$Pred <- round(pred)
test_new$result <- ifelse(test_new$COB_SAVINGS_AMT==test_new$Pred, "match","Unmatch")

#show in graph match and unmatch
barplot(table(test_new$result),col = c("forestgreen", "deepskyblue"))

#show the unmatch rows
test_new[test_new$result == "Unmatch",1:8]

View(test_new)

#Learning Curve

# create empty data frame 
learnCurve <- data.frame(m = integer(nrow(train)),
                         trainRMSE = integer(nrow(train)),
                         cvRMSE = integer(nrow(train)))


metric <- "RMSE"

# loop over training examples
for (i in 3:nrow(sparse_matrix)) {
  learnCurve$m[i] <- i
  # train learning algorithm with size i
  model <- xgboost(data = sparse_matrix[1:i,],
                   label = train$COB_SAVINGS_AMT[1:i],
                   max.depth = 4,
                   eta = 1,      # set 0, 0.5 and 1
                   nthread = 2,
                   nround = 100,  
                   objective = "reg:linear")
    # train learning algorithm with size i       
  learnCurve$trainRMSE[i] <- model$evaluation_log$train_rmse
  
  # use trained parameters to predict on test data
  prediction <- predict(model, newdata = sparse_matrix_test)
  rmse <- postResample(prediction, test$COB_SAVINGS_AMT)
  learnCurve$cvRMSE[i] <- rmse[1]
}

pdf("LearningCurve.pdf", width = 7, height = 7, pointsize=12)

# plot learning curves of training set size vs. error measure
# for training set and test set
plot(log(learnCurve$trainRMSE),type = "o",col = "red", xlab = "Training set size",
     ylab = "Error (RMSE)", main = "Linear Model Learning Curve")
lines(log(learnCurve$cvRMSE), type = "o", col = "blue")
legend('topright', c("Train error", "Test error"), lty = c(1,1), lwd = c(2.5, 2.5),
       col = c("red", "blue"))

dev.off()
