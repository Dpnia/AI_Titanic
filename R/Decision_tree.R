train_data <- read.csv("train.csv")
test_data <- read.csv("test.csv")

train_data$Survived <- as.factor(train_data$Survived)
train_data$Pclass <- as.factor(train_data$Pclass)
train_data$Name <- as.character(train_data$Name)
train_data$Ticket <- as.character(train_data$Ticket)
train_data$Cabin <- as.character(train_data$Cabin)

test_data$Pclass <- as.factor(test_data$Pclass)
test_data$Name <- as.character(test_data$Name)
test_data$Ticket <- as.character(test_data$Ticket)
test_data$Cabin <- as.character(test_data$Cabin)

# Age 컬럼의 비어있는 row에 Age 컬럼의 중앙값을 채워준다.
train_data$Age[is.na(train_data$Age)] <- median(train_data$Age, na.rm=TRUE)
test_data$Age[is.na(test_data$Age)] <- median(test_data$Age, na.rm=TRUE)

# Embarked 컬럼의 비어있는 row에 Embarked 컬럼의 최빈값을 채워준다.
levels(train_data$Embarked)[1] <- NA
train_data$Embarked[is.na(train_data$Embarked)] <- names(table(train_data$Embarked))[which.max(table(train_data$Embarked))]

# Gender 컬럼을 새로 만들어, 0, 1로 바꿔준다
train_data$Gender <- ifelse(train_data$Sex=='male',1,0)
test_data$Gender <- ifelse(test_data$Sex=='male',1,0)

# Fare 컬럼의 비어있는 row에 Fare 컬럼의 중앙값을 채워준다.
train_data$Fare[is.na(train_data$Fare)] <- median(train_data$Fare, na.rm=TRUE)
test_data$Fare[is.na(test_data$Fare)] <- median(test_data$Fare, na.rm=TRUE)


library(rpart)
rpartmod <- rpart(Survived ~ Pclass + Gender + Age + SibSp + Parch + Fare + Embarked , data=train_data, method="class")
plot(rpartmod)
text(rpartmod)

rpartpred <- predict(rpartmod, newdata=test_data, type='class')
rpartpred

decision_tree <- data.frame(test_data$PassengerId, rpartpred)
names(decision_tree) <- c("PassengerID", "Survived")
write.csv(decision_tree, "Decision_tree.csv", row.names=FALSE)
