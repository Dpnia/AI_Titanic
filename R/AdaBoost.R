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

# Age NA -> Median of Age
train_data$Age[is.na(train_data$Age)] <- median(train_data$Age, na.rm=TRUE)
test_data$Age[is.na(test_data$Age)] <- median(test_data$Age, na.rm=TRUE)

# Embarked Na -> Mode of Embarked
levels(train_data$Embarked)[1] <- NA
train_data$Embarked[is.na(train_data$Embarked)] <- names(table(train_data$Embarked))[which.max(table(train_data$Embarked))]

# New column Gender -> Male : 1 , FeMale : 0
train_data$Gender <- ifelse(train_data$Sex=='male',1,0)
test_data$Gender <- ifelse(test_data$Sex=='male',1,0)

# Fare NA -> Median of Fare
train_data$Fare[is.na(train_data$Fare)] <- median(train_data$Fare, na.rm=TRUE)
test_data$Fare[is.na(test_data$Fare)] <- median(test_data$Fare, na.rm=TRUE)

library(adabag)
ada_boo <- boosting(Survived ~ Pclass + Gender + Age + SibSp + Parch + Fare + Embarked, data = train_data, boos=TRUE, mfinal = 6)
ada_pre <- predict(ada_boo, test_data)
ada_pre

ada_df <- data.frame(test_data$PassengerId, ada_pre$class)
names(ada_df) <- c("PassengerID", "Survived")
write.csv(ada_df, "AdaBoost.csv", row.names=FALSE)
