############### Packages Required ##############

install.packages("naivebayes")
library(naivebayes)

library(ggplot2)

library(caret)

library(psych)

library(e1071)



########## Reading and understanding the data #########


####reading the train dataset #####
train_sal <- read.csv(file.choose())
View(train_sal)
nrow(train_sal)#[1] 30161
ncol(train_sal)#[1] 14
colnames(train_sal)
str(train_sal)

train_sal$educationno <- as.factor(train_sal$educationno)

class(train_sal)


####reading the test dataset ###########

test_sal <- read.csv(file.choose())

str(test_sal)

View(test_sal)
nrow(test_sal)#[1] 15060
ncol(test_sal)#[1] 14

test_sal$educationno <- as.factor(test_sal$educationno)

class(test_sal)

###################### Visualization ##############

##### Plot and ggplot #####

ggplot(data=train_sal,aes(x=train_sal$Salary, y = train_sal$age, fill = train_sal$Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")


plot(train_sal$workclass,train_sal$Salary)


plot(train_sal$education,train_sal$Salary)


plot(train_sal$educationno,train_sal$Salary)


plot(train_sal$maritalstatus,train_sal$Salary)


plot(train_sal$occupation,train_sal$Salary)


plot(train_sal$relationship,train_sal$Salary)


plot(train_sal$race,train_sal$Salary)


plot(train_sal$sex,train_sal$Salary)


ggplot(data=train_sal,aes(x=train_sal$Salary, y = train_sal$capitalgain, fill = train_sal$Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")


ggplot(data=train_sal,aes(x=train_sal$Salary, y = train_sal$capitalloss, fill = train_sal$Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")


ggplot(data=train_sal,aes(x=train_sal$Salary, y = train_sal$hoursperweek, fill = train_sal$Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")


plot(train_sal$native,train_sal$Salary)


########Density Plot ###############

ggplot(data=train_sal,aes(x = train_sal$age, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("Age - Density Plot")

ggplot(data=train_sal,aes(x = train_sal$workclass, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("Workclass Density Plot")

ggplot(data=train_sal,aes(x = train_sal$education, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')

ggtitle("education Density Plot")

ggplot(data=train_sal,aes(x = train_sal$educationno, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("educationno Density Plot")

ggplot(data=train_sal,aes(x = train_sal$maritalstatus, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("maritalstatus Density Plot")

ggplot(data=train_sal,aes(x = train_sal$occupation, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')



ggtitle("occupation Density Plot")

ggplot(data=train_sal,aes(x = train_sal$sex, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("sex Density Plot")

ggplot(data=train_sal,aes(x = train_sal$relationship, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("Relationship Density Plot")

ggplot(data=train_sal,aes(x = train_sal$race, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("Race Density Plot")

ggplot(data=train_sal,aes(x = train_sal$capitalgain, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("Capitalgain Density Plot")

ggplot(data=train_sal,aes(x = train_sal$capitalloss, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("Capitalloss Density Plot")

ggplot(data=train_sal,aes(x = train_sal$hoursperweek, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


ggtitle("Hoursperweek Density Plot")

ggplot(data=train_sal,aes(x = train_sal$native, fill = train_sal$Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')

ggtitle("native Density Plot")



####################### Naive Bayes Model ################################

Model <- naiveBayes(train_sal$Salary ~ ., data = train_sal)
Model

#model prediction
Model_pred <- predict(Model,test_sal)
mean(Model_pred==test_sal$Salary)#[1] 0.8187251


## confusion matrix

confusionMatrix(Model_pred,test_sal$Salary)
####Accuracy : 0.8187
# accuracy is 81.87%