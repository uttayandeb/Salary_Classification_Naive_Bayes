##############################################################################
########################## Naive-Bayes #######################################
##############################################################################


#importing packages and loading the data

import pandas as pd
import numpy as np
salary_train = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Naive Bayes\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Naive Bayes\\SalaryData_Test.csv")

#our targated variable is salary

salary_train.info()
salary_train.dtypes

#creating a list of columns which are objects excluding the "Salary" column
string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

##Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder
#labelling the categorical columns
number = LabelEncoder()
for i in string_columns:
    salary_train[i]= number.fit_transform(salary_train[i])
    salary_test[i]=number.fit_transform(salary_test[i])
    


##Capturing the column names which can help in futher process
colnames = salary_train.columns
colnames
len(colnames)#14






######Splitting the train dta into feature set and tagreted set
x_train = salary_train[colnames[0:13]]
y_train = salary_train[colnames[13]]

##### Splitting the test dat into feature set and targated set
x_test = salary_test[colnames[0:13]]
y_test = salary_test[colnames[13]]





#############################################################
############## MODEL BUILDING ###############################
#############################################################

##Building Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

########  Building the Multinomial naive bayes model #########

classifier_mb = MB()
classifier_mb.fit(x_train,y_train)
pred_mb = classifier_mb.predict(x_train)
accuracy_mb_train = np.mean(pred_mb == y_train)#0.7729186698053778
#77.2%
pd.crosstab(pred_mb, y_train)
#Salary   <=50K   >50K
#row_0                
# <=50K   21717   5913
# >50K      936   1595


##predicting on test data
pred_mb_test = classifier_mb.predict(x_test)
accuracy_mb_test = np.mean(pred_mb_test == y_test)#0.7749667994687915
#77.4%
pd.crosstab(pred_mb_test, y_test)
#Salary   <=50K   >50K
#row_0                
# <=50K   10891   2920
# >50K      469    780



############## Building Gaussian model #####################

classifier_gb = GB()
classifier_gb.fit(x_train, y_train)
pred_gb = classifier_gb.predict(x_train)
accuracy_gb_train = np.mean(pred_gb == y_train)# 0.7953317197705646
#79.5%
pd.crosstab(pred_gb,y_train)
#Salary   <=50K   >50K
#row_0                
# <=50K   21505   5025
# >50K     1148   2483

##for test data
pred_gb_test = classifier_gb.predict(x_test)
accuracy_gb_test = np.mean(pred_gb_test == y_test)#0.7946879150066402
#79.4%
pd.crosstab(pred_gb_test,y_test)
#Salary   <=50K   >50K
#row_0                
# <=50K   10759   2491
# >50K      601   1209



