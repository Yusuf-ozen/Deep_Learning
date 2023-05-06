# -*- coding: utf-8 -*-
"""
Created on Fri May  5 19:10:10 2023

@author: yusuf
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer  ## missing values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


train_data = pd.read_csv("train.csv")
print(train_data.head())
print(train_data.columns)
print(train_data.info())
print(train_data.describe())
print(train_data.isnull().sum())  ## Age = 177, Cabin = 687, Embarked = 2 have missing values


## Missing values

# Age
Age_mean = SimpleImputer(missing_values = np.nan, strategy = "mean")
Age = train_data[["Age"]].values
Age_mean = Age_mean.fit(Age[:, :1])
Age[:, :1] = Age_mean.transform(Age[:, :1])

train_data[["Age"]] = Age

train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])



train_data = train_data.drop(["Cabin", "Name", "Ticket"], axis = 1)


## plot
sbn.catplot(x = "Sex", data = train_data, kind = "count").set(title = "Sex")
plt.show()

sbn.catplot(x = "Sex", hue = "Survived", data = train_data, kind = "count").set(title = "Sex and Survived")
plt.show()


sbn.catplot(x = "Embarked", hue = "Survived", data = train_data, kind = "count").set(title = "Embarked and Survived")
plt.show()


print(train_data.columns)
print(train_data.info())




## label encoding for embarked column
le = LabelEncoder()
embarked = train_data.iloc[:, -1]   ## (891,)
embarked = le.fit_transform(embarked)  ## s = 2, c = 0, q = 1
print(embarked.ndim)

## one hot encoder  for embarked column
one = OneHotEncoder()
embarked = embarked.reshape(-1, 1)
embarked = one.fit_transform(embarked).toarray()
print(embarked)

embarked_df = pd.DataFrame(data = embarked, index = range(891), columns = ["c", "q", "s"])


## label encoding for Sex column
le1 = LabelEncoder()
sex = train_data.iloc[:, 3]   ## (891,)
sex = le.fit_transform(sex)  ## 
print(sex.ndim)

## one hot encoder  for Sex column
one = OneHotEncoder()
sex = sex.reshape(-1, 1)
sex = one.fit_transform(sex).toarray()
print(sex)

sex_df = pd.DataFrame(data = sex, index = range(891), columns = ["male", "female"])

train_data = train_data.drop(["Sex", "Embarked"], axis = 1)
train_data = pd.concat([sex_df, embarked_df, train_data], axis = 1)


## correlation
print(train_data.corr()["Survived"].sort_values)

train_data.corr()["Survived"].sort_values().plot(kind = "bar")
## we should drop passenger id
train_data = train_data.drop(["PassengerId"], axis = 1)

"""
sbn.countplot(x = "Survived", data = train_data)
plt.show()
"""

## independent and dependent variables
y = train_data["Survived"].values   ##(891,)
print(y.shape)

x = train_data.drop(["Survived"], axis = 1).values
print(x.shape)


## test and train

x_train, x_test, y_train, y_true = train_test_split(x, y, test_size = 0.3, random_state = 10)

## scaling

scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)

## model
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

## confusion matrix and accuracy_score
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"conf Matrix : \n{conf_matrix}")
acc_score = accuracy_score(y_true, y_pred)
print(f"accuracy score : \n{acc_score * 100}")  ## 0.81

############ Logistic Regression model = 0.81  ##############
print("\n#####################################################################\n")

"""
for i in range(1, 10):
   
    model2 = KNeighborsClassifier(n_neighbors = i)
    model2.fit(x_train, y_train)
    print(f"for {i} score : {model2.score(x_test, y_true) * 100}")  >> for n= 4 has best score 
    
"""

model2 = KNeighborsClassifier(n_neighbors = 4, metric = "minkowski", p = 2)
model2.fit(x_train, y_train)

y_pred = model2.predict(x_test)

conf_matrix = confusion_matrix(y_true, y_pred)
print(f"conf Matrix : \n{conf_matrix}")
acc_score = accuracy_score(y_true, y_pred)
print(f"accuracy score : \n{acc_score * 100}")  ## 0.81

############ KNN model = 0.81  ##############
print("\n#####################################################################\n")

## Decision Tree
model3 = tree.DecisionTreeClassifier(criterion = 'log_loss', random_state = 0)
## “gini”, “entropy”, “log_loss”
model3.fit(x_train, y_train)

y_pred = model3.predict(x_test)
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"conf Matrix : \n{conf_matrix}")
acc_score = accuracy_score(y_true, y_pred)
print(f"accuracy score : \n{acc_score * 100}")  ## 0.787


############ Decision tree model = 0.78  ##############
print("\n#####################################################################\n")
## RandomForest
model4 = GaussianNB()
model4.fit(x_train, y_train)

y_pred = model4.predict(x_test)

conf_matrix = confusion_matrix(y_true, y_pred)
print(f"conf Matrix : \n{conf_matrix}")
acc_score = accuracy_score(y_true, y_pred)
print(f"accuracy score : \n{acc_score * 100}")  ## 0.794

############ Random Forest model = 0.794  ##############
print("\n#####################################################################\n")

## SVM model
model5 = SVC(kernel = "rbf", random_state = 123) # rbf=82.4, linear=80.5, sigmoid=67.9
## ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
## default = "rbf

model5.fit(x_train, y_train)

## prediction

y_pred = model5.predict(x_test)

conf_matrix = confusion_matrix(y_true, y_pred)
print(f"conf Matrix : \n{conf_matrix}")
acc_score = accuracy_score(y_true, y_pred)
print(f"accuracy score : \n{acc_score * 100}")  # 82.46

############ Random Forest model = 0.824  ##############
print("\n#####################################################################\n")

"""
SVM has a best score = 82.46
"""


