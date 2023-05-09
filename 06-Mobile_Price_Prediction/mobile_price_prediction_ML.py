# -*- coding: utf-8 -*-
"""
Created on Tue May  9 19:52:40 2023

@author: yusuf
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression # for model1
from sklearn.neighbors import KNeighborsClassifier  # for model2
from sklearn import tree   # for model3
from sklearn.naive_bayes import GaussianNB  # for model4
from sklearn.svm import SVC   # for model5
from sklearn.metrics import confusion_matrix, accuracy_score

## SVM has best score = 94.16 for train_data


train_data = pd.read_csv("train.csv")

print(train_data.head())
print(train_data.info())
print(train_data.isnull().sum())
print(train_data.columns)


## plot with seaborn
sbn.countplot(x = "price_range", data = train_data)
plt.show()

## corr()
print(train_data.corr()["price_range"].sort_values)
train_data.corr()["price_range"].sort_values().plot(kind = "bar")

## independent and dependent variables
y = train_data["price_range"].values
x = train_data.drop(["price_range"], axis = 1).values

## train and test
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size = 0.3, random_state = 10)

## scaling 
scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)


print("\n#####################################################################\n")

## model1 >>> KNN

"""
for i in range(1, 50):
   
    model1 = KNeighborsClassifier(n_neighbors = i)
    model1.fit(x_train, y_train)
    print(f"for {i} score : {model1.score(x_test, y_true) * 100}")  # for n = 39 has best score 
    ## but score = 52.5
"""

model1 = KNeighborsClassifier(n_neighbors = 39, metric = "minkowski")
model1.fit(x_train, y_train)

y_pred1 = model1.predict(x_test)

conf1 = confusion_matrix(y_true, y_pred1)
acc1 = accuracy_score(y_true, y_pred1)
print("KNN\n")
print(f"conf Matrix 1: \n{conf1}")
print(f"accuracy score 1: \n{acc1 * 100}")  ## 52.5

print("\n#####################################################################\n")

## model3 >>> Decision Tree

model2 = tree.DecisionTreeClassifier(criterion = 'log_loss', random_state = 0)
## “gini”, “entropy”, “log_loss”
model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test)
conf2 = confusion_matrix(y_true, y_pred2)
print("Decision Tree\n")
print(f"conf Matrix 2: \n{conf2}")
acc2 = accuracy_score(y_true, y_pred2)
print(f"accuracy score 2: \n{acc2 * 100}")  ## 83.16

print("\n#####################################################################\n")

## model3 >>> RandomForest

model3 = GaussianNB()
model3.fit(x_train, y_train)

y_pred3 = model3.predict(x_test)

conf3 = confusion_matrix(y_true, y_pred3)
acc3 = accuracy_score(y_true, y_pred3)
print("Random Forest\n")
print(f"conf Matrix 3: \n{conf3}")
print(f"accuracy score 3: \n{acc3 * 100}")  ## 81.83

print("\n#####################################################################\n")

## model4 >>> SVM 

model4 = SVC(kernel = "linear", random_state = 123) 
## ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
## default = "rbf

model4.fit(x_train, y_train)

y_pred4 = model4.predict(x_test)

conf4 = confusion_matrix(y_true, y_pred4)
acc4 = accuracy_score(y_true, y_pred4)
print("SVM\n")
print(f"conf Matrix 4: \n{conf4}")
print(f"accuracy score 4: \n{acc4 * 100}")  # 94.16


## test

test_data = pd.read_csv("test.csv")
print(train_data.head())
print(train_data.info())
print(train_data.isnull().sum())

test_id = test_data["id"].values
test_data = test_data.drop(["id"], axis = 1)

test_data = scale.fit_transform(test_data)
test_predict = model4.predict(test_data)


## neden id'nin 503'ten başladığını bulamadım.
for i in range(1, len(test_data)):
    print(f"id : {test_id[i]} >> predict : {test_predict[i]}")
    #print(test_predict[i])
    
