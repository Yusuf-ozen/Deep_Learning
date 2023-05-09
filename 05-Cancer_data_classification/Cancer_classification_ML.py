# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:38:52 2023

@author: yusuf
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
## MinMaxScaler, verilerin sınırlı bir aralığa sahip olduğu veya dağılımın Gauss olmadığı durumlarda kullanışlıdır
## StandardScaler, veriler bir Gauss dağılımına sahip olduğunda veya algoritma standartlaştırılmış özellikler gerektirdiğinde kullanışlıdır.

from sklearn.linear_model import LogisticRegression # for model1
from sklearn.neighbors import KNeighborsClassifier  # for model2
from sklearn import tree   # for model3
from sklearn.naive_bayes import GaussianNB  # for model4
from sklearn.svm import SVC   # for model5
from sklearn.metrics import confusion_matrix, accuracy_score

## KNN has best score = 97.66

data = pd.read_csv("Cancer_Data.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())


data = data.drop(["Unnamed: 32"], axis = 1)
print(data.columns)

## plot with seaborn
sbn.countplot(x = "diagnosis", data = data)
plt.show()

## label encoding
le = LabelEncoder()
diagnosis = data.iloc[:, 1]
diagnosis = le.fit_transform(diagnosis) # m = 1, b = 0
print(diagnosis.shape)
print(diagnosis[1:5])
print(diagnosis[-5:])

## drop original diagnosis column and add new column
data = data.drop(["diagnosis"], axis = 1)
diagnosis_df = pd.DataFrame(data = diagnosis, index = range(569), columns = ["Diagnosis"])

data = pd.concat([diagnosis_df, data], axis = 1)

## corr()

print(data.corr()["Diagnosis"].sort_values)

#data = data.drop(["id", "fractal_dimension_mean", "texture_se", "smoothness_se", "symmetry_se", "fractal_dimension_se"])              
## once derrin ogrenmede yaptıgım sutunları kullanıcam en son bu sutunları atıp deneyeceğim. 

data = data.drop(["id"], axis = 1)
print(data.head())

data.corr()["Diagnosis"].sort_values().plot(kind = "bar")

## X and Y
y = data["Diagnosis"].values
x = data.drop(["Diagnosis"], axis = 1).values

## train and test
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size = 0.3, random_state = 10)

## scaling 
scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)

print("\n#####################################################################\n")

## model1 >>> LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)

y_pred = model1.predict(x_test)

conf1 = confusion_matrix(y_true, y_pred)
acc1 = accuracy_score(y_true, y_pred)
print(f"conf Matrix 1: \n{conf1}")
print(f"accuracy score 1: \n{acc1 * 100}")  ## 93.5

print("\n#####################################################################\n")

## model2 >>> KNN

"""
for i in range(1, 10):
   
    model2 = KNeighborsClassifier(n_neighbors = i)
    model2.fit(x_train, y_train)
    print(f"for {i} score : {model2.score(x_test, y_true) * 100}")  #>> for n = 8 has best score 

"""
model2 = KNeighborsClassifier(n_neighbors = 8, metric = "minkowski")
model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test)

conf2 = confusion_matrix(y_true, y_pred2)
acc2 = accuracy_score(y_true, y_pred2)
print(f"conf Matrix 2: \n{conf2}")
print(f"accuracy score 2: \n{acc2 * 100}")  ## 97.66

print("\n#####################################################################\n")

## model3 >>> Decision Tree

model3 = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 0)
## “gini”, “entropy”, “log_loss”
model3.fit(x_train, y_train)

y_pred3 = model3.predict(x_test)
conf3= confusion_matrix(y_true, y_pred3)
print(f"conf Matrix 3: \n{conf3}")
acc3 = accuracy_score(y_true, y_pred3)
print(f"accuracy score 3: \n{acc3 * 100}")  ## 86.54

print("\n#####################################################################\n")

## model4 >>> RandomForest

model4 = GaussianNB()
model4.fit(x_train, y_train)

y_pred4 = model4.predict(x_test)

conf4 = confusion_matrix(y_true, y_pred4)
acc4 = accuracy_score(y_true, y_pred4)
print(f"conf Matrix 4: \n{conf4}")
print(f"accuracy score 4: \n{acc4 * 100}")  ## 88.88

print("\n#####################################################################\n")

## model5 >>> SVM 

model5 = SVC(kernel = "rbf", random_state = 123) 
## ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
## default = "rbf

model5.fit(x_train, y_train)

y_pred5 = model5.predict(x_test)

conf5 = confusion_matrix(y_true, y_pred5)
acc5 = accuracy_score(y_true, y_pred5)
print(f"conf Matrix 5: \n{conf5}")
print(f"accuracy score 5: \n{acc5 * 100}")  # 90.64
