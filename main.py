import numpy as np

import pandas as pd

import sklearn as sk

import matplotlib

import seaborn

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm

from sklearn import metrics


url = "BostonHousing.csv"
data = pd.read_csv(url,nrows=508)
target = data.values[: , 2] #[1::2, 2]

data["MEDV"] = target

#!_----------------------------------------------------------------
seaborn.boxplot(data=data)
#plt.show()

normalizedData = (data - data.mean())/data.std()

seaborn.heatmap(data=normalizedData.corr(),annot=True,cmap="coolwarm",vmin=-1)
#plt.show()

seaborn.kdeplot(data=data["MEDV"])
#plt.show()

seaborn.catplot(kind="bar", data=data)
#plt.show()

#!-------------------------------------------

x = data[['age','tax','rm','lstat','dis','zn','indus']]
y = data.MEDV

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state = 5 )

#Linear regression Prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
predictLR = pd.DataFrame({'Actual price': y_test, 'Predicted price': y_pred})
predictLR.sort_index()

#* Normalized Data feature selection
xN = data[['age','tax','rm','lstat','dis','zn','indus']]
yN = data.MEDV

xN_train, xN_test, yN_train, yN_test = train_test_split(xN,yN, test_size = 0.3,random_state = 5 )

#RandomForestRegressor
rfr = RandomForestRegressor(max_depth=2, random_state=0)
rfr.fit(xN_train, yN_train)
rfrPredict = rfr.predict(xN_test)
predictRfr = pd.DataFrame({'Actual price': yN_test, 'Predicted price': rfrPredict})
predictRfr.sort_index()

#Support Vector Regressor  
svm = svm.SVR()
svm.fit(xN_train, yN_train)
svmPredict = svm.predict(xN_test)
predictSVM = pd.DataFrame({'Actual price': yN_test, 'Predicted price': svmPredict})

#print(data)
print("Linear Regression: ")
print(predictLR)
print("Random Forest Regressor: ")
print(predictRfr)
print("Support Vector Machine Regressor: ")
print(predictSVM)



#data.info()
#print(data.describe())
