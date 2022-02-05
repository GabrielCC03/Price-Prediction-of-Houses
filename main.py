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
#Plots
"""
seaborn.boxplot(data=data)
#plt.show()

normalizedData = (data - data.mean())/data.std()

seaborn.heatmap(data=normalizedData.corr(),annot=True,cmap="coolwarm",vmin=-1)
#plt.show()

seaborn.kdeplot(data=data["MEDV"])
#plt.show()

seaborn.catplot(kind="bar", data=data)
#plt.show()

"""

#!-------------------------------------------

x = data[['age','tax','rm','lstat','dis','zn','indus']]
y = data.MEDV

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state = 5 )

#* Linear regression Prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
linearRegressionPredict = regressor.predict(x_test)
linearRegressionDF = pd.DataFrame({'Actual price': y_test, 'Predicted price': linearRegressionPredict})
linearRegressionDF.sort_index()

#* Normalized Data feature selection

xN_train, xN_test, yN_train, yN_test = train_test_split(x,y, test_size = 0.3,random_state = 5 )

#RandomForestRegressor
randomForest = RandomForestRegressor(max_depth=2, random_state=1 )
randomForest.fit(xN_train, yN_train)
randomForestPredict = randomForest.predict(xN_test)
randomForestDF = pd.DataFrame({'Actual price': yN_test, 'Predicted price': randomForestPredict})
randomForestDF.sort_index()

#* Support Vector Regressor  
svm = svm.SVR()
svm.fit(xN_train, yN_train)
svmPredict = svm.predict(xN_test)
svmDF = pd.DataFrame({'Actual price': yN_test, 'Predicted price': svmPredict})
svmDF.sort_index()

#print(data)
print("Linear Regression: ")
print(linearRegressionDF)
print("Random Forest Regressor: ")
print(randomForestDF)
print("Support Vector Machine Regressor: ")
print(svmDF)

#* Analyzing data
#? Mean Absolute error
print("Mean sAbsolute Error: ")
print("Linear Regression:",metrics.mean_absolute_error(y_test,linearRegressionPredict))
print("Random Forest:",metrics.mean_absolute_error(yN_test,randomForestPredict))
print("Support Vector Machine:",metrics.mean_absolute_error(yN_test,svmPredict))

#? Mean squared error
print("\nMean Squared Error: ")
print("Linear Regression:",metrics.mean_squared_error(y_test,linearRegressionPredict))
print("Random Forest:",metrics.mean_squared_error(yN_test,randomForestPredict))
print("Support Vector Machine:",metrics.mean_squared_error(yN_test,svmPredict))

#? R2 Score
print("\nR2 Score: ")
print("Linear Regression:",metrics.r2_score(y_test,linearRegressionPredict))
print("Random Forest:",metrics.r2_score(yN_test,randomForestPredict))
print("Support Vector Machine:",metrics.r2_score(yN_test,svmPredict))

#? Plots
x1 = np.array(x_test["tax"])
y1 = np.array(linearRegressionPredict)
m,b= np.polyfit(x1,y1,1)

#* Plots of Results:
plt.plot(x_test["tax"], m*x_test["tax"] + b)
plt.plot(x_test["tax"],linearRegressionPredict,'o')
plt.xlabel("Tax")
plt.ylabel("Median Value")
plt.show()

seaborn.pairplot(data,x_vars=['age','tax','rm','lstat','dis','zn','indus'],y_vars="MEDV",kind="reg")
plt.show()


#data.info()
#print(data.describe())
