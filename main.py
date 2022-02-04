import numpy as np

import pandas as pd

import sklearn as sk

import matplotlib

import seaborn

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt


url = "BostonHousing.csv"

data = pd.read_csv(url,nrows=508)
target = data.values[: , 2] #[1::2, 2]

data["MEDV"] = target

seaborn.boxplot(data=data)
plt.show()

normalizedData = (data - data.mean())/data.std()

seaborn.heatmap(data=normalizedData.corr(),annot=True,cmap="coolwarm",vmin=-1)
plt.show()

seaborn.kdeplot(data=data["MEDV"])
plt.show()

seaborn.catplot(kind="bar", data=data)
plt.show()

#print(data)
#data.info()

#print(data.describe())
