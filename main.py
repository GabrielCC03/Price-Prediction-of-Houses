import numpy as np

import pandas as pd

import sklearn as sk

import matplotlib

import seaborn

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt


url = "BostonHousing.csv"

data = pd.read_csv(url,nrows=508)
target = data.values[:, 2] #[1::2, 2]

data["MEDV"] = target


print(data)


#print(data.describe())
