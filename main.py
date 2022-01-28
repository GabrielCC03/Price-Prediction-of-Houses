import numpy as np

import pandas as pd

import sklearn as sk

import matplotlib

import seaborn

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

url = "BostonHousing.csv"

data = pd.read_csv(url,nrows=508)

#print(data)

print(data.describe())
