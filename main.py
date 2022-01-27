import numpy as np

import pandas as pd

import sklearn as sk

import matplotlib

import seaborn

from sklearn.datasets import load_boston

data = load_boston()

url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(url, sep="s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

print(raw_df)