import matplotlib.pyplot as plt
import csv
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = read_csv('voice.csv')
data_x = data.iloc[:, :data.shape[1] - 1]
data_y = data.iloc[:, data.shape[1] - 1:data.shape[1]]
data_x = data_x.apply(lambda x: np.log(x + 1))
scaler = MinMaxScaler(feature_range=(0, 1))

data_x = scaler.fit_transform(data_x)
cols = data_x.columns

for col in cols:
    values = data_x[col]
    plt.clf()
    male = []
    female = []
    for x in range(1, 1585):
        male.append(values[x])
    plt.plot(male, 'ro', label='Male')
    for x in range(1585, 3168):
        female.append(values[x])
    plt.plot(female, 'bo', label='Female')
    plt.legend()
    plt.ylabel(col)
    plt.xlabel('samples')
    plt.show()

data_x = data_x.apply(lambda x: np.log(x + 1))
scaler = MinMaxScaler(feature_range=(0, 1))
data_x = scaler.fit_transform(data_x)
for i in xrange(len(cols)):
    values = data_x[:, i]
    col = cols[i]
    plt.clf()
    male = []
    female = []
    for x in range(0, 1585):
        male.append(values[x])
    plt.plot(male, 'ro', label='Male')
    for x in range(1585, 3168):
        female.append(values[x])
    plt.plot(female, 'bo', label='Female')
    plt.legend()
    plt.ylabel('normal_' + col)
    plt.xlabel('samples')
    plt.show()
