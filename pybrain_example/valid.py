
"""
This is modified from zygmuntz's code from github.

https://github.com/zygmuntz/pybrain-practice/blob/master/kin_predict.py

modified by : alisonbwen@gmail.com
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer as BP

from pybrain.datasets.supervised import SupervisedDataSet
import pandas as pd
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
try:
    import cpickle as pickle
except:
    import pickle
net = pickle.load(open('net.pk', 'rb'))


test = pd.read_csv('../data/test.csv')

x_test = test.values[:, 0:-1]
y_test = np.array([[y] for y in test.values[:, -1]])
y_test_dumy = np.zeros(y_test.shape)

_, in_size = x_test.shape
_, out_size = y_test.shape

assert(net.indim == in_size)
assert(net.outdim == out_size)

ds = SupervisedDataSet(in_size, out_size)

ds.setField('input', x_test)
ds.setField('target', y_test_dumy)

p = net.activateOnDataset(ds)

mse = MSE(y_test, p)

rmse = sqrt(mse)

print "testing RMSE"
