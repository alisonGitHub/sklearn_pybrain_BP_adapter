"""

"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer as BP
from pybrain.datasets.supervised import SupervisedDataSet
import pandas as pd
from math import sqrt
import numpy as np

h_size = 50
epo = 400

train = pd.read_csv('../data/train.csv')

x_train = train.values[:, 0:-1]
y_train = np.array([[y] for y in train.values[:, -1]])

_, in_size = x_train.shape
_, out_size = y_train.shape

ds = SupervisedDataSet(in_size, out_size)

ds.setField('input', x_train)
ds.setField('target', y_train)

net = buildNetwork(in_size, h_size, out_size, hiddenclass=TanhLayer,
                   outclass=SoftmaxLayer, bias=True)
trainer = BP(net, ds)

print ("start training ...")

trainer.trainUntilConvergence(verbose=True, maxEpochs=400)
#for n in xrange(epo):
    #mse = trainer.train()
    #rmse = sqrt(mse)
    #print ("RMSE = %8.3f epoch = %d" % (rmse, n))

print "done"



