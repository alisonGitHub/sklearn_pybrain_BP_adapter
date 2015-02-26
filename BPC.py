"""

ref:
    http://scikit-learn.org/stable/developers/index.html#rolling-your-own-estimator

author:
    alisonbwen@gmail.com

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer as BP
from pybrain.datasets.supervised import SupervisedDataSet as SDS

from math import sqrt


class BPClassifier(BaseEstimator, ClassifierMixin):
    """

    for two class for now.

    """

    def __init__(self, h_size=2, epo=2):
        self.h_size = h_size
        self.epo = epo
        pass

    def fit(self, X, y):
        _, self.in_size = X.shape
        _, self.out_size = y.shape

        ds = SDS(self.in_size, self.out_size)

        ds.setField('input', X)
        ds.setField('target', y)

        self.net = buildNetwork(self.in_size,
                                self.h_size, self.out_size, bias=True)
        trainer = BP(self.net, ds)

        print ("start training ...")

        #mse = trainer.train()
        #trainer.trainUntilConvergence(verbose=True, maxEpochs=4)

        for n in xrange(self.epo):
            mse = trainer.train()
            rmse = sqrt(mse)
            print ("RMSE = %8.3f epoch = %d" % (rmse, n))
        return self

    def predict(self, X):
        pass

    def predict_proba(self, X):

        row_size, in_size = X.shape

        y_test_dumy = np.zeros([row_size, self.out_size])

        assert(self.net.indim == in_size)

        ds = SDS(in_size, self.out_size)

        ds.setField('input', X)
        ds.setField('target', y_test_dumy)

        p = self.net.activateOnDataset(ds)
        return p


if __name__ == '__main__':
    pass
    h_size = 5
    epo = 100

    train = pd.read_csv('./data/train.csv')

    x_train = train.values[:, 0:-1]
    y_train = np.array([[y] for y in train.values[:, -1]])

    bpc = BPClassifier(h_size=h_size, epo=epo)

    bpc.fit(x_train, y_train)

    p = bpc.predict_proba(x_train)

