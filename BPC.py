"""

ref:
    http://scikit-learn.org/stable/developers/index.html#rolling-your-own-estimator

author:
    alisonbwen@gmail.com

"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score

from scipy.stats import randint as sp_randint

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer as BP
from pybrain.datasets.supervised import SupervisedDataSet as SDS

from math import sqrt


class BPClassifier(BaseEstimator, ClassifierMixin):
    """

    for two class for now.

    """

    def __init__(self, h_size=2, epo=2, verbose=False):
        self.h_size = h_size
        self.epo = epo
        self.verbose = verbose
        pass

    def fit(self, X, y):

        y_train = np.array([[yn] for yn in y])
        _, self.in_size = X.shape
        _, self.out_size = y_train.shape

        ds = SDS(self.in_size, self.out_size)

        ds.setField('input', X)
        ds.setField('target', y_train)

        self.net = buildNetwork(self.in_size,
                                self.h_size, self.out_size, bias=True)
        trainer = BP(self.net, ds)

        print ("start training ...")

        #mse = trainer.train()
        #trainer.trainUntilConvergence(verbose=True, maxEpochs=4)

        for n in xrange(self.epo):
            mse = trainer.train()
            rmse = sqrt(mse)
            if self.verbose:
                print ("RMSE = %8.3f epoch = %d" % (rmse, n))
        return self

    def predict(self, X):
        p = self.predict_proba(X)
        p_class = np.array([1 if pn[0] > 0.5 else 0 for pn in p])

        return p_class

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

    h_size = 5
    epo = 3

    train = pd.read_csv('./data/train.csv')

    x_train = train.values[:, 0:-1]
    y_train = train.values[:, -1]

    bpc = BPClassifier(h_size=h_size, epo=epo)

    #test pipeline

    #anova_filter = SelectKBest(f_regression, k=2)
    #anova_bp = Pipeline([
        #('anava', anova_filter),
        #('bpc', bpc)
    #])

    #anova_bp.fit(x_train, y_train)

    #p = anova_bp.predict_proba(x_train)

    #p_c = anova_bp.predict(x_train)

    #y_c = np.array([1 if yn > 0.5 else 0 for yn in y_train])

    #accuracy = float(sum([1 for tf in p_c == y_c if tf]))/float(len(p_c))

    #print "accuracy is %8.4f" % accuracy

    # test gridsearch
    param_dist = {"h_size": sp_randint(2, 10)}
    clf = bpc
    n_iter_search = 2
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       verbose=1,
                                       scoring=precision_score
                                       )
	#scoring = getPred_Auc

    print("start fitting ....")
    random_search.fit(x_train,y_train)
    print("Best parameters set found on development set:")
    print()
    print(random_search.best_estimator_)
    print()
    print("scores on development set:")
    print()
    for params, mean_score, scores in random_search.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
