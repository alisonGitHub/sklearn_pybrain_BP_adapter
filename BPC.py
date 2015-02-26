"""

ref:
    http://scikit-learn.org/stable/developers/index.html#rolling-your-own-estimator

author:
    alisonbwen@gmail.com

"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BPClassifier(BaseEstimator, ClassifierMixin):
    """

    for twp class for now.

    """

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


if __name__ == '__main__':
    pass
