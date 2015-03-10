
"""

"""

import pandas as pd
from BPC import BPClassifier
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint



h_size = 8
epo = 3

train = pd.read_csv('./data/train.csv')

x_train = train.values[:, 0:-1]
y_train = train.values[:, -1]

y_train_01 = np.array([1 if yn > 0.5 else 0 for yn in y_train])

bpc = BPClassifier(h_size=h_size, epo=epo)

bpc.fit(x_train, y_train_01)

p = bpc.predict(x_train)

score = bpc.score(x_train, y_train_01)
print "score = ", score
print "starting grid search ... "
param_dist = {"h_size": sp_randint(2, 10)}

clf = bpc
n_iter_search = 2
random_search = RandomizedSearchCV(clf,
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   verbose=1
                                   )
                                       #,
                                       #scoring=accuracy_score
                                       #)

print("start fitting ....")
random_search.fit(x_train, y_train_01)
print("Best parameters set found on development set:")
print()
print(random_search.best_estimator_)
print()
print("scores on development set:")
print()
for params, mean_score, scores in random_search.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() / 2, params))

