"""

"""

import pandas as pd
from BPC import BPClassifier
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from sklearn.pipeline import Pipeline


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

#test pipeline

anova_filter = SelectKBest(f_regression, k=2)
anova_bp = Pipeline([
    ('anava', anova_filter),
    ('bpc', bpc)
    ])

anova_bp.fit(x_train, y_train)

p = anova_bp.predict_proba(x_train)

p_c = anova_bp.predict(x_train)

y_c = np.array([1 if yn > 0.5 else 0 for yn in y_train])

accuracy = float(sum([1 for tf in p_c == y_c if tf]))/float(len(p_c))

print "accuracy is %8.4f" % accuracy

    # test gridsearch
