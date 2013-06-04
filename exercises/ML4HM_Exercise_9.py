# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0, shuffle=False)

# <markdowncell>

# Univariate selection

# <codecell>

from sklearn.feature_selection import SelectPercentile, f_classif, chi2

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()

# <codecell>

# plot the scores

# <codecell>

X_transform = selector.fit_transform(X, y)

# <markdowncell>

# Recursive feature elimination

# <codecell>

from sklearn.svm import SVC
from sklearn.feature_selection import RFE

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)

# <codecell>

rfe.ranking_

# <markdowncell>

# To find the best, use cross validation

# <codecell>

from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import zero_one_loss

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
              loss_func=zero_one_loss)
rfecv.fit(X, y)

# <codecell>

rfecv.cv_scores_

# <markdowncell>

# We didn't quite get to ensemble methods. Too bad.  Here is something you can do with them, though.  I can't resist throwing it in.

# <codecell>

from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

# <codecell>

bar(range(len(importances)), importances, yerr=std, color='grey')

# <markdowncell>

# How do you think you discretize an array in Python?

# <codecell>

n = 32
p = 4
X = rand(n,p)

# <markdowncell>

# PCA

# <codecell>

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_transform = pca.fit_transform(X)

# <markdowncell>

# Random projection

# <codecell>

>>> import numpy as np
>>> from sklearn import random_projection

>>> X = np.random.rand(100, 10000)
>>> transformer = random_projection.GaussianRandomProjection()
>>> X_new = transformer.fit_transform(X)
>>> X_new.shape

