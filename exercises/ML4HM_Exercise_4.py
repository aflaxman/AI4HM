# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## Exercise 4: Find the best $k$ for $k$-NN predictions of cellphone ownership

# <codecell>

import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/RWA_DHS6_2010_2011_HH_ASSETS.CSV', index_col=0)
codebook = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/RWA_DHS6_2010_2011_HH_ASSETS_codebook.CSV', index_col=0)

# <codecell>

X = array(df.drop(['hv243a'], axis=1))
y = array(df.hv243a)

# <codecell>

import sklearn.neighbors, sklearn.cross_validation

k = 10
X = array(df.drop(['hv243a'], axis=1))
y = array(df.hv243a)

clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
cv = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=10, test_size=.25)
scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv=cv)

print k, scores.mean(), scores.std()

