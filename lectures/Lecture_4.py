# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Machine Learning for Health Metricians
# 
# ## Lecture 4: Evaluating what has been learned

# <markdowncell>

# ## Lecture 4 Outline
# 
# * Homework
# * Evaluating what has been learned
# * Exercise 4: Find the best $k$ for $k$-NN

# <markdowncell>

# ## Any questions?

# <markdowncell>

# ## Homework
# 
# * Project proposal (max 300 words)

# <markdowncell>

# # Homework
# 
# * Predicting cellphone ownership

# <codecell>

import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/RWA_DHS6_2010_2011_HH_ASSETS.CSV', index_col=0)
codebook = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/RWA_DHS6_2010_2011_HH_ASSETS_codebook.CSV', index_col=0)

# <codecell>

codebook

# <codecell>

import sklearn.neighbors

X = array(df.drop(['hv243a'], axis=1))
y = array(df.hv243a)

clf = sklearn.neighbors.KNeighborsClassifier()
clf.fit(X, y)

# <codecell>

y_pred = clf.predict(X)
print 'accuracy:', mean(y == y_pred)

# <markdowncell>

# ## Is this any good?

# <codecell>

import sklearn.tree
clf = sklearn.tree.DecisionTreeClassifier()
clf.fit(X, y)
y_pred = clf.predict(X)
print 'accuracy:', mean(y == y_pred)

# <markdowncell>

# ## Evaluating what has been learned

# <markdowncell>

# ## Training and testing
# 
# * It is important that the test data is _not used in any way_ to create the classifier.

# <markdowncell>

# ## How?
# 
# * stratified holdout cross-validation
# * (repeated) (stratified) $k$-fold cross-validation
# * leave-one-out cross-validation
# * bootstrap

# <codecell>

import sklearn.cross_validation

clf = sklearn.tree.DecisionTreeClassifier()
scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv=10)

print scores.mean(), scores.std()

# <codecell>

# stratified 75/25 holdout cross-validation
cv = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=10, test_size=.25)
scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv=cv)

print scores.mean(), scores.std()

# <codecell>

# part of a leave-one-out cross-validation
#cv = sklearn.cross_validation.LeaveOneOut(len(y))

cv = sklearn.cross_validation.ShuffleSplit(len(y), test_size=1, n_iter=100)
scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv=cv)

print scores.mean(), scores.std()

# <codecell>

# bootstrap cross-validation

cv = sklearn.cross_validation.Bootstrap(len(y), train_size=.5, n_iter=10)
scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv=cv)

print scores.mean(), scores.std()

# <markdowncell>

# ## Exercise 4: Find the best $k$ for $k$-NN predictions of cellphone ownership

# <codecell>

import sklearn.neighbors, sklearn.cross_validation

k = 10
X = array(df.drop(['hv243a'], axis=1))
y = array(df.hv243a)

clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
cv = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=10, test_size=.25)
scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv=cv)

print k, scores.mean(), scores.std()

# <markdowncell>

# # Homework
# 
# * Project proposal feedback ($+, \Delta$)
# * OOSPV of $k$-NN; find the best $k$.

# <codecell>

!cd /homes/abie/nbconvert/; cp /homes/abie/notebook/2013_04_21_ML4HM_Lecture_4.ipynb L4.ipynb; ./nbconvert.py --format reveal L4.ipynb

# <codecell>

!cp /homes/abie/nbconvert/L4* /home/j/Project/Machine-Learning/ML4HM/

# <codecell>

import ipynb_style
reload(ipynb_style)
ipynb_style.presentation()

# <codecell>


# <codecell>


