# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Machine Learning for Health Metricians
# 
# ## Lecture 3: Basic Algorithms

# <markdowncell>

# ## Lecture 3 Outline:
#     
# * Homework
# * Exercise 2: In-sample and Out-of-sample Predictive Validitiy
# * Basic Algorithms
# * Exercise 3: Applying basic algorithms

# <markdowncell>

# # Any questions?

# <markdowncell>

# ## Homework
# 
# * Elevator pitch + Minimum viable hypothesis + Test dataset

# <markdowncell>

# # Exercise 2 / Homework
# 
# * ISL Figures 2.15 and 2.17 

# <codecell>

import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/ISL/Fig_2_9_data.csv', index_col=0)

# fix the color names
df['color'] = df['Y'].map({'red': 'orange', 'blue': 'purple'})

# <codecell>

# make Figure 2.9

figure(figsize=(8,8))
for g, dfg in df.groupby('color'):
    plot(dfg['X.1'], dfg['X.2'], 'o', mec=g, color='none', mew=3, ms=8)

# <codecell>

import sklearn.neighbors

n_neighbors=10
weights='uniform'
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights)

X = array(df.filter(like='X'))
y = array(df.color.map({'orange':0.0, 'purple':1.0}))
clf.fit(X, y)

# <codecell>

# explore the decision boundary. For that, we will predict each
# point in the mesh [x_min, m_max]x[y_min, y_max].
h = .02

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# <codecell>

figure(figsize=(8,8))
# show k-NN decision boundary
Z = Z.reshape(xx.shape)
contour(xx, yy, Z, levels=[.5], colors=['k'], linewidths=[2])

# also the training points
for g, dfg in df.groupby('color'):
    plot(dfg['X.1'], dfg['X.2'], 'o', mec=g, color='none', mew=3, ms=8, zorder=-10)
title("KNN (k = %i, weights = '%s')" % (n_neighbors, weights))
axis('tight');

# <codecell>

# also get the bayesian decision boundary, if possible
blue_mean = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/ISL/Fig_2_9_blue_mean.csv', index_col=0)
orange_mean = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/ISL/Fig_2_9_orange_mean.csv', index_col=0)

def pr_orange(y):
    like_blue = [mc.normal_like(y, X_j, .5**-2) for j, X_j in blue_mean.iterrows()]
    like_orange = [mc.normal_like(y, X_j, .5**-2) for j, X_j in orange_mean.iterrows()]
    
    return exp(mc.flib.logsum(like_orange) - mc.flib.logsum(like_blue + like_orange))
pr_orange([0,0])

# <codecell>

h = .1
xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z2 = pd.DataFrame([xx2.ravel(), yy2.ravel()]).apply(pr_orange)

# <codecell>

figure(figsize=(8,8))
# show Bayes decision boundary
Z2 = Z2.reshape(xx2.shape)
contour(xx2, yy2, Z2, levels=[.5], colors='purple', linewidths=[2], linestyles='dashed')

# also the training points
for g, dfg in df.groupby('color'):
    plot(dfg['X.1'], dfg['X.2'], 'o', mec=g, color='none', mew=3, ms=8, zorder=-10)
title("KNN (k = %i, weights = '%s')" % (n_neighbors, weights))
axis('tight');

# <codecell>

figure(figsize=(8,8))
contour(xx, yy, Z, levels=[.5], colors='black', linewidths=[2], linestyles='solid')
contour(xx2, yy2, Z2, levels=[.5], colors='purple', linewidths=[2], linestyles='dashed')
# show the training points
for g, dfg in df.groupby('color'):
    plot(dfg['X.1'], dfg['X.2'], 'o', mec=g, color='none', mew=3, ms=8, zorder=-10)
title("KNN (k = %i, weights = '%s')" % (n_neighbors, weights))
axis('tight');

# <codecell>

figure(figsize=(8,8))
# color in background
cmap_light = mpl.colors.ListedColormap(['#FF7F50', '#FFCCCC'])
pcolormesh(xx, yy, Z, cmap=cmap_light)
# put that together with the k-NN decision boundary
contour(xx, yy, Z, levels=[.5], colors='black', linewidths=[2], linestyles='solid')
contour(xx2, yy2, Z2, levels=[.5], colors='purple', linewidths=[2], linestyles='dashed')
# show the training points
for g, dfg in df.groupby('color'):
    plot(dfg['X.1'], dfg['X.2'], 'o', mec=g, color='none', mew=3, ms=8, zorder=2)
title("KNN (k = %i, weights = '%s')" % (n_neighbors, weights))
axis('tight');

# <markdowncell>

# ## Basic Algorithms (in Python/scikits-learn)
# 
# ### Try it as we go: in the Exercise 3 Notebook
# 
# http://dismod.ihme.washington.edu:3000/ if you don't have your own yet.

# <codecell>

# simulate some data so we can see the algorithms work

import random, pymc as mc
def simulate(n, p, seed):
    random.seed(123456+seed)
    mc.np.random.seed(123456+seed)
    
    # make A clusters, beta distributed
    A = 5
    X_true = mc.rbeta(.5, .5, size=(A,p))
    y_true = mc.rbeta(1, 1, size=A)
    
    
    X = zeros((n,p))
    p_true = zeros(n)
    for i in range(n):
        a_i = random.randrange(A)
        X[i] = mc.rbernoulli(X_true[a_i])
        p_true[i] = y_true[a_i]
    
    y = mc.rbinomial(1, p_true)
    
    test = random.sample(range(n), n/4)
    train = list(set(range(n)) - set(test))
    
    X_train = X[train]
    y_train = y[train]
    
    X_test = X[test]
    y_test = y[test]

    return locals()
data = simulate(n=1000, p=10, seed=0)

# <markdowncell>

# ## OneR / HyperPipes

# <codecell>

import sklearn.tree
clf = sklearn.tree.DecisionTreeClassifier(max_depth=1)
clf.fit(data['X'], data['y'])

# <markdowncell>

# ## Naive Bayes
# 
# Statisticians don't believe that this is even Bayesian...

# <codecell>

import sklearn.naive_bayes
clf = sklearn.naive_bayes.BernoulliNB()
clf.fit(data['X_train'], data['y_train'])

# <markdowncell>

# ## Decision trees
# 
# Here is where we are getting into dangerous territory...

# <codecell>

import sklearn.tree
clf = sklearn.tree.DecisionTreeClassifier()
clf.fit(data['X'], data['y'])

# <markdowncell>

# ## Decision rules (Prism)
# 
# This one is not in sklearn.  Adding it would be a very instructive project...

# <markdowncell>

# ## Association rules (a priori)
# 
# Also not in sklearn.  What's up with that?

# <markdowncell>

# # Regression Models

# <markdowncell>

# ## Linear regression in sklearn

# <codecell>

import sklearn.linear_model
clf = sklearn.linear_model.LinearRegression()
clf.fit(data['X'], data['y'])

# <markdowncell>

# ## Logistic regression

# <codecell>

import sklearn.linear_model
clf = sklearn.linear_model.LogisticRegression()
clf.fit(data['X'], data['y'])

# <markdowncell>

# ## Perceptron

# <codecell>

import sklearn.linear_model
clf = sklearn.linear_model.Perceptron()
clf.fit(data['X'], data['y'])

# <markdowncell>

# ## Winnow
# 
# Not available in sklearn

# <markdowncell>

# # Instance-based classifiers

# <markdowncell>

# ## $k$-NN

# <codecell>

import sklearn.neighbors
clf = sklearn.neighbors.KNeighborsClassifier()
clf.fit(data['X'], data['y'])

# <markdowncell>

# ## $k$-Means clustering

# <codecell>

import sklearn.cluster
clf = sklearn.cluster.KMeans()
clf.fit(data['X'], data['y'])

# <markdowncell>

# # Exercise 3

# <markdowncell>

# ## Exercise 3: Predicting Cell Phone Ownership
# 
# URL: http://dismod.ihme.washington.edu:3000/

# <codecell>

import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/RWA_DHS6_2010_2011_HH_ASSETS.CSV')
codebook = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/RWA_DHS6_2010_2011_HH_ASSETS_codebook.CSV')

# <markdowncell>

# ## Homework
# 
# * Project proposal write up (maximum 300 words)
#  * Describe project (elevator pitch)
#  * State hypothesis (minimum viable)
#  * Describe dataset (table)
# * Continue predicting cell phone ownership

# <codecell>

!cd /homes/abie/nbconvert/; cp /homes/abie/notebook/2013_04_14_ML4HM_Lecture_3.ipynb L3.ipynb; ./nbconvert.py --format reveal L3.ipynb

# <codecell>

!cp /homes/abie/nbconvert/L3* /home/j/Project/Machine-Learning/ML4HM/

# <codecell>

import ipynb_style
reload(ipynb_style)
ipynb_style.presentation()

# <codecell>


