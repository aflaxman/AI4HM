# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Exercise 3: Applying many ML methods

# <codecell>

# what to apply them to?  perhaps my simulated data would be good.

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

# <codecell>

data = simulate(n=1000, p=10, seed=0)

# <markdowncell>

# # Methods
# 
# ## OneR / Hyperpipes

# <codecell>

import sklearn.tree
clf = sklearn.tree.DecisionTreeClassifier(max_depth=1)
clf.fit(data['X'], data['y'])

# <codecell>

y_pred = clf.predict(data['X'])
print 'in-sample accuracy:', mean(y_pred == data['y'])

# <markdowncell>

# ## Naive Bayes

# <markdowncell>

# ## Additional Methods

# <markdowncell>

# # Application to RWA DHS data

# <codecell>

import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/RWA_DHS6_2010_2011_HH_ASSETS.CSV')
codebook = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/RWA_DHS6_2010_2011_HH_ASSETS_codebook.CSV')

# <codecell>


