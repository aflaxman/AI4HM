# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd, sklearn.cluster as cluster
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/ISL/Fig_2_9_data.csv', index_col=0)
X = array(df.filter(like='X'))

# <codecell>

plot(X[:,0], X[:,1], 'o', color='none', mec='k', mew=2, ms=7, alpha=.95)

# <markdowncell>

# ## k-Means

# <codecell>

labels = cluster.KMeans(n_clusters=2).fit_predict(X)

# <codecell>

for l in unique(label):
    plot(X[label==l, 0], X[label==l, 1], 'o',
         color='none', mec='kr'[l], mew=2, ms=7, alpha=.95)

# <markdowncell>

# ## Other Methods

# <codecell>

clf = cluster.Ward(n_clusters=10)
label = clf.fit_predict(X)

# <codecell>

import sklearn.mixture as mixture
clf = mixture.GMM(n_components=4)
clf.fit(X)
label = clf.predict_proba(X)

# <codecell>

cluster.AffinityPropagation

# <codecell>

cluster.SpectralClustering

# <markdowncell>

# ## Exercise

# <codecell>

import json
posts = json.load(file('/home/j/Project/Machine-Learning/ML4HM/blog_posts.json'))

# <codecell>

posts[0][:300]

# <markdowncell>

# * http://scikit-learn.github.io/scikit-learn-tutorial/working_with_text_data.html

# <codecell>

>>> from sklearn.feature_extraction.text import CountVectorizer
>>> count_vect = CountVectorizer()
>>> X_train_counts = count_vect.fit_transform(posts)
>>> X_train_counts.shape

