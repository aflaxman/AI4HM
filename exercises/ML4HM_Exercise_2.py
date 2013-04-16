# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Exercise 2: Reproducing ISL Figure 2.9 and 2.17

# <codecell>

import sklearn.datasets, sklearn.neighbors

# <codecell>

iris = sklearn.datasets.load_iris()
print iris.DESCR

# <codecell>

X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

# <codecell>

n_neighbors=15
weights='uniform'
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights)
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

# Create color maps
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# <codecell>

# Put the result into a color plot
figure(figsize=(8,5))

Z = Z.reshape(xx.shape)
pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
title("3-Class classification (k = %i, weights = '%s')"
             % (n_neighbors, weights))

axis('tight')
pass

# <markdowncell>

# ## Load the ISL data

# <codecell>

df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/ISL/Fig_2_9_data.csv', index_col=0)

# <codecell>

# fix the color names

df['color'] = df['Y'].map({'red': 'orange', 'blue': 'purple'})

# <codecell>

# make Figure 2.9

figure(figsize=(8,8))
for g, dfg in df.groupby('color'):
    plot(dfg['X.1'], dfg['X.2'], 'o', mec=g, color='none', mew=3, ms=8)

