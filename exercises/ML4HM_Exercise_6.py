# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## Exercise 6: SVM for the thing we used $k$-NN for before

# <codecell>

import numpy as np, pylab as pl, sklearn.svm as svm, pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/ISL/Fig_2_9_data.csv', index_col=0)

# <codecell>

X = array(df.filter(['X.1', 'X.2']))
y = array(df.Y.map({'blue': 0, 'red': 1}))

# <codecell>

scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Spectral,
        s=200, linewidth=0, alpha=.8)

# <markdowncell>

# # a (bad) start

# <codecell>

clf = svm.SVC(kernel='linear', C=1.).fit(X, y)

# <codecell>

# to see how bad...
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

pl.contour(xx, yy, Z, levels=[.5], linewidths=[3], colors='k')
pl.contourf(xx, yy, Z, cmap=pl.cm.Spectral_r)
scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Spectral,
        s=200, linewidth=1, alpha=.8)
axis('off');

# <markdowncell>

# ## but you know how to really see how bad, right?

# <codecell>


# <markdowncell>

# ## how does it compare to $k$-NN?

# <codecell>


