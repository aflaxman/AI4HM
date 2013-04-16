# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Machine Learning for Health Metricians
# 
# ## Lecture 2: Input, Output, and Accuracy

# <markdowncell>

# ## Lecture 2 Outline:
#     
# * Homework Solutions
# * Input
# * Output
# * Accuracy
# * Exercise 2: In-sample and Out-of-sample Predictive Validitiy

# <markdowncell>

# # Any questions?

# <markdowncell>

# Anticipated questions: based on last class: what is machine learning, how does it differ from statistics; based on exercise/homework: what is a decision tree, how do I ____ in python; based on reading: what is a feature, what out-of-sample pv, what is this obsession with rule-based learning

# <markdowncell>

# ## Homework Solutions
# 
# * Length-two decision list
# * Elevator pitch

# <markdowncell>

# The length-two decision list may be too complex to go over in class, but may not be.  It was probably too hard for the first homework, considering the target time of 2 hours outside of class also includes reading and the elevator pitch.
# 
# Going over the elevator pitch is a bit of a challenge, because it will take too long to do everyone.  I plan to ask if anyone is really happy with theirs and let them do it, and then ask if anyone is really stuck with theirs and get the group to workshop it a little bit.

# <markdowncell>

# ## Input

# <codecell>

import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/college.csv', index_col=0)
df.head()

# <markdowncell>

# ## What is a concept?

# <markdowncell>

# The thing you are learning.  DM categorizes machine learning tasks into four categories: classification, association, clustering, and "numeric prediction" aka regression.

# <markdowncell>

# ## What is an example?

# <markdowncell>

# What we are learning from.  This is also called an "instance", or a feature vector.  It is a row in the input dataset.

# <markdowncell>

# ## What is an attribute?

# <markdowncell>

# Also called a "feature", this is a column in the input dataset.

# <markdowncell>

# ## Last word on input
# 
# > Preparing input for a data mining investigation usually consumes the bulk of the effort invested in the entire data mining process.

# <codecell>

import IPython.display

# <markdowncell>

# # Video Interlude

# <codecell>

IPython.display.YouTubeVideo("4TBcQ8h_kXU")

# <markdowncell>

# ## Output

# <markdowncell>

# ## Tables

# <markdowncell>

# ## Linear Models
# 
# * $$\mathrm{PRP} = 37.06 + 2.47\cdot\mathrm{CACH}$$
# * $$2.0 - 0.5 \cdot \mathrm{PETAL-LENGTH} - 0.8 \cdot\mathrm{PETAL-WIDTH} = 0$$

# <markdowncell>

# ## Trees
# 
# Picture of Tree TK

# <markdowncell>

# ## Classification Rules
# 
#     if a and b then x
#     if c and d then x

# <markdowncell>

# ## Association Rules
# 
#     if windy = false and play = no then
#         outlook = sunny and humidity = high

# <markdowncell>

# ## Rules with Exceptions

# <markdowncell>

# The authors of *Data Mining* show their CS/AI background by their focus on rule-based knowledge representation and especially on this rules-with-exceptions section, since these are almost never learned empirical from data (in my experience).  This comes from the pre-statistical learning days of AI, when people thought AI would emerge from putting together enough true facts and the appropriate logic-based inference rules.

# <markdowncell>

# ## Instance-based Representations

# <markdowncell>

# Despite to luke-warm attitude of the authors of *DM*, this is one of my favoriate approaches, and we will explore it with the $k$-nearest neighbor algorithm today.

# <markdowncell>

# ## Clusters

# <markdowncell>

# Unsupervised learning is often much more of a challenge that supervised learning, and cluster are often what comes out.  The question is "are they any good"?

# <markdowncell>

# ## Accuracy

# <codecell>

IPython.display.Image(filename='/home/j/Project/Machine-Learning/ML4HM/ISL_Fig_2_9.png', embed=True)

# <codecell>

IPython.display.Image(filename='/home/j/Project/Machine-Learning/ML4HM/ISL_Fig_2_15.png', embed=True)

# <codecell>

IPython.display.Image(filename='/home/j/Project/Machine-Learning/ML4HM/ISL_Fig_2_17.png', embed=True)

# <markdowncell>

# # Exercise 2

# <markdowncell>

# ## Exercise 2: In-sample and out-of-sample predictive validity and $k$-NN
# 
# URL: http://dismod.ihme.washington.edu:3000/

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

# ## Predictive validity of $k$-NN in-sample and out-of-sample

# <codecell>

df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/ISL/Fig_2_9_data.csv', index_col=0)

# <codecell>

df['color'] = df['Y'].map({'red': 'orange', 'blue': 'purple'})

# <codecell>

figure(figsize=(8,8))
for g, dfg in df.groupby('color'):
    plot(dfg['X.1'], dfg['X.2'], 'o', mec=g, color='none', mew=3, ms=8)

# <markdowncell>

# ## Homework:
# 
# * Complete the replication of ISL Figure 2.17
# * Turn your elevator pitch into a "minimum viable hypothesis"
# * Assemble smallest dataset necessary to demonstrate your elevator pitch
# * *Read*

# <codecell>

!cd /homes/abie/nbconvert/; cp /homes/abie/notebook/2013_04_08_ML4HM_Lecture_2.ipynb L2.ipynb; ./nbconvert.py --format reveal L2.ipynb

# <codecell>

import ipynb_style
reload(ipynb_style)
ipynb_style.presentation()

# <codecell>

cp /homes/abie/nbconvert/L2* /home/j/Project/Machine-Learning/ML4HM/

