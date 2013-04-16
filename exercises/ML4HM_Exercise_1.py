# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Exercise 1: Predicting Weather

# <codecell>

import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/weather-numeric.csv')

# <codecell>

df.head()

# <codecell>

# use TAB to explore commands

# <markdowncell>

# ## Abie's dumb predictor

# <codecell>

def predict(s):
    if s['outlook'] == 'sunny':
        return 'no'
    else:
        return 'yes'

# <codecell>

predict(df.xs(1)) # xs means "cross-section" aka row

# <markdowncell>

# ## How good is this dumb predictor?

# <codecell>

i = 0
predict(df.xs(i)) == df.play[i]

# <codecell>

for i in df.index:
    # count how many predictions are correct
    pass

# <markdowncell>

# ## How much better can you do with a single rule?

# <codecell>


