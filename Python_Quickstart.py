# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Here are a few useful things to know about Python
# 
# All touched on in Class 1

# <codecell>

# assignment
x = 0

# <codecell>

# conditional statements (and print output)
if x > 0:
    print 'positive'
elif x == 0:
    print 'zero'
else:
    print 'negative'

# <codecell>

# functions
def my_func(y):
    return y**2
my_func(2)

# <codecell>

# loops
for x in range(10):
    print my_func(x),

# <codecell>

# DataFrames and reading csvs
import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/weather-numeric.csv')

# <codecell>

df.index

# <codecell>

df.columns

# <codecell>

print df

# <codecell>

df.humidity

# <codecell>

df.xs(12)

# <codecell>

from ipynb_style import clean
clean()

# <codecell>


