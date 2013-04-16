# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Machine Learning for Health Metricians

# <markdowncell>

# # Lecture 1 Outline:
#     
# * Administrivia
# * What is machine learning?
# * Exercise 1: Predicting XXX

# <markdowncell>

# # Administrivia

# <markdowncell>

# ## Class expectations
# 
# * *Read*
# * Participate in class
# * Do a project

# <markdowncell>

# ## Who needs credit?

# <markdowncell>

# For this late addition to the schedule, it would be simplest to keep this completely informal.  I show up, you show up, we read, we learn, but there are no grades, and no credits.  It will be the size of a 1 credit class, however, comparable to the work in journal club, but with more programming exercises in class and as homework (20-30 min in class/30-60 min at home).
# 
# Independent study turns out not to be a good option, unfortunately.  If the 1 credit is very important to any of you, we can brainstorm creative alternative solutions.

# <markdowncell>

# ## When should we meet?

# <markdowncell>

# The time I originally picked is now: Monday 4-5 PM.
#     
# Sean would prefer we move to Friday 2-3 PM.
# 
# David would like to be later on Friday, 2:30-3:30 PM.
#         

# <markdowncell>

# ## What is machine learning?

# <markdowncell>

# One/Two/Four exercise, or writing and sharing, depending on class size and dynamics

# <markdowncell>

# ## A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$.
# ###---Tom Mitchell

# <markdowncell>

# ## What is the difference between machine learning and statistics?

# <markdowncell>

# The web suggests several criteria: searching hypotheses vs testing hypothesis; prediction vs inference; good marketing vs bad marketing; publishing in conferences vs publishing in journals; or simply sitting in a CS department vs sitting in a Stats dept.
# 
# I think there is a more fundamental difference, which may betray my CMU math department upbringing: a foundation of mathematical logic vs a foundation of real analysis.

# <codecell>

import IPython.display

# <markdowncell>

# ## Can a machine do what we can do?

# <codecell>

IPython.display.Image("http://upload.wikimedia.org/wikipedia/en/c/c8/Alan_Turing_photo.jpg")

# <codecell>

IPython.display.YouTubeVideo("W7Rq-PEW5qM")

# <markdowncell>

# ## But they [computers] are useless. They can only give you answers.
# 
# ### ---Picasso

# <codecell>

IPython.display.Image('http://upload.wikimedia.org/wikipedia/en/1/1c/Stravinsky_picasso.png')

# <markdowncell>

# ## What will we do in this class?

# <markdowncell>

# # Exercise 1: Predicting Something
# 
# * Contact Lenses
# * Weather
# * Iris
# * CPU Performance
# * Labor Negotiations
# * Soybean
# * Wage
# * Smarket
# * NCI60
# * Advertising
# * Income

# <markdowncell>

# ## Exercise 1: Predicting Weather
# 
# URL: http://dismod.ihme.washington.edu:3000/

# <codecell>

import pandas as pd
df = pd.read_csv('/home/j/Project/Machine-Learning/ML4HM/weather-numeric.csv')

# <codecell>

df.head()

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

# <markdowncell>

# ## Homework:
# 
# * Find the best "length-two decision list" for this
# * Develop an "elevator pitch" for a machine learning project related to your IHME research
# * *Read*

# <codecell>

!cd /homes/abie/nbconvert/; cp /homes/abie/notebook/2013_03_31_ML4HM_Lecture_1.ipynb L1.ipynb; ./nbconvert.py --format reveal L1.ipynb

# <codecell>

import ipynb_style
reload(ipynb_style)
ipynb_style.presentation()

# <codecell>


