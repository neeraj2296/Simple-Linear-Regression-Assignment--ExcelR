# pip install lifelines
# import lifelines
# from datasets importing load_waltons data
# from lifelines.datasets import load_waltons

import pandas as pd
# Loading the the survival un-employment data
survival_unemp = pd.read_csv("C:\\Datasets_BA\\Python Scripts\\survival analysis\\survival_unemployment1.csv")
survival_unemp.head()
survival_unemp.describe()

survival_unemp["spell"].describe()

# Spell is referring to time 
T = survival_unemp.spell

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T,event_observed=survival_unemp.event)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
survival_unemp.ui.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_unemp.ui==1], survival_unemp.event[survival_unemp.ui==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_unemp.ui==0], survival_unemp.event[survival_unemp.ui==0], label='0')
kmf.plot(ax=ax)



##############################################################################














# loading the inbuilt load_waltons data set into spyder
df = load_waltons()
help(load_waltons)

df.head(10)
# df.T => Array of durations 
# df.E => Boolean values representing the occurance of death (1 or 0)

# Storing the time and events for death in T and E
T = df['T']
E = df['E']

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T,event_observed=E)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group

df.group.value_counts()
# control - 129
# miR-137 - 34

# Applying KaplanMeierFitter model on Time and Events for the group "control"
kmf.fit(T[df.group=="control"], E[df.group=="control"], label='control')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "miR-137"
kmf.fit(T[df.group=="miR-137"], E[df.group=="miR-137"], label='miR-137')
kmf.plot(ax=ax)

# Plot the Time-line estimations plot for 2 groups on same graphs itself
