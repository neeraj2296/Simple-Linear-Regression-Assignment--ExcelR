# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:55:11 2020

@author: Neeraj Kumar S J
"""
##############################Importing the necassary modules################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing
##############################Importing the necassary modules################################
delt = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\delivery_time.csv")
delt.columns = 'dt','st'

############################# Visualization #################################################
plt.hist(delt['dt'])
plt.hist(delt['st'])
plt.boxplot(delt['dt'])
plt.boxplot(delt['st'])
#Calculating the correlation between delivery time and sorting time
delt.corr()
#normalizing
norm_delt = preprocessing.normalize(delt)
plt.hist(norm_delt)
###########################Building the model#################################################
#Building Model 1
mod1 = smf.ols('dt~st',data = delt).fit()
mod1.summary()
#predicting with model 1
p1_delivery = mod1.predict(delt) 
p1_delivery
plt.scatter(delt.st, delt.dt, color='red');plt.plot(delt.st, p1_delivery,color='black')
#defnimg the Mean Square Error Function
def mse(y_pred,y_actual):
    a=y_pred
    b=y_actual
    mse=np.square(np.subtract(a, b)).mean()
    return mse
#Mean Square Error of Model 1
mse1 = mse(p1_delivery,delt.dt)
mse1
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Building Model 2
mod2 = smf.ols('dt~np.log(st)',data = delt).fit()
mod2.summary()
#predicting with model 2
p2_delivery = mod2.predict(delt) 
p2_delivery
plt.scatter(delt.st, delt.dt, color='red');plt.plot(delt.st, p2_delivery,color='black')
#Mean Square Error of Model 2
mse2 = mse(p2_delivery,delt.dt)
mse2
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Building Model 3
mod3 = smf.ols('np.log(dt)~(st)',data = delt).fit()
mod3.summary()
#predicting with model 3
p3_del = mod3.predict(delt)
p3_del
p3_delivery = np.exp(p3_del)
plt.scatter(delt.st,delt.dt, color ='red');plt.plot(delt.st, p3_delivery, color='black')
#Mean Square Error of Model 3
mse3 = mse(p3_delivery,delt.dt)
mse3
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Building Model 4
delt['st_sq'] = delt.st*delt.st
mod4 = smf.ols('dt~st_sq',data=delt).fit()
mod4.summary()
#predicting with model 4
p4_delivery = mod4.predict(delt)
p4_delivery
plt.scatter(delt.st,delt.dt, color='red');plt.plot(delt.st, p4_delivery,color = 'black')
#Mean Square Error of Model 4
mse4 = mse(p4_delivery,delt.dt)
mse4

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Building Model 5
mod5 =smf.ols('np.log(dt)~np.log(st)',data = delt).fit()
mod5.summary()
#predicting with model 5
p5_del = mod5.predict(delt)
p5_del
p5_delivery = np.exp(p5_del)
plt.scatter(delt.st,delt.dt,color='red');plt.plot(delt.st, p5_delivery,color = 'black')
#Mean Square Error of Model 5
mse5 = mse(p5_delivery,delt.dt)
mse5