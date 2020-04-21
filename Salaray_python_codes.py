# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:01:27 2020

@author: Neeraj Kumar S J
"""
##################################### Importing Necassary Modules ################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing
##################################### Importing Necassary DataSet ################################################################

sal = pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\Salary_data.csv")
sal
sal.columns = 'ye','sal'
##################################### Visualisation ##############################################################################

plt.hist(sal.ye)
plt.hist(sal.sal)
plt.boxplot(sal.ye)
plt.boxplot(sal.sal)
#Normalizing
nsal = preprocessing.normalize(sal)
nsal
#Mean Square Error Functoion
def mse(y_pred,y_actual):
    a = y_pred
    b = y_actual
    mse = np.square(np.subtract(a,b)).mean()
    return mse
#Calculating The Correlation Between Years of Experience and Salary
sal.corr()
##################################### Building and analyzing Modues ##############################################################
#Building Model 1
mod1 = smf.ols('sal~ye',data = sal).fit()
mod1.summary()
mod1.params
#Predicting Using Model 1
p1_sal = mod1.predict(sal)
p1_sal
plt.scatter(sal.ye,sal.sal,color='red');plt.plot(sal.ye,p1_sal,color='black');plt.xlabel('experience in year');plt.ylabel('salary')
#Mean Square Error Calculation
mse1 = mse(p1_sal,sal.sal)
mse1
#Building Model 2
mod2 = smf.ols('sal~np.log(ye)',data = sal).fit()
mod2.summary()
mod1.params
#Predicting Using Model 2
p2_sal = mod2.predict(sal)
p2_sal
plt.scatter(sal.ye,sal.sal,color='red');plt.plot(sal.ye,p2_sal,color='black');plt.xlabel('experience in year');plt.ylabel('salary')
#Mean Square Error Calculation
mse2 = mse(p2_sal,sal.sal)
mse2
#Building Model 3
mod3 = smf.ols('np.log(sal)~ye',data = sal).fit()
mod3.summary()
mod3.params
#Predicting Using Model 3
p3_ye_log = mod3.predict(sal)
p3_sal = np.exp(p3_ye_log)
plt.scatter(sal.ye,sal.sal,color='red');plt.plot(sal.ye,p3_sal,color='black');plt.xlabel('experience in year');plt.ylabel('salary')
#Mean Square Error Calculation
mse3 = mse(p3_sal,sal.sal)
mse3
##################################### From camparing R-Squared Values of all modules ##############################################
##################################### We Can Conclude that Modue 1 is better model for prediction #################################