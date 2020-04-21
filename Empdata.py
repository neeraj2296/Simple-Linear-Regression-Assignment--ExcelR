# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:58:54 2020

@author: Neeraj Kumar S J
"""
#########################################Importing necassary modules#############################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing
######################################### Importing The Dataset###################################################################################################
empData = pd.read_csv('E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\emp_data.csv')
empData
empData.columns = 'sh','cr'
#########################################Visualising the data ###################################################################################################

plt.hist(empData.sh)
plt.boxplot(empData.sh)
plt.hist(empData.cr)
plt.boxplot(empData.cr)
#########################################Normalising the data#######################################################################################
nempData = preprocessing.normalize(empData)
nempData

empData.corr()
#Defining a function that calculates Mean Square Error############################################################################################################
def mse(y_predict,y_actual):
    a = y_predict
    b = y_actual
    mse = np.square(np.subtract(a,b)).mean()
    return mse
######################################### Buliding Model 1 #######################################################################################################
mod1 = smf.ols('cr~sh',data = empData).fit()
mod1.summary()
mod1.params
#Predicting Using Model 1
p1_emp = mod1.predict(empData)
p1_emp
plt.scatter(empData.sh,empData.cr,color = 'red');plt.plot(empData.sh,p1_emp,color = 'black')
mse_emp = mse(p1_emp,empData.cr)
mse_emp
######################################Building Model 2############################################################################################################
mod2 = smf.ols('cr~np.log(sh)',data = empData).fit()
mod2.summary()
mod2.params
#Predicting Using Model 2
p2_emp = mod2.predict(empData)
p2_emp
plt.scatter(empData.sh,empData.cr,color = 'red');plt.plot(empData.sh,p2_emp,color='black')
mse_emp1 = mse(p2_emp,empData.cr)
mse_emp1
######################################Building Model 3############################################################################################################
mod3 = smf.ols('np.log(cr)~sh',data = empData).fit()
mod3.summary()
mod3.params
#Predicting Using Model 3
p3 = mod3.predict(empData)
p3_emp = np.exp(p3)
p3
p3_emp
plt.scatter(empData.sh,empData.cr,color = 'red');plt.plot(empData.sh,p3_emp,color='black')
mse_emp2 = mse(p3_emp,empData.cr)
mse_emp2
######################################Building Model 4############################################################################################################
mod4 = smf.ols('np.log(cr)~np.log(sh)',data = empData).fit()
mod4.summary()
mod4.params
#Predicting Using Model 4
p4 = mod4.predict(empData)
p4_emp = np.exp(p4)
p4_emp
plt.scatter(empData.sh,empData.cr,color = 'red');plt.plot(empData.sh,p4_emp,color='black')
mse_emp3 = mse(p4_emp,empData.cr)
mse_emp3
######################################Building Model 5############################################################################################################
empData['sh_sq'] = np.square(empData.sh)
mod5 = smf.ols('cr~sh_sq',data = empData).fit()
mod5.summary()
#Predicting Using Model 5
p5_emp = mod5.predict(empData)
p5_emp
plt.scatter(empData.sh,empData.cr,color='red');plt.plot(empData.sh,p5_emp,color='black')
mse_emp4 = mse(p5_emp,empData.cr)
mse_emp4
####################################As the Model 4 has given the best fit line exactly linearly i.e. it accepts to all Values, Hence Solved######################
