# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:58:54 2020

@author: Neeraj Kumar S J
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing

empData = pd.read_csv('E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\emp_data.csv')
empData
empData.columns = 'sh','cr'

plt.hist(empData.sh)
plt.boxplot(empData.sh)
plt.hist(empData.cr)
plt.boxplot(empData.cr)

nempData = preprocessing.normalize(empData)
nempData

empData.corr()

def mse(y_predict,y_actual):
    a = y_predict
    b = y_actual
    mse = np.square(np.subtract(a,b)).mean()
    return mse

mod1 = smf.ols('cr~sh',data = empData).fit()
mod1.summary()
mod1.params
p1_emp = mod1.predict(empData)
p1_emp
plt.scatter(empData.sh,empData.cr,color = 'red');plt.plot(empData.sh,p1_emp,color = 'black')
mse_emp = mse(p1_emp,empData.cr)
mse_emp

mod2 = smf.ols('cr~np.log(sh)',data = empData).fit()
mod2.summary()
mod2.params
p2_emp = mod2.predict(empData)
p2_emp
plt.scatter(empData.sh,empData.cr,color = 'red');plt.plot(empData.sh,p2_emp,color='black')
mse_emp1 = mse(p2_emp,empData.cr)
mse_emp1

mod3 = smf.ols('np.log(cr)~sh',data = empData).fit()
mod3.summary()
mod3.params
p3 = mod3.predict(empData)
p3_emp = np.exp(p3)
p3
p3_emp
plt.scatter(empData.sh,empData.cr,color = 'red');plt.plot(empData.sh,p3_emp,color='black')
mse_emp2 = mse(p3_emp,empData.cr)
mse_emp2

mod4 = smf.ols('np.log(cr)~np.log(sh)',data = empData).fit()
mod4.summary()
mod4.params
p4 = mod4.predict(empData)
p4_emp = np.exp(p4)
p4_emp
plt.scatter(empData.sh,empData.cr,color = 'red');plt.plot(empData.sh,p4_emp,color='black')
mse_emp3 = mse(p4_emp,empData.cr)
mse_emp3

empData['sh_sq'] = np.square(empData.sh)
mod5 = smf.ols('cr~sh_sq',data = empData).fit()
mod5.summary()
p5_emp = mod5.predict(empData)
p5_emp
plt.scatter(empData.sh,empData.cr,color='red');plt.plot(empData.sh,p5_emp,color='black')
mse_emp4 = mse(p5_emp,empData.cr)
mse_emp4
