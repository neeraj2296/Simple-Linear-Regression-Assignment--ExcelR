# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:37:37 2020

@author: Neeraj Kumar S J
"""
#########################################Importing necassary modules#############################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
######################################### Importing The Dataset###################################################################################################
cal_consumed = pd.read_csv('E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\calories_consumed.csv')
cal_consumed.info()
calories = cal_consumed
calories.columns = 'weight','calories'
calories['weight'].value_counts()
calories['calories'].value_counts()
#########################################Visualising the data ###################################################################################################
plt.boxplot(calories['weight'])
plt.hist(calories['weight'])
plt.boxplot(calories['calories'])
plt.hist(calories['calories'])

calories.skew()
calories.kurt()
#########################################Normalising and stdizing the data#######################################################################################
#Normalizing the data
n_calories = preprocessing.normalize(calories)
plt.hist(n_calories[0])
#Scaling the data
s_calories = preprocessing.scale(calories)
plt.hist(s_calories)
#Standardized scaling the data
Std_Scaler = preprocessing.StandardScaler()
stdsc_calories = Std_Scaler.fit_transform(calories)
plt.hist(stdsc_calories)

#plt.plot(stdsc_calories[0],stdsc_calories[1],'ro');plt.xlabel = 'Weight Gained';plt.ylabel = 'Calories Consumed'
plt.plot(calories.weight,calories.calories,'go');plt.xlabel = 'Weight Gained';plt.ylabel = 'Calories Consumed'
#Fiding its correlation
calories.corr()
calories.calories.corr(calories.weight)
np.corrcoef(calories.weight,calories.calories)
#########################################Importing necassary modules #############################################################################################
import statsmodels.formula.api as smf
######################################### Buliding Model 1 #######################################################################################################
mod1 = smf.ols('weight~calories',data=calories).fit()
type(mod1)
mod1.params
mod1.summary()
mod1.conf_int(0.05)
#Predicting Using Model 1
P_calories = mod1.predict(calories)
resid = P_calories - calories.weight
resid.mean()
#Studentizing the residuals
student_resid = mod1.resid_pearson
student_resid
#Plotting the graph caloris wrt predicted weight
plt.scatter(calories.calories, calories.weight, color='Red');plt.plot(calories.calories, P_calories, color='black');plt.xlabel('Calorie');plt.ylabel('WeightGained')
#Defining a function that calculates Mean Square Error############################################################################################################
def mse(y_pred,y_actual):
    a = y_pred
    b = y_actual
    mse = np.square(np.subtract(a, b)).mean()
    return mse
#Calculating Mean Square Error For Model 1
mse_mod1 = mse(P_calories, calories.weight)
######################################Building Model 2############################################################################################################
mod2 =  smf.ols('weight~np.log(calories)',data = calories).fit()
mod2.params
mod2.summary()
#Predicting Using Model 2
P1_calories = mod2.predict(calories)
resid = P1_calories - calories.weight
resid.mean()
#Studentizing the residuals
student_resid = mod2.resid_pearson
student_resid
#Plotting the graph caloris wrt predicted weight
plt.scatter(calories.calories, calories.weight, color='Red');plt.plot(calories.calories, P1_calories, color='black');plt.xlabel('Calories');plt.ylabel('WeightGained')
#Calculating Mean Square Error For Model 2
mse_mod2 = mse(P2_calories, calories.weight)
######################################Building Model 3############################################################################################################
mod3=smf.ols("np.log(weight)~(calories)",data=calories).fit()
mod3.params
mod3.summary()
#Predicting Using Model 3
P2_log = mod3.predict(calories)
P2_calories=np.exp(P2_log)
resid = P2_calories - calories.weight
resid.mean()
#Studentizing the residuals
student_resid = mod3.resid_pearson
student_resid
#Plotting the graph caloris wrt predicted weight
plt.scatter(calories.calories, calories.weight, color='Red');plt.plot(calories.calories, P2_calories, color='black');plt.xlabel('Calories');plt.ylabel('WeightGained')
#Calculating Mean Square Error For Model 3
mse_mod3 = mse(P2_calories, calories.weight)
######################################Building Model 4#############################################################################################################
mod4 = smf.ols("np.log(weight)~np.log(calories)",data = calories).fit()
mod4.params
mod4.summary()
#Predicting Using Model 4
P3_calories_log = mod4.predict(calories)
P3_calories=np.exp(P3_calories_log)
resid = P3_calories - calories.weight
resid.mean()
#Studentizing the residuals
student_resid = mod4.resid_pearson
student_resid
#Plotting the graph caloris wrt predicted weight
plt.scatter(calories.calories, calories.weight,color = 'Red');plt.plot(calories.calories, P3_calories, color = 'Black');plt.xlabel('Calories');plt.ylabel('WeightGained')
#Calculating Mean Square Error For Model 4
mse_mod4 = mse(P3_calories, calories.weight)
###################################### As the Model 4 has given the best fit line exactly linearly i.e. it accepts to all Values, Hence Solved #####################
