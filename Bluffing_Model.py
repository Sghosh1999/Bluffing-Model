# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 19:00:18 2019

@author: Sayantan Ghosh
Project Name : Bluffing Model of a new Employee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

from sklearn.cross_validation import train_test_split
X_train,Y_train,X_test,Y_test = train_test_split(X, Y, test_size = 0.3,random_state = 0)

# Fitting Linear Regression Model 

from sklearn.linear_model import LinearRegression
linreg1 = LinearRegression()
linreg1.fit(X,Y)
accuracy = linreg1.score(X,Y)
print("Accuracy of linear regresson Model is",accuracy*100,'%')


# Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
linreg2 = LinearRegression()
linreg2.fit(X_poly,Y)
accuracy1 = linreg2.score(X_poly,Y)
print("Accuracy of polynomial regression model is",accuracy1*100,'%')
Y_pred = linreg2.predict(poly_reg.fit_transform(1)) #Enter the level to predict the Salary
print("Predicted Value is:",Y_pred)

#Plotting of Linear Regression Model
plt.scatter(X,Y, color = 'red')
plt.plot(X,linreg1.predict(X), color = 'blue')
plt.title('Bluffing Model-Linear Regression')
plt.xlabel('Postion Level')
plt.ylabel('Predicted Salary')
plt.show()

#Plotting Of Polynomial Regression Model
X_grid = np.arange(min(X) , max(X) ,0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y, color = 'red')
plt.plot(X_grid,linreg2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title('Bluffing Model-Polynomial Regression')
plt.xlabel('Postion Level')
plt.ylabel('Predicted Salary')
plt.show()