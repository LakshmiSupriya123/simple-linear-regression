"""
Created on Thu Apr 14 18:28:50 2022

@author: Supriya
"""
#import pandas
import numpy as np
import pandas as pd
# Loading the data
df = pd.read_csv("C:/Users/NAVEEN REDDY/Downloads/delivery_time.csv")
df.shape
type(df)
list(df)
df.ndim
X = df['Delivery Time'] # Only Independent variables
X.shape
X.ndim
type(X)
X = X[:, np.newaxis] # converting in to 2 D arrary from 1 D array
X.ndim
type(X)
Y = df['Sorting Time']  # Only Dependent variable
Y.shape
Y.ndim

# scatter plot
df.plot.scatter(x='Delivery Time', y='Sorting Time')
df.corr()

# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, Y)
model.intercept_  ## To check the Bo values
model.coef_       ## To check the coefficients (B1)

# Make predictions using independent variable values
Y_Pred = model.predict(X)

# Plot outputs
import matplotlib.pyplot as plt
plt.scatter(X, Y,  color='black')
plt.plot(X, Y_Pred, color='red')
plt.show()


# Errors are the difference between observed and predicted values.
Y_error = Y-Y_Pred
print(Y_error)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred)
mse

RMSE = np.sqrt(mse)
RMSE
''' conclusion : R sq and mse of the Model is Good and the model can be accepted '''