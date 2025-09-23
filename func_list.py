#Functions used in Project 1 

#import packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import (
    PolynomialFeatures,
)  # use the fit_transform method of the created object!
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE

#The function creates a designmatrix X either with or without the intercept.
def polynomial_features(x, p, intercept=bool):
    n = len(x)
    X = np.zeros((n, p + 1)) 
    if intercept == True: #Keeps a first column of ones
        for i in range(p+1):
            X[:, i] = x**i
    elif intercept == False: #jumps to first column of values
        for i in range(1,p+1):
            X[:,i] = x**i
    else:
        raise TypeError(f"Please include a boolean response to the function parameter the intercept")
    return X


#Creating the OLS parameter by using 
def OLS_parameters(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y  #Optimization of OLS theta calculated week 35. 
    return theta

#MSE and changing the polynomial degree
