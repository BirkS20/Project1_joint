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

def Ridge_parameters(X, y, lambda_own):
    p = (len(X[0,:]))
    # Assumes X is scaled and has no intercept column
    # np.eye calculates a matrix with values on the diagonal and else zeros.
    #Ie. the identity matrix in this instance
    return np.linalg.pinv(X.T @ X + lambda_own * np.eye(p,p)) @ X.T @ y


#MSE and changing the polynomial degree
def mse_poly_plot_OLS(degree, p, intercept=bool):
    for n in [100,1000,10000]:
        #create dataset
        x = np.linspace(-1,1, n) #x within interval [-1,1]
        denominator = 1+(25*x**2)
        y = 1/denominator# + np.random.normal(0, 1, x.shape) 

        #create empty lists
        poly_deg = np.linspace(1,degree,degree+1)
        mse_train_list = np.zeros(degree+1)
        mse_test_list = np.zeros(degree+1)
        R2_test = np.zeros(degree+1)
        R2_train = np.zeros(degree+1)
        beta_matrix = np.zeros((degree+1, degree+1))

        #range polynomial degree
        for p in range(1, degree+1):
            X = polynomial_features(x,p,intercept=intercept)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            scaler = StandardScaler(with_std = False)
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            y_mean = np.mean(y_train)
            y_scaled_train = (y_train - y_mean)

            beta = OLS_parameters(X_train_s,y_scaled_train)
        
            y_pred_train = (X_train_s @ beta + y_mean)
            y_pred_test = (X_test_s @ beta + y_mean)

            mse_train_list[p] = MSE(y_train,y_pred_train)
            mse_test_list[p] = MSE(y_test, y_pred_test)

            R2_test[p] = r2_score(y_test, y_pred_test)
            #print(f'The R2 score is: {R2_test} for degree: {p}')
            R2_train[p] = r2_score(y_train, y_pred_train)

            for i in range(len(beta)):
                beta_matrix[p-1,i] = beta[i]
            #print(beta_matrix)
        
        num_rows, num_columns = beta_matrix.shape
        bar_width = 0.2
        r = np.arange(num_columns)
        plt.figure(figsize=(20,8))
        for m in range(num_rows): 
            plt.bar(r + m * bar_width, beta_matrix[m, :degree+1], label = f'Row: {m+1}', width=0.2)
            plt.yscale('log')

        fig,ax = plt.subplots(3, figsize=(10,12))
        ax[0].plot(x, y, label = "x,y", color = "forestgreen")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].legend()
        ax[0].grid(True)
        ax[0].set_title('Function')
        ax[1].plot(poly_deg, mse_test_list, label = "MSE test", color = "magenta")
        ax[1].plot(poly_deg, mse_train_list, label = "MSE train", color = "cyan")
        ax[1].set_title(f'MSE for n = {n}')
        ax[1].set_ylabel("MSE")
        ax[1].set_xlabel("Polynomial Degree")
        ax[2].plot(poly_deg, R2_test, label = 'R2 score test data', color = 'm')
        ax[2].plot(poly_deg, R2_train, label = 'R2 score train data', color = 'b')
        ax[2].set_ylim(-1,1)
        ax[2].set_title(f'R2 score for n = {n}')
        ax[1].set_xlim(2)
        ax[1].grid(True)
        ax[1].legend()
        ax[2].legend()
        ax[2].grid(True)
        plt.tight_layout()
        plt.show()
    return mse_test_list, mse_train_list

