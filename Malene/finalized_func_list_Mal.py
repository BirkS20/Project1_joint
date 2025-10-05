#All functions used in assignments 1a - 1d

#Importing necessary packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import (
    PolynomialFeatures,
)  # use the fit_transform method of the created object!
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import seaborn as sns 
from numpy.random import rand
from numpy.random import seed

#Setting a random seed to reproduce the same data
np.random.seed(3155)

#This function creates a designmatrix X either with or without the intercept.
#Arguments: vector x with n values. P is the polynomial degree.
def polynomial_features(x, p, intercept=bool): 
    n = len(x)
    X = np.zeros((n, p + 1)) 
    if intercept == True: #Keeps a first column filled with 1´s 
        for i in range(p+1):
            X[:, i] = x**i
    elif intercept == False: #""Jumps"" to the first column of values
        for i in range(1,p+1):
            X[:,i] = x**i
    else: #Error message if one forgets to declare wether the intercept should be included or not
        raise TypeError(f"Please include a boolean response to the function parameter the intercept") 
    return X #returns design matrix X 

#Creating the OLS parameter Theta
#Arguments: X from polynomial feature (or created by Scikit learn). y is our function. 
def OLS_parameters(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y  #Optimization of OLS theta. 
    return theta

#Creating the Ridge Parameter Theta
#Arguments: X from polynomial feature (or created by Scikit learn). y is our function. Lambda is a set regularization parameter.
def Ridge_parameters(X, y, lambda_own):
    p = (len(X[0,:]))
    #Assumes X is scaled and has no intercept column
    #np.eye calculates a matrix with values on the diagonal and else zeros. 
    return np.linalg.pinv(X.T @ X + lambda_own * np.eye(p,p)) @ X.T @ y #Returns the optimization of Ridge theta


#Full function calculating the MSE and R2 scores of the OLS regression. 
#Arguments: Does not include the intercept as default. Degree is the wished polynomial degree. 
def mse_poly_plot_OLS(degree, intercept=False): 
    poly_deg = np.arange(1, degree+1)
    results = {}

    #Running the function for varying values of n in our dataset
    for n in [300, 400, 500]:
        #Defining the dataset
        x = np.linspace(-1, 1, n)
        denominator = 1+(25*x**2)
        y = 1.0 / denominator + np.random.normal(0, 1, x.shape)

        #Initialize empty arrays for later use
        mse_train_list = np.zeros(degree)
        mse_test_list = np.zeros(degree)
        R2_test = np.zeros(degree)
        R2_train = np.zeros(degree)
        beta_matrix = np.zeros((degree, degree+1))
        beta_array = np.arange(degree+1)

        #Range for the chosen polynomial degree
        for p in range(1, degree+1):
            #Calculate design matrix X
            X = polynomial_features(x, p, intercept=intercept)
            #Split data into subsets of traning and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
            #Centre data (Does not include standard deviation)
            scaler = StandardScaler(with_std=False)
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            y_mean = np.mean(y_train)
            y_scaled_train = (y_train - y_mean)

            #Calculate the parameters through the optimization of OLS Theta (called beta in our function but it can be called on as whatever variable one wishes)
            beta = OLS_parameters(X_train_s, y_scaled_train)

            #Make the predictions: y tilde
            y_pred_train = X_train_s @ beta + y_mean
            y_pred_test = X_test_s @ beta + y_mean

            #Calculate MSE and R2 scores using the Scikit learn functions. 
            #Links to the functions used: 
            #MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
            #R2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
            mse_train_list[p-1] = MSE(y_train, y_pred_train)
            mse_test_list[p-1] = MSE(y_test, y_pred_test)
            R2_test[p-1] = r2_score(y_test, y_pred_test)
            R2_train[p-1] = r2_score(y_train, y_pred_train)

            # Storing betas in a matrix and plotting the betas
            for i in range(len(beta)):
                beta_matrix[p-1, i] = beta[i]
            plt.scatter(beta_array, beta_matrix[p-1, :], label=f'p={p}')

        #Add information to the scatter plot of the betas 
        plt.xlabel(r'$\theta$ index')
        plt.ylabel(r'Value of $\theta$')
        plt.title(fr'$\theta$ for n={n}')
        plt.legend(bbox_to_anchor=(1.2,0.5), loc='center right')
        plt.show()

        #Storing the results from the MSE and R2 functions for plotting later. 
        #Using the functionality of a dictionary. 
        results[n] = {
            "mse_train": mse_train_list,
            "mse_test": mse_test_list,
            "R2_train": R2_train,
            "R2_test": R2_test
        }

    #Creating subplot 1 for MSE values. 
    fig1, ax1 = plt.subplots(1,2, figsize=(12,5))

    #Plotting for all n values chosen earlier in code - but in the same plot. 
    for n, vals in results.items():
        ax1[0].plot(poly_deg, vals["mse_test"], label=f"MSE test (n={n})")
        ax1[1].plot(poly_deg, vals["mse_train"], label=f"MSE train (n={n})")

    #Adding information to the subplot 1
    ax1[0].set_title("MSE OLS Regression Test ")
    ax1[1].set_title("MSE OLS Regression Train")
    ax1[0].set_xlabel("Polynomial degree")
    ax1[1].set_xlabel("Polynomial degree")
    ax1[0].set_ylabel("MSE")
    ax1[0].set_ylabel("MSE")
    ax1[0].grid(True)
    ax1[1].grid(True)
    ax1[1].legend()
    ax1[0].legend()

    #Creating subplot 2 for R2 scores. 
    #Plotting for all n values chosen earlier in code - but in the same plot. 
    fig2, ax2 = plt.subplots(1,2, figsize=(12,5))
    for n, vals in results.items():
        ax2[0].plot(poly_deg, vals["R2_test"], label=fr'$R^2$ test (n={n})')
        ax2[1].plot(poly_deg, vals["R2_train"], label=fr'$R^2$ train (n={n})')
    
    #Adding information to the subplot 2 
    ax2[0].set_title(r"$R^2$ OLS Regression Test ")
    ax2[0].set_xlabel("Polynomial degree")
    ax2[0].set_ylabel(r"$R^2$")
    ax2[1].set_ylabel(r"$R^2$")
    ax2[1].set_xlabel("Polynomial degree")
    ax2[0].grid(True)
    ax2[1].grid(True)
    ax2[0].legend()
    ax2[1].legend()
    ax2[1].set_title(r"$R^2$ OLS Regression Train")      
    ax2[0].legend() 
    plt.show()

    return beta


#Full function calculating the MSE and R2 scores for the Ridge regression. 
#Arguments: Does not include the intercept as default. Degree is the wished polynomial degree and here sat as a default value that can be changed. 
def heatmap_ridge(intercept=False, degree=16):
    #Defining varying n´s and lambdas to iterate through
    n_list=(300, 400, 500)
    nlambdas = 16
    lambdas = np.logspace(-5, 1, nlambdas)
    
    #Iteration for the chosen n values
    for n in [300,400,500]:
        #Creating the dataset
        x = np.linspace(-1, 1, n)
        denominator = 1+(25*x**2)
        y = 1.0 / denominator + np.random.normal(0, 1, x.shape)

        #Initialize empty matrices for storing the results for the heatmaps
        mse_train = np.zeros((degree, nlambdas))
        mse_test  = np.zeros((degree, nlambdas))
        r2_train  = np.zeros((degree, nlambdas))
        r2_test   = np.zeros((degree, nlambdas))

        #Iterations through the chosen polynomial degree (p)
        for p in range(1, degree + 1):
            #Creating the design matrix X 
            X = polynomial_features(x, p, intercept=intercept)
            #Splitting the data into subsets of training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3155)
            #Scaling the data with standard deviation for the Ridge Regression
            scaler = StandardScaler(with_std=True)
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
            y_mean = y_train.mean()
            y_center = y_train - y_mean / np.std(y)

            #Ridge Regression optimizer for the varying lambdas defined above in the code 
            for j, lmb in enumerate(lambdas):
                beta = Ridge_parameters(X_train_s, y_center, lmb)

                #Create the predictions of y tilde
                y_pred_tr = X_train_s @ beta + y_mean
                y_pred_te = X_test_s  @ beta + y_mean

                #Calculate MSE values and R2 scores using the Scikit learn functions: 
                #Links to the functions used: 
                #MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
                #R2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
                mse_train[p-1, j] = MSE(y_train, y_pred_tr)
                mse_test [p-1, j] = MSE(y_test,  y_pred_te)
                r2_train [p-1, j] = r2_score(y_train, y_pred_tr)
                r2_test  [p-1, j] = r2_score(y_test,  y_pred_te)

        #Creating default information for the heatmaps
        xticks = [f'{lmb:.1e}' for lmb in lambdas]
        yticks = np.arange(1, degree + 1)
        cmap = 'PiYG'

        #Subplot 1 for the MSE training data
        fig_mse, axes_mse = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        #Plotting the training data and providing plot information
        sns.heatmap(mse_train, ax=axes_mse[0], xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar=True)
        axes_mse[0].set_title(f'MSE Ridge Regression Train n={n}')
        axes_mse[0].set_xlabel(r'$\lambda$')
        axes_mse[0].set_ylabel('Polynomial degree')
        axes_mse[0].invert_yaxis()
        axes_mse[0].invert_xaxis()
        axes_mse[0].tick_params(axis='x', rotation=45)

        #Plotting the test data and providing plot information
        sns.heatmap(mse_test, ax=axes_mse[1], xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar=True)
        axes_mse[1].set_title(f'MSE Ridge Regression Test n={n}')
        axes_mse[1].set_xlabel(r'$\lambda$')
        axes_mse[1].set_ylabel('Polynomial degree')
        axes_mse[1].invert_yaxis()
        axes_mse[1].invert_xaxis()
        axes_mse[1].tick_params(axis='x', rotation=45)

        #Create subplot 2 
        fig_r2, axes_r2 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        #Plotting the training data and providing plot information
        sns.heatmap(r2_train, ax=axes_r2[0], xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar=True, vmin=0.0, vmax=1.0)
        axes_r2[0].set_title(f'$R^2$ Ridge Regression Train n={n}')
        axes_r2[0].set_xlabel(r'$\lambda$')
        axes_r2[0].set_ylabel('Polynomial degree')
        axes_r2[0].invert_yaxis()
        axes_r2[0].invert_xaxis()
        axes_r2[0].tick_params(axis='x', rotation=45)
        #Plotting the test data and providing plot information
        sns.heatmap(r2_test, ax=axes_r2[1], xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar=True, vmin=0.0, vmax=1.0)
        axes_r2[1].set_title(f'$R^2$ Ridge Regression Test n={n}')
        axes_r2[1].set_xlabel(r'$\lambda$') 
        axes_r2[1].set_ylabel('Polynomial degree')
        axes_r2[1].invert_yaxis()
        axes_r2[1].invert_xaxis()
        axes_r2[1].tick_params(axis='x', rotation=45)
        plt.show()

        #This plots return two subplots for three values of n




#Common comments
#Initialize empty arrays for later use