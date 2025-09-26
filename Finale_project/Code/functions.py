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
import seaborn as sns 

#CODE FOR THE HEATMAP
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

def Ridge_parameters(X, y, lambda_own):
    p = (len(X[0,:]))
    # Assumes X is scaled and has no intercept column
    # np.eye calculates a matrix with values on the diagonal and else zeros.
    #Ie. the identity matrix in this instance
    return np.linalg.pinv(X.T @ X + lambda_own * np.eye(p,p)) @ X.T @ y


from Project1_joint.Malene.func_list import Ridge_parameters

#MSE and changing lambda
def poly_plot_ridge(intercept=False, annotate = False):
    np.random.seed(3155)
    nlambdas = 16
    lambdas = np.logspace(-5,1,nlambdas)
    degree = 16
    poly_deg = np.arange(1,degree,degree+1) #c 
    for n in [100,1000,10000]:
        #create dataset
        x = np.linspace(-1,1, n) #x within interval [-1,1]
        denominator = 1+(25*x**2)
        y = 1/denominator# + np.random.normal(0, 1, x.shape) 

        #create empty lists
        mse_train_list = np.zeros((degree, nlambdas,))
        mse_test_list = np.zeros((degree, nlambdas))
        R2_test = np.zeros((degree, nlambdas))
        R2_train = np.zeros((degree, nlambdas))
        #beta_matrix = np.zeros(((nlambdas+1, nlambdas)))

        #range polynomial degree
        for p in range(1, degree+1):
            for j, lmb in enumerate(lambdas):
                X = polynomial_features(x,p,intercept=intercept)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                scaler = StandardScaler(with_std = True) #scale with standard deviation this time
                scaler.fit(X_train)
                X_train_s = scaler.transform(X_train)
                X_test_s = scaler.transform(X_test)
                y_mean = np.mean(y_train)
                #y_std = np.std(y_train)
                y_scaled_train = (y_train - y_mean) #centered, not scaled

                beta = Ridge_parameters(X_train_s, y_scaled_train, lmb)
    
                y_pred_train = (X_train_s @ beta + y_mean)
                y_pred_test = (X_test_s @ beta + y_mean)

                mse_train_list[p-1,j] = MSE(y_train,y_pred_train)
                mse_test_list[p-1,j] = MSE(y_test, y_pred_test)

                R2_test[p-1,j] = r2_score(y_test, y_pred_test)
                R2_train[p-1,j] = r2_score(y_train, y_pred_train)
        
        fig,axes = plt.subplots(2,2, figsize=(16,16)) #c 
        titles = ['MSE train', 'MSE test', '$R^2$ train', '$R^2$ test']
        data = [mse_train_list, mse_test_list, R2_train, R2_test]
        for ax, title,plotting in zip(axes.ravel(), titles, data):
            sns.heatmap(plotting, ax=ax, xticklabels=[f'{lmb:.1e}' for lmb in lambdas], yticklabels=np.arange(1,degree+1), cmap='PiYG', annot = annotate, cbar = True)
            ax.set_title(f'{title} for n: {n}')
            ax.set_ylabel('Polynomial Degree')
            ax.set_xlabel(f'$\lambda$')
            ax.invert_yaxis()
            ax.invert_xaxis()
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
    plt.tight_layout()
    plt.show()
    #return mse_test_list, mse_train_list
    return beta

    #return mse_test_list, mse_train_list

beta_ridge = poly_plot_ridge()


#CODE FOR THE OTHER FIGURE
n_vals = [50,100,150,200,250,300,350,400,450,500]
bootstraps = 1000
degree = 6
"""
x_true = np.linspace(-3, 3, n).reshape(-1,1)
y_true = np.exp(-(x_true**2)) + 1.5 * np.exp(-((x_true - 2) ** 2)) + np.random.normal(0, 0.1)
"""
biases = []
variances = []
mses = []

for n in [50,100,150,200,250,300,350,400,450,500]:
    x_true = np.linspace(-3, 3, n).reshape(-1,1)
    y_true = np.exp(-(x_true**2)) + 1.5 * np.exp(-((x_true - 2) ** 2)) + np.random.normal(0, 0.1)
    X_train, X_test, y_train, y_test = train_test_split(x_true, y_true, test_size = 0.2)

    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=True))
    predictions = np.empty((y_test.shape[0], bootstraps))
    targets = y_train
    for b in range(bootstraps):
        X_train_re, y_train_re = resample(X_train, y_train)

        # fit your model on the sampled data
        model.fit(X_train_re, y_train_re)

        # make predictions on the test data
        predictions[:,b] = model.predict(X_test).ravel()

    biases.append ( np.mean((y_test - np.mean(predictions, axis = 1, keepdims=True))**2)) 
    variances.append( np.mean(np.var(predictions, axis = 1, keepdims=True)))
    mses.append(np.mean(np.mean((y_test - predictions)**2, axis=1, keepdims=True)))


plt.plot(n_vals, biases, label = 'Bias', color = 'forestgreen')
plt.plot(n_vals, variances, label = 'Variance', color = 'navy')
plt.legend()


plt.title(f'Variance and Bias with varying n')
plt.ylabel('y')
plt.xlabel('n')
plt.grid(True)