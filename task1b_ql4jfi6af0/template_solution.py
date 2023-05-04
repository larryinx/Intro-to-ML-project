# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge




def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # TODO: Enter your code here
    transformer = [lambda x: x, lambda x: x**2, np.exp, np.cos, lambda x: 1]
    for i in range(5):
        for j in range(5):
            if (i == 4 and j == 1):
                break
            X_transformed[:, 5*i+j] = transformer[i](X[:, j])
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y, alpha):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    # TODO: Enter your code here
    
    # xTx = np.dot(X_transformed.T, X_transformed)
    # w = np.dot(np.dot(np.linalg.inv(xTx + 24.53072*np.eye(xTx.shape[0])),X_transformed.T),y)    

    model = Ridge(alpha=350, fit_intercept=False)
    model.fit(X_transformed, y)
    w = model.coef_
    assert w.shape == (21,)
    return w

def calculate_RMSE(w, X, y):   
    RMSE = 0
    # TODO: Enter your code here
    RMSE = np.sqrt(np.mean((np.dot(X, w) - y)**2))
    
    assert np.isscalar(RMSE)
    return RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    

    # find the best lambda value

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    lambdas = np.linspace(0.001, 20, 200)

    # Train Ridge regression models with different Lambdas
    res = []
    for lam in lambdas:
        ridge = Ridge(alpha=lam, fit_intercept=False)
        scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ridge.fit(X_train, y_train)
            w = ridge.coef_
            scores.append(calculate_RMSE(w, X_test, y_test))
        score = np.mean(scores)
        print("Lambda:", lam, "RMSE:", score)
        res.append((lam, np.mean(score)))

    # Sort the results by the score 
    best_lambda = sorted(res, key=lambda x: x[1], reverse=False)[0][0]

    print("Best Lambda:", best_lambda)



    # The function retrieving optimal LR parameters
    w = fit(X, y, best_lambda)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
