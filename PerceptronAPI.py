import numpy as np 
import pandas as pd
import matplotlib as mpl

"""
Attributes: 
w_ : 1d-array representing weights after fitting
b_ : scalar representing the bias after fitting 
"""

class Perceptron: 

    """
    learning rate: eta value is set to 0.01 (multiplier by which we change the weight and bias)
    epochs: maximum number of iterations/passes is 50 
    random state: seed for an RNG for random weight initialization
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1): 
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state 

    """
    X is a matrix of size n_examples by n_features 
    y contains the target values
    """
    def fit(self, X, y):
        # using the seed instantiate a RNG
        rgen = np.random.RandomState(self.random_state)

        # generating an 2d array of random numbers where the shape of the subarray is n_features
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # creates a float? 
        self.b_ = np.float_(0.)
        self.errors_ = [] 

        for _ in range(self.n_iter): 
            errors = 0
            for xi, target in zip(X, y): 
                # the perceptron learning rule 
                update = self.eta * (target - self.predict(xi))
                # does this operation apply to all values? 
                # update of the weight depends on the feature
                self.w_ += update*xi 
                # update of the bias does not depend on the feature
                self.b_ += update 
                # if there is error, add this to the errors
                errors += int(update != 0.0)

            self.errors_.append(errors)
    
    def net_input(self, X): 
        return np.dot(X, self.w_) + self.b_

    """
    Make predictions
    """
    def predict(self, X):
        # return the predicted class label after the unit step
        # "where" is like a ternary 
        return np.where(self.net_input(X) >= 0.0, 1, 0)