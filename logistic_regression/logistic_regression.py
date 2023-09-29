import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    def __init__(self, transformer=None):
        self.transformer = transformer #data/feature transformer


    def fit(self, X, y, epochs = 1000):
 
        X = X.copy()
        X = np.asarray(X); y = np.asarray(y)

        if self.transformer: X = self.transformer.transform(X)

        self.weights = np.zeros(X.shape[1])        
        self.bias = 0.0
        gradient_alpha = 0.1
        epoch = 0

        self.training_losses = np.array([])
        self.training_accuracies = np.array([])

        while(True):
            #compute predictions
            pred = sigmoid(np.dot(X, self.weights) + self.bias)

            #compute gradients
            delta_weights, delta_bias = gradients(X, y, pred)

            #update
            self.weights -= gradient_alpha * delta_weights
            self.bias -= gradient_alpha * delta_bias 
            
            pred_class = np.array([1 if p > 0.5 else 0 for p in pred])

            self.training_accuracies = np.append(self.training_accuracies, [binary_accuracy(y, pred_class)], axis=0)
            self.training_losses = np.append(self.training_losses, [binary_cross_entropy(y, pred)])

            if(epoch > epochs): break

            epoch+=1
            
        self.finishing_epoch = epoch

    def predict(self, X):
        X = X.copy()
        X = np.asarray(X)
        
        if self.transformer: X = self.transformer.transform(X)        

        return sigmoid(np.dot(X, self.weights) + self.bias)
    

# --- Some utility functions 
class PolynomialTransformer:
    #Data/feature transform object generalized.
    def __init__(self, degree=2):
        if degree < 2:
            raise ValueError("Degree of polynomial transformation must be 2 or greater.")
        else:
            self.degree = degree

    def transform(self, X):
        X_poly = np.asarray(X).copy()
        X_poly = (X_poly - np.mean(X_poly, axis=0)) / np.std(X_poly, axis=0) #normalizing dataset

        for col in range(X_poly.shape[1]):
            for d in range(2, self.degree + 1):
                X_poly = np.column_stack((X_poly, X[:, col] ** d))

        return X_poly

def polynomial_degree_tester(X, y, degrees):
    #tests accuracies on training set  
    accuracies = np.zeros(len(degrees))
    cross_entropies = np.zeros(len(degrees))

    for i, d in enumerate(degrees):
        poly = PolynomialTransformer(degree=d)
        model = LogisticRegression(poly)
        model.fit(X, y)
        pred = model.predict(X)
        accuracies[i] = binary_accuracy(y_true=y, y_pred=pred, threshold=0.5)
        cross_entropies[i] = binary_cross_entropy(y_true=y, y_pred=pred)
        print(f"degree: {d}, \t accuracy: {accuracies[i]} \t cross entropy: {cross_entropies[i]}")

    return accuracies, cross_entropies


def gradients(x, y_true, y_pred):
    #computes a step in gradient decent
    diff = y_pred - y_true
    gradient_b = np.mean(diff)
    gradient_ws = np.matmul(np.transpose(x),diff)
    gradient_ws = np.array([np.mean(gradient) for gradient in gradient_ws])
    return gradient_ws, gradient_b

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        