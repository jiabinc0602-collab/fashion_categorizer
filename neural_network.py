import numpy as np
import pandas as pd

def relu(Z):
    return np.maximum(0, Z)
        
def softmax(Z):
    x = Z - np.max(Z, axis = 0, keepdims = True)
    numerator = np.exp(x)
    denominator = np.sum(numerator, axis = 0, keepdims = True)
    return numerator / denominator

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def one_hot(Y, C):
    Y_hot = np.eye(C)[Y.reshape(-1)]
    return Y_hot.T
