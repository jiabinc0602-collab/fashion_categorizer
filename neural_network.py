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

class DeepNeuralNetwork:
    def __init__(self, layer_dims):
        self.params = {}
        self.L = len(layer_dims) - 1
        self.layer_dims = layer_dims

        for l in range(1, len(layer_dims)):
            current_dim = layer_dims[l]
            prev_dim = layer_dims[l-1]

            self.params["W" + str(l)] = np.random.randn(current_dim, prev_dim) * np.sqrt((2/prev_dim))
            self.params["b" + str(l)] = np.zeros((current_dim, 1))

    def forward_propagation(self, X):
        caches = []
        A = X
        for l in range(1, self.L):
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]

            A_prev = A
            
            Z = np.dot(W, A_prev) + b
            A_next = relu(Z)

            caches.append((A, W, b, Z))
            
            A = A_next
        
        W = self.params["W" + str(self.L)]
        b = self.params["b" + str(self.L)]

        Z = np.dot(W, A) + b
        AL = softmax(Z)

        caches.append((A, W, b, Z))

        return AL, caches
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]

        sum = np.sum(Y * np.log(AL + 1e-10))
        cost = -(1/m) * sum

        return np.squeeze(cost)

    def backward_propagation(self, AL, Y, caches):
        grads = {}

        L = self.L

        Y = np.reshape(Y, AL.shape)

        m = AL.shape[1]

        current_cache = caches[L-1]

        A_prev, W, b, Z = current_cache

        dZ = AL - Y
        grads["dW" + str(L)] = (1/m) * np.dot(dZ, A_prev.T)
        grads["db" + str(L)] = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
        grads["dA" + str(L-1)] = np.dot(W.T, dZ)

        for l in reversed(range(1, L)):
            current_cache = caches[l-1]

            A_prev, W, b, Z = current_cache

            dA = grads["dA" + str(l)]

            dZ = relu_backward(dA, Z)

            grads["dW" + str(l)] = (1/m) * np.dot(dZ, A_prev.T)
            grads["db" + str(l)] = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
            grads["dA" + str(l-1)] = np.dot(W.T, dZ)
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        L = self.L

        for l in range(L):
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    def predict(self, X):
        AL, _ = self.forward_propagation(X)

        predictions = np.argmax(AL, axis = 0)

        return predictions
