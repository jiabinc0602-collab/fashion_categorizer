import numpy as np
import pandas as pd
from neural_network import DeepNeuralNetwork, one_hot
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    train = pd.read_csv('data/fashion-mnist_train.csv')
    test = pd.read_csv('data/fashion-mnist_test.csv')

    train = np.array(train)
    np.random.shuffle(train)

    #Separating training set into X(Pixels) and Y(Labels/answers)
    X_train = train[:, 1:]
    Y_train = train[:, 0]

    #Transposing and normalizing for dot product in forward propagation
    X_train = (X_train / 255).T
    Y_train = one_hot(Y_train,10)

    #Defining the layers, 784 inputs (for 784 pixels), 128 simple features, 64 complex, 10 output categories
    layer_dims = [784, 128, 64, 10]

    model = DeepNeuralNetwork(layer_dims)

    learning_rate = 0.1
    epochs = 2000
    costs = []

    for i in range(epochs):
        AL, caches = model.forward_propagation(X_train)

        cost = model.compute_cost(AL, Y_train)
        costs.append(cost)
        
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

        grads = model.backward_propagation(AL, Y_train, caches)

        model.update_parameters(grads, learning_rate)
    
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title("Learning Curve")
    plt.savefig('training_curve.png')
    print("Training curve saved to training_curve.png")

    #Separating test set into X(Pixels) and Y(Labels/answers)
    test = np.array(test)
    X_test = test[:, 1:]
    Y_test = test[:, 0]

    #Transposing and normalizing for dot product in forward propagation
    X_test = (X_test / 255).T

    predictions = model.predict(X_test)

    accuracy = np.mean(predictions == Y_test)
    

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

