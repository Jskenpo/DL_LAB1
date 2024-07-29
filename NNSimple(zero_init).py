import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.zeros((n_h, n_x))
    b1 = np.zeros((n_h, 1))
    W2 = np.zeros((n_y, n_h))
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def forward_prop(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"A1": A1, "A2": A2}
    return A2, cache

def loss_function(A2, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2)))/m
    return np.squeeze(cost)

def backward_prop(X, Y, cache, parameters):
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]
    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    W1 = W1 - learning_rate * grads["dW1"]
    b1 = b1 - learning_rate * grads["db1"]
    W2 = W2 - learning_rate * grads["dW2"]
    b2 = b2 - learning_rate * grads["db2"]
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_of_iters):
        a2, cache = forward_prop(X, parameters)
        cost = loss_function(a2, Y)
        grads = backward_prop(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            costs.append(cost)
            print(f'Cost after iteration# {i}: {cost}')

    return parameters, costs

# Set up the data
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

# Set the hyperparameters
n_x, n_h, n_y = 2, 2, 1
num_of_iters = 10000
learning_rate = 0.1

# Train with zero initialization
zero_parameters, zero_costs = model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate)

# Plot the cost
plt.figure(figsize=(10, 6))
plt.plot(range(0, num_of_iters, 100), zero_costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations (Zero Initialization)')
plt.show()

# Test the model
X_test = np.array([[1], [1]])
zero_prediction, _ = forward_prop(X_test, zero_parameters)

print(f"Prediction with zero initialization: {zero_prediction}")

# Print final parameters
print("\nFinal Parameters:")
for key, value in zero_parameters.items():
    print(f"{key}:\n{value}")