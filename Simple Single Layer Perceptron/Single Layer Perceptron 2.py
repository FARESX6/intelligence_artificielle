 import numpy as np  

def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x)*(1-sigmoid(x))

# CREATE DATA SET
X = np.array([[0,1,0], [0,0,1], [1,0,0], [1,1,0], [1,1,1]])  
y = np.array([[1,0,0,1,1]])  
y = y.reshape(5,1)  

# HYPERPARAMETERS
np.random.seed(42)  
weights = np.random.rand(3, 1)  
bias = np.random.rand(1)  
lr = 0.05        # learning rate
nb_epoch = 20000

m = len(X)

# TRAINING
for epoch in range(nb_epoch):  

    # FEEDFORWARD
    z = np.dot(X, weights) + bias  # (5x1)
    a = sigmoid(z)  # (5X1)

    # BACKPROPAGATION
    # 1. cost function: MSE
    J = (1/m)*(a - y)**2   # (5X1)
    print(J.sum())

    # 2. weights
    dJ_da = (2/m)*(a-y) 
    da_dz = sigmoid_der(z)
    dz_dw = X.T

    gradient_w = np.dot(dz_dw, da_dz*dJ_da)  # chain rule 
    weights -= lr * gradient_w               # gradient descent

    # 3. bias
    gradient_b = da_dz*dJ_da   # chain rule
    bias -= lr*sum(gradient_b)  # gradient descent


# TEST PHASE
example1 = np.array([1,0,0])  
result1 = sigmoid(np.dot(example1, weights) + bias)  
print(result1.round())
print('A person who is smoking, not obese and does not exercise is classified as not diabetic.')

example2 = np.array([0,1,0])  
result2 = sigmoid(np.dot(example2, weights) + bias)  
print(result2.round())
print('A person who is not smoking, obese and does not exercise is classified as diabetic.')