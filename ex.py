import numpy as np
import random

def sigmoid(z, derivative = False):
    if not derivative:
        return 1 / (1 + np.exp(-z))
    else:
        return z * (1-z)

def cross_entropy_CF(outputs, targets, derivative = False):
    if not derivative:
        return -np.mean(outputs * np.log(targets) + (1 - outputs) * np.log(1 - targets))
    else :
        return (outputs - targets) / (outputs * (1 - targets))

X_train = np.asarray([[1, 2]]).T
Y_train = np.asarray([[2, 0, 3]]).T

hidden_size = 1
output_size = 3
learning_rate = 0.1
random.seed(42)
w1 = np.random.randn(hidden_size, 2) * 0.1
b1 = np.zeros((hidden_size, 1))
w2 = np.random.randn(output_size, hidden_size) * 0.1
b2 = np.zeros((output_size, 1))

for i in range(1):
  # forward pass

  Z1 = np.dot(w1, X_train) + b1
  A1 = sigmoid(Z1)

  Z2 = np.dot(w2, A1) + b2
  A2 = sigmoid(Z2)



  cost = -np.mean(Y_train * np.log(A2) + (1 - Y_train) * np.log(1 - A2))


  # backward pass

  dA2 = (A2 - Y_train) / (A2 * (1 - A2))

  dZ2 = np.multiply(dA2, A2 * (1 - A2))

  dw2 = np.dot(dZ2, A1.T)

  db2 = np.sum(dZ2, axis=1, keepdims=True)

  dA1 = np.dot(w2.T, dZ2)
  dZ1 = np.multiply(dA1, A1 * (1 - A1))
  dw1 = np.dot(dZ1, X_train.T)
  db1 = np.sum(dZ1, axis=1, keepdims=True)
  W = [dw2, dw1]
  B = [db2, db1]
  print(W, B)
  w1 = w1 - learning_rate * dw1
  w2 = w2 - learning_rate * dw2
  b1 = b1 - learning_rate * db1
  b2 = b2 - learning_rate * db2
