import numpy as np
import random
import graph
from math import floor

def cnn(nn_layers, negatives):
    weights = []
    biases = []
    for layer in range(0, len(nn_layers)):
        if layer + 1 < len(nn_layers):
            right_layer_neurons_count = nn_layers[layer+1]
            left_layer_neurons_count = nn_layers[layer]
            if negatives :
                w = np.random.randn(right_layer_neurons_count, left_layer_neurons_count) * 0.1
            else :
                w = np.random.rand(right_layer_neurons_count, left_layer_neurons_count)
            b = np.zeros((right_layer_neurons_count, 1))
            weights.append(w)
            biases.append(b)

    return weights, biases


def ReLU(x, derivative = False):
    if derivative:
        return 1. * (x > 0)
    else:
        return x * (x > 0)

def sigmoid(z, derivative = False):
    if not derivative:
        return 1 / (1 + np.exp(-z))
    else:
        return z * (1-z)

def cross_entropy_CF(outputs, targets, derivative = False):
    if not derivative:
        return -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
    else :
        return (outputs - targets) / (outputs * (1 - targets))


def forward(inputs_batch, weights, biases, activation_function = ReLU):
    #activation_function = globals()[activation_function]()
    all_layers_activated_values = []

    all_layers_activated_values.append(inputs_batch.transpose())

    nn_layers = len(weights)

    for layer in range(0, nn_layers):

        I = all_layers_activated_values[layer]
        W = weights[layer]
        B = biases[layer]
        Z = np.dot(W, I) + B
        A = activation_function(Z)
        all_layers_activated_values.append(A)

    return all_layers_activated_values

def squared_error_CF(output, target, derivative = False):
    if derivative:
        result = output - target
    else:
        result = 0.5 * (output-target)**2
    return result

def backward(activated_values, targets, weights, activation_function, cost_function):
    outputs = activated_values[-1]
    targets = targets.transpose()
    RMSE = np.sqrt(np.mean(cost_function(outputs, targets)))

    dY_dOutNet = cost_function(outputs, targets, True)
    dOutNet_dOutBrut = activation_function(outputs, True)
    dY_dOutBrut = np.multiply(dY_dOutNet, dOutNet_dOutBrut)

    dZ = dY_dOutBrut
    weight_gradients = []
    bias_gradients = []

    for layer in range(2, len(weights)+2):

        layer *= -1
        #obtain gradients weights and biases
        A = activated_values[layer]

        W = np.dot(dZ, A.transpose())
        weight_gradients.insert(0,W)
        B = np.sum(dZ, axis=1, keepdims=True)
        bias_gradients.insert(0,B)

        #calculate next backward inputs
        dA = np.dot(weights[layer + 1].transpose(), dZ)
        dZ = np.multiply(dA, activation_function(A, True))

    # print(bias_gradients)
    return weight_gradients, bias_gradients, RMSE

def update(weights, weight_gradients, biases, bias_gradients, learning_rate, datapoints_count):
    for index in range(0, len(weights)):
        weights[index] -= weight_gradients[index]/datapoints_count * learning_rate
    for index in range(0, len(biases)):
        biases[index] -= bias_gradients[index]/datapoints_count * learning_rate
    return weights, biases


def train(data,
          targets,
          layers,
          batch_size = 100,
          negatives = False,
          learning_rate = 0.00001,
          epochs = 100,
          activation_function=ReLU,
          cost_function=squared_error_CF):
    weights, biases = cnn(layers, negatives)
    #batch management
    full_batches = floor(len(data)/batch_size)
    remainder_data_points = len(data) % batch_size

    full_batches_plus_remainder_batch = full_batches

    if remainder_data_points > 0:
        full_batches_plus_remainder_batch = full_batches + 1

    for epoch in range(0,epochs):
        print("Epoch " + str(epoch + 1) + "/" + str(epochs))
        for batch in range(0, full_batches_plus_remainder_batch):
            #batch management
            start_index = batch * batch_size
            end_index = start_index + batch_size
            if batch == full_batches:
                end_index = remainder_data_points
            data_set = data[start_index:end_index]
            target_set = targets[start_index:end_index]

            #shuffle the data and corresponding targets
            c = list(zip(data_set, target_set))
            random.shuffle(c)
            data_set, target_set = zip(*c)

            activated_values = forward(data, weights, biases, activation_function)
            weight_gradients, bias_gradients, RMSE = backward(activated_values, targets, weights, activation_function, cost_function)

            weights, biases = update(weights, weight_gradients, biases, bias_gradients, learning_rate, datapoints_count = (end_index-start_index))
        print("RMSE: " + str(RMSE))

    # print("Creating NN graph...")
    # graph_name = activation_function.__name__ + " - Epoch " + str(epochs) + ' @ lr ' + str(learning_rate)


    return weights, biases

def createData(lower, upper, points, number):
    data = []
    for integer in range(0, number):
        datapoint = []
        for point in range(0, points):
            datapoint.append(random.randint(lower,upper))
        data.append(datapoint)
    return np.array(data)

def createTargets(data, function):
    targets  = []
    for datapoint in data:
        targets.append(function(datapoint))
    return np.array(targets)

def quadratic(value, intercept=0):
    return [value[0]**2 + intercept]

random.seed(42)

data = createData(1, 10, 1, 1000)
targets = createTargets(data, quadratic)

weights, biases = train(data,
                        targets,
                        layers = [1, 4, 4, 1],
                        negatives = False,
                        batch_size = 8,
                        learning_rate = 0.000001,
                        epochs = 1000,
                        activation_function = ReLU,
                        cost_function = squared_error_CF
                        )
print(forward(np.array([[13]]), weights, biases)[-1])
