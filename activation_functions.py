from math import exp

def leakyReLU(value, derivative = False, alpha = 0.01):
    if derivative:
        result = 1
        if value < 0:
            result = alpha
    else:
        result = value
        if value < 0:
            result = value * alpha
    return result

def sigmoid(value, derivative = False):
    if derivative:
        result = exp(value)/((1+exp(-value))**2)
    else:
        result = 1/(1+exp(-value))
    return result

def identity(value, derivative = False):
    if derivative:
        result = 1
    else :
        result = value
    return result
