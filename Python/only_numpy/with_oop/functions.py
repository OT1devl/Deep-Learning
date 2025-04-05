import numpy as np

def tanh(x, derv=False):
    if derv: return 1 - np.tanh(x)**2
    return np.tanh(x)

def derv_tanh(x):
    return 1 - x**2

def sigmoid(x, derv=False):
    s = 1 / (1 + np.exp(-x))
    if derv: return s * (1 - s)
    return s

def derv_sigmoid(x):
    return x * (1 - x)

def identity(x, derv=False):
    if derv: return 1
    return x

def derv_identity(x):
    return 1