import numpy as np
from math import *
from numba import njit

#####FUNCTION#####
(
    FUNCTION_BINARY_STEP,
    FUNCTION_LINEAR,
    FUNCTION_SIGMOID,
    FUNCTION_TANH,
    FUNCTION_RELU,
    FUNCTION_LEAKY_RELU,
    FUNCTION_PARAMETERISED_RELU,
    FUNCTION_EXPONENTIAL_LINEAR_UNIT
) = range(8)
##################

def ActivationFunction(functionType: int, z: float, prime: bool = False, alpha: float = 1) -> float:
    """
        type : "FUNCTION_#####"
        z : Pre-activation
        prime : True/False
        alpha : Default(1)
    Funtion :
        FUNCTION_BINARY_STEP (z)
        FUNCTION_LINEAR (z, alpha)
        FUNCTION_SIGMOID (z)
        FUNCTION_TANH (z)
        FUNCTION_RELU (z)
        FUNCTION_LEAKY_RELU (z, alpha)
        FUNCTION_PARAMETERISED_RELU (z, alpha)
        FUNCTION_EXPONENTIAL_LINEAR_UNIT (z, alpha)
    """
    y = 0
    if functionType == FUNCTION_BINARY_STEP:
        if not prime:
            if z < 0: y = 0
            else: y = 1
        else: 
            # pas de deriver
            pass
    if functionType == FUNCTION_LINEAR:
        if not prime:       y = z*alpha
        else:               y = alpha
    if functionType == FUNCTION_SIGMOID:
        if not prime:       y = 1/(1+np.exp(-z))
        else:               y = (1/(1+np.exp(-z))) * (1-(1/(1+np.exp(-z))))
    if functionType == FUNCTION_TANH:
        if not prime:       y = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        else:               y = 1 - (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))**2
    if functionType == FUNCTION_RELU:
        if not prime:       y = np.max(0,z)
        else:
            if z >= 0:          y = 1
            else:               y = 0
    if functionType == FUNCTION_LEAKY_RELU:
        if not prime:       y = np.max(alpha*z, z)
        else:
            if z > 0:           y = 1
            else:               y = alpha
    if functionType == FUNCTION_PARAMETERISED_RELU:
        if not prime:
            if z >= 0:      y = z
            else:           y = alpha*z
        else:
            if z >= 0:          y = 1
            else:               y = alpha
    if functionType == FUNCTION_EXPONENTIAL_LINEAR_UNIT:
        if not prime:
            if z >= 0:      y = z
            else:           y = alpha*(np.exp(z)-1)
        else:
            if z >= 0:          y = z
            else:               y = alpha*(np.exp(y))
    return y


@njit
def ActivationFunctionGPU(functionType: int, z: np.float64, alpha: np.float64, prime: bool = False) -> np.float64:
    """
        type : "FUNCTION_#####"
        z : Pre-activation
        prime : True/False
        alpha : Default(1)
    Funtion :
        FUNCTION_BINARY_STEP (z)
        FUNCTION_LINEAR (z, alpha)
        FUNCTION_SIGMOID (z)
        FUNCTION_TANH (z)
        FUNCTION_RELU (z)
        FUNCTION_LEAKY_RELU (z, alpha)
        FUNCTION_PARAMETERISED_RELU (z, alpha)
        FUNCTION_EXPONENTIAL_LINEAR_UNIT (z, alpha)
    """
    y = z
    if functionType == FUNCTION_BINARY_STEP:
        if not prime:
            if z < 0: y = 0
            else: y = 1
        else: 
            # pas de deriver
            pass
    if functionType == FUNCTION_LINEAR:
        if not prime:       y = z*alpha
        else:               y = alpha
    if functionType == FUNCTION_SIGMOID:
        if not prime:       y = 1/(1+np.exp(-z))
        else:               y = (1/(1+np.exp(-z))) * (1-(1/(1+np.exp(-z))))
    if functionType == FUNCTION_TANH:
        if not prime:       y = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        else:               y = 1 - (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))**2
    if functionType == FUNCTION_RELU:
        if not prime:       
            if z > 0.0:
                y = z
            else:
                y = 0.0
        else:
            if z >= 0:          y = 1
            else:               y = 0
    if functionType == FUNCTION_LEAKY_RELU:
        if not prime:
            if alpha*z > z:
                y = alpha*z
            else:
                y = z
        else:
            if z > 0:           y = 1
            else:               y = alpha
    if functionType == FUNCTION_PARAMETERISED_RELU:
        if not prime:
            if z >= 0:      y = z
            else:           y = alpha*z
        else:
            if z >= 0:          y = 1
            else:               y = alpha
    if functionType == FUNCTION_EXPONENTIAL_LINEAR_UNIT:
        if not prime:
            if z >= 0:      y = z
            else:           y = alpha*(np.exp(z)-1)
        else:
            if z >= 0:          y = z
            else:               y = alpha*(np.exp(y))
    return y

    