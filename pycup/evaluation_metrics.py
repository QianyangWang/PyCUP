import numpy as np
import math

def OneMinusNSE(real_data,pred_data):
    """
    One minus the Nash-Sutcliffe efficiency coefficient (NSE)

    :argument
    real_data: observed data array -> array like, np.array will be better
    pred_data: predicted data array -> array like, np.array will be better

    :return:
    1 - NSE: float

    Note:
    Here we use one minus NSE instead of NSE itself because the algorithms are minimizing the fitness values.
    """
    mean_obs = np.mean(real_data)
    upper = []
    lower = []
    for i in range(len(real_data)):
        upper.append((real_data[i]-pred_data[i])**2)
        lower.append((real_data[i]-mean_obs)**2)
    _upper = sum(upper)
    _lower = sum(lower)
    NSE = 1-(_upper/_lower)
    return 1 - NSE


def MAE(real_data,pred_data):
    """
    Mean absolute error

    :argument
    real_data: observed data array -> array like, np.array will be better
    pred_data: predicted data array -> array like, np.array will be better

    :return:
    mae: the mean absolute error value -> float
    """
    errors = []
    m= len(real_data)
    for i in range(m):
        errors.append(abs(real_data[i]-pred_data[i]))
    error = sum(errors)
    mae = error/m
    return mae


def MSE(real_data,pred_data):
    """
    Mean squared error

    :argument
    real_data: observed data array -> array like, np.array will be better
    pred_data: predicted data array -> array like, np.array will be better

    :return:
    mse: the mean squared error value
    """
    errors = []
    m= len(real_data)
    for i in range(m):
        errors.append((real_data[i]-pred_data[i])**2)
    error = sum(errors)
    mse = error/m
    return mse


def RMSE(real_data,pred_data):
    """
    Root mean squared error

    :argument
    real_data: observed data array -> array like, np.array will be better
    pred_data: predicted data array -> array like, np.array will be better

    :return:
    rmse: the mean squared error value
    """
    errors = []
    m= len(real_data)
    for i in range(m):
        errors.append((real_data[i]-pred_data[i])**2)
    error = sum(errors)
    rmse = math.sqrt(error/m)
    return rmse


def MAPE(real_data,pred_data):
    """
    Mean absolute error

    :argument
    real_data: observed data array -> array like, np.array will be better
    pred_data: predicted data array -> array like, np.array will be better

    :return:
    mape: the mean absolute error value -> float, unit = %
    """
    errors = []
    m= len(real_data)
    for i in range(m):
        errors.append(100 * abs(real_data[i]-pred_data[i])/real_data[i])
    error = sum(errors)
    mape = error/m
    return mape


def generational_distance(pareto_true,pareto_pred):
    dim1 = pareto_true.shape[1]
    dim2 = pareto_pred.shape[1]
    if dim1 != dim2:
        raise ValueError("The dimension of the real pareto front should be equal to the one of the predicted pareto front.")
    min_distance = []
    for i in pareto_pred:
        distances = []
        for j in pareto_true:
            dist = np.linalg.norm(i - j)
            distances.append(dist)
        min_dis = np.min(distances)
        min_distance.append(min_dis**2)

    GD = np.power(np.sum(min_distance),1/2)/pareto_pred.shape[0]
    return GD


def inverted_generational_distance(pareto_true,pareto_pred):
    dim1 = pareto_true.shape[1]
    dim2 = pareto_pred.shape[1]
    if dim1 != dim2:
        raise ValueError("The dimension of the real pareto front should be equal to the one of the predicted pareto front.")
    min_distances = []
    for i in pareto_true:
        distances = []
        for j in pareto_pred:
            dist = np.linalg.norm(i - j)
            distances.append(dist)
        min_dis = np.min(distances)
        min_distances.append(min_dis**2)
    IGD = np.power(np.sum(min_distances),1/2)/pareto_true.shape[0]
    return IGD


