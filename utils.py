import numpy as np


# evaluation
def evaluate(data, Y):  # type:(np.array, np.array) -> list[float]
    """
    Evaluate the error between the given Y and the original data
    :param data: The original data
    :param Y: The model trained data
    :return: relative MAE, relative MSE, residual error, MAE, MSE
    """
    rmae = abs(data - Y).sum() / abs(data).sum()
    rmse = ((data - Y) ** 2).sum() / (data ** 2).sum()
    residual_error = np.linalg.norm(data - Y) / np.linalg.norm(data)
    mae = abs(data - Y).sum() / np.prod(data.shape)
    mse = ((data - Y) ** 2).sum() / np.prod(data.shape)
    return [rmae, rmse, residual_error, mae, mse]


def count_group(bb, tau):  # type:(np.array, float) -> [int]
    """
    count the number of groups in a row of the matrix
    :param bb: the given array of beta
    :param tau: the interval of two groups
    :return: the number of groups
    """
    beta = bb.copy()
    beta.sort()
    difference = beta[1:] - beta[:-1]
    groups = 1
    dif = 0
    for d in difference:
        if dif + d > tau:
            groups += 1
            dif = 0
        else:
            dif += d
    return groups


def group_evaluation(U, V, tau):  # type: (np.array, np.array, float) -> list(float)
    """
    count the mean and std of two given matrices U, and V
    :param U:
    :param V:
    :param tau: the interval of two groups
    :return: the mean and std of two given matrices U, and V
    """
    tmp = []
    for i in range(U.shape[0]):
        tmp.append(count_group(U[i, :], tau))
    for j in range(V.shape[0]):
        tmp.append(count_group(V[j, :], tau))
    return [np.mean(tmp), np.std(tmp)]


def sparsity_evaluation(U, V):  #type: (np.array, np.array) -> float
    """
    count the proportion of zeros of two matrix
    :param U:
    :param V:
    :return: the sparsity of two matrices
    """
    U_size, V_size = np.prod(U.shape), np.prod(V.shape)
    U_zero = U_size - np.count_nonzero(U)
    V_zero = V_size - np.count_nonzero(V)
    return (U_zero + V_zero) / (U_size+V_size)


# adding outliers
def add_outlier(data, outlier_size=0.2):  # type:(np.array, float) -> np.array
    """
    add the pepper&salt error to the given dataset
    :param data:
    :param outlier_size:
    :return:
    """
    # pepper&salt
    new_data = data.copy()
    mask = np.random.choice([0, 1], size=data.shape, p=[1 - outlier_size, outlier_size]).astype(np.bool)
    ps = np.random.choice([0, 1], size=data.shape, p=[0.5, 0.5])
    ps = ps * 255
    new_data[mask] = ps[mask]
    return new_data