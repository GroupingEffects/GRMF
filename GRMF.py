import numpy as np
from abc import ABCMeta, abstractmethod
import multiprocessing

# global variable name
TAU = "tau"
TAU_1 = 'tau_1'
TAU_2 = 'tau_2'
LAMBDA0 = 'lambda_0'
LAMBDA1 = "lambda_1"
LAMBDA2 = "lambda_2"
LAMBDA3 = 'lambda_3'
INNTER_TOL = "inner_tol"
OUTER_TOL = "outer_tol"
APPROX_TOL = 'approx_tol'
K_ITERS = "k_iters"
M_ITERS = "m_iters"
V_ALPHA = 'v_alpha'
V_INIT = 'v_init'

np.random.seed(42)
num_cores = int(multiprocessing.cpu_count())


# Serializing GRMF hyper parameters
class hyper_param(object):
    """
    Serializing algorithm hyper-parameters
    """
    def __init__(self, **kwargs):
        self.tau = kwargs[TAU] if TAU in kwargs else 1
        self.tau_1 = kwargs[TAU_1] if TAU_1 in kwargs else 1
        self.tau_2 = kwargs[TAU_2] if TAU_2 in kwargs else 1
        self.lambda_0 = kwargs[LAMBDA0] if LAMBDA0 in kwargs else 0.1
        self.lambda_1 = kwargs[LAMBDA1] if LAMBDA1 in kwargs else 0.1
        self.lambda_2 = kwargs[LAMBDA2] if LAMBDA2 in kwargs else 0.1
        self.lambda_3 = kwargs[LAMBDA3] if LAMBDA3 in kwargs else 0.1
        self.inner_tol = kwargs[INNTER_TOL] if INNTER_TOL in kwargs else 1e-4
        self.outer_tol = kwargs[OUTER_TOL] if OUTER_TOL in kwargs else 1e-2
        self.approx_tol = kwargs[APPROX_TOL] if APPROX_TOL in kwargs else 1e-5
        self.k_iters = int(kwargs[K_ITERS]) if K_ITERS in kwargs else 100
        self.m_iters = int(kwargs[M_ITERS]) if M_ITERS in kwargs else 100
        self.v_alpha = kwargs[V_ALPHA] if V_ALPHA in kwargs else 1.1
        self.v_init = kwargs[V_INIT] if V_INIT in kwargs else 1.5

    @property
    def to_json(self):
        """
        serialize
        :return:
        """
        json_data = {
            TAU: self.tau,
            TAU_1: self.tau_1,
            TAU_2: self.tau_2,
            LAMBDA0: self.lambda_0,
            LAMBDA1: self.lambda_1,
            LAMBDA2: self.lambda_2,
            LAMBDA3: self.lambda_3,
            INNTER_TOL: self.inner_tol,
            OUTER_TOL: self.outer_tol,
            APPROX_TOL: self.approx_tol,
            K_ITERS: self.k_iters,
            M_ITERS: self.m_iters,
            V_ALPHA: self.v_alpha,
            V_INIT: self.v_init
        }
        return json_data

    @classmethod
    def from_json(cls, data):
        """

        :param data:
        :return:
        """
        return cls(**data)


# Define the uniform structure for the algorithm
class DC_algorithems(object):
    """
    Build a base class to support the processing of diverse and potentially novel objects in a uniform way
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def loss_function(self, beta, F, E, N):
        pass

    @abstractmethod
    def inner_loop(self, beta_m, beta_jj_m, F, E, N):
        pass

    @abstractmethod
    def fit(self):
        pass

    @staticmethod
    def soft_threshold(b, delta):
        """Soft threshold function used for normalized data and lasso regression"""
        if b < - delta:
            return b + delta
        elif b > delta:
            return b - delta
        else:
            return 0


# Approximated L1 version of DC algorithm (For GRMF)
class approx_LS_L1(DC_algorithems):
    """Approximated L1 version DC algorithm"""
    def __init__(self, X, y, beta_init, param=dict()):
        self.X = X
        self.y = y
        self.param = hyper_param.from_json(param)
        self.n, self.p = X.shape
        self.beta_init = beta_init

    def loss_function(self, beta, F, E, N):
        p1s = np.arange(self.p)
        p2s = p1s.reshape(-1, 1)

        predict = self.y - self.X @ beta
        p0 = 1 / self.n * sum(np.sqrt(predict * predict + np.ones(self.n) * self.param.approx_tol))

        p1_temp = beta.copy()
        p1_temp[F == 0] = self.param.tau_1
        p1 = self.param.lambda_1 / self.param.tau_1 * sum(p1_temp)

        p2_temp = abs(beta.reshape(-1, 1) - beta)
        p2_temp[E == 0] = self.param.tau_2
        p2_temp[p1s <= p2s] = 0
        p2_temp = p2_temp.reshape(-1, 1)
        p2 = self.param.lambda_2 / self.param.tau_2 * (sum(p2_temp))

        p3_temp = beta.copy()
        p3_temp[N == 0] = 0
        p3 = self.param.lambda_3 * (p3_temp @ p3_temp)

        return p0 + p1 + p2 + p3

    def inner_loop(self, beta_m, beta_jj_m, F, E, N):
        beta = beta_m
        beta_jj = beta_jj_m

        old_beta = beta.copy()

        Tau = np.ones([self.p, self.p])

        for k in range(self.param.k_iters):
            A = (self.y - self.X @ beta) * (self.y - self.X @ beta) + np.ones(self.n) * self.param.approx_tol
            A_invsqrt = A ** (-0.5)
            for j in range(self.p):
                alpha = 1 / self.n * ((A_invsqrt * self.X[:, j]) @ self.X[:, j]) +\
                        self.param.v_init * E[j,] @ np.ones(self.p) + 2 * self.param.lambda_3 * N[j] + 1e-10
                b = self.y - self.X @ beta + self.X[:, j] * beta[j]
                gamma_star = 1 / self.n * ((A_invsqrt * self.X[:, j]) @ b) - Tau[j,] @ E[j,] +\
                             self.param.v_init * (beta + beta_jj[j,]) @ E[j,]
                if F[j] == 0:
                    beta[j] = (1 / alpha) * gamma_star
                else:
                    beta[j] = 1 / alpha * self.soft_threshold(gamma_star, self.param.lambda_1 / self.param.tau_1)

            if np.mean(abs(beta - old_beta)) < self.param.inner_tol:
                return beta

            beta_jj = Tau + self.param.v_init * (beta.reshape(-1, 1) - beta)
            # soft-thresholding
            m1 = beta_jj > self.param.lambda_2 / self.param.tau_2
            m2 = beta_jj < -self.param.lambda_2 / self.param.tau_2
            m3 = abs(beta_jj) < self.param.lambda_2 / self.param.tau_2
            beta_jj[m1] = beta_jj[m1] - self.param.lambda_2 / self.param.tau_2
            beta_jj[m2] = beta_jj[m2] + self.param.lambda_2 / self.param.tau_2
            beta_jj[m3] = 0
            beta_jj = 1 / self.param.v_init * beta_jj
            # not updating beta_jj' if (j,j') not in E^(m-1)
            beta_jj[E == 0] = beta_jj_m[E == 0]
            Tau = Tau + self.param.v_init * (beta.reshape(-1, 1) - beta - beta_jj)
            self.param.v_init = self.param.v_init * self.param.v_alpha
            old_beta = beta.copy()
        return beta

    def fit(self):
        beta = self.beta_init.copy()
        p1s = np.arange(self.p)
        p2s = p1s.reshape(-1, 1)
        old_beta = beta.copy()
        loss = []
        for m in range(self.param.m_iters):
            beta_jj = beta.reshape(-1, 1) - beta

            F = np.zeros(self.p)
            F[abs(beta) < self.param.tau_1] = 1

            E = np.zeros([self.p, self.p])
            E[abs(beta.reshape(-1, 1) - beta) < self.param.tau_2] = 1
            E[p1s == p2s] = 0

            N = np.zeros(self.p)
            N[beta < 0] = 1

            ls = self.loss_function(beta, F, E, N)
            loss.append(ls)

            beta = self.inner_loop(beta, beta_jj, F, E, N)

            if np.mean(abs(beta - old_beta)) < self.param.outer_tol:
                return [beta, loss]
            old_beta = beta.copy()
        return [beta, loss]


# L2 version DC algorithm (For GMF-L2)
class least_square_L2(DC_algorithems):
    def __init__(self, X, y, beta_init, param=dict()):
        self.X = X
        self.y = y
        self.param = hyper_param.from_json(param)
        self.n, self.p = X.shape
        self.beta_init = beta_init

    def loss_function(self, beta, F, E, N):
        p1s = np.arange(self.p)
        p2s = p1s.reshape(-1, 1)

        predict = self.y - self.X @ beta
        p0 = 1 / 2 / self.n * predict @ predict

        p1_temp = beta.copy()
        p1_temp[F == 0] = self.param.tau_1
        p1 = self.param.lambda_1 / self.param.tau_1 * sum(p1_temp)

        p2_temp = abs(beta.reshape(-1, 1) - beta)
        p2_temp[E == 0] = self.param.tau_2
        p2_temp[p1s <= p2s] = 0
        p2_temp = p2_temp.reshape(-1, 1)
        p2 = self.param.lambda_2 / self.param.tau_2 * (sum(p2_temp))

        p3_temp = beta.copy()
        p3_temp[N == 0] = 0
        p3 = self.param.lambda_3 * (p3_temp @ p3_temp)

        return p0 + p1 + p2 + p3

    def inner_loop(self, beta_m, beta_jj_m, F, E, N,):
        beta = beta_m
        beta_jj = beta_jj_m

        old_beta = beta.copy()

        Tau = np.ones([self.p, self.p])

        for k in range(self.param.k_iters):
            for j in range(self.p):
                alpha = 1 / self.n * self.X[:, j] @ self.X[:, j] + self.param.v_init * E[j, :].sum() + \
                        2 * self.param.lambda_3 * N[j]
                b = self.y - self.X @ beta + self.X[:, j] * beta[j]
                gamma_star = 1 / self.n * self.X[:, j] @ b - Tau[j, :] @ E[j, :] + \
                             self.param.v_init * (beta + beta_jj[j, :]) @ E[j, :]
                if F[j] == 0:
                    beta[j] = 1 / alpha * gamma_star
                else:
                    beta[j] = 1 / alpha * self.soft_threshold(gamma_star, self.param.lambda_1 / self.param.tau_1)

            if np.mean(abs(beta - old_beta)) < self.param.inner_tol:
                return beta

            beta_jj = Tau + self.param.v_init * (beta.reshape(-1, 1) - beta)
            # soft-thresholding
            m1 = beta_jj > self.param.lambda_2 / self.param.tau_2
            m2 = beta_jj < -self.param.lambda_2 / self.param.tau_2
            m3 = abs(beta_jj) < self.param.lambda_2 / self.param.tau_2
            beta_jj[m1] = beta_jj[m1] - self.param.lambda_2 / self.param.tau_2
            beta_jj[m2] = beta_jj[m2] + self.param.lambda_2 / self.param.tau_2
            beta_jj[m3] = 0
            beta_jj = 1 / self.param.v_init * beta_jj
            # not updating beta_jj' if (j,j') not in E^(m-1)
            beta_jj[E == 0] = beta_jj_m[E == 0]

            Tau = Tau + self.param.v_init * (beta.reshape(-1, 1) - beta - beta_jj)
            self.param.v_init = self.param.v_init * self.param.v_alpha
            old_beta = beta.copy()
        return beta

    def fit(self):
        beta = self.beta_init.copy()
        p1s = np.arange(self.p)
        p2s = p1s.reshape(-1, 1)
        old_beta = beta.copy()
        loss = []
        for m in range(self.param.m_iters):
            beta_jj = beta.reshape(-1, 1) - beta

            F = np.zeros(self.p)
            F[abs(beta) < self.param.tau_1] = 1

            E = np.zeros([self.p, self.p])
            E[abs(beta.reshape(-1, 1) - beta) < self.param.tau_2] = 1
            E[p1s == p2s] = 0

            N = np.zeros(self.p)
            N[beta < 0] = 1

            ls = self.loss_function(beta, F, E, N)
            loss.append(ls)

            beta = self.inner_loop(beta, beta_jj, F, E, N,)
            if np.mean(abs(beta - old_beta)) < self.param.outer_tol:
                return [beta, loss]
            old_beta = beta.copy()
        return [beta, loss]


# multiprocessing version to accelerate calculation
def train_on_parameter(name, param, hyper_param, result_dict, result_lock, version='L1'):
    """
    The processor for the multiprocessing. It decide whether to use L1 version or L2 version
    :param name:
    :param param:
    :param hyper_param:
    :param result_dict:
    :param result_lock:
    :param version: L1 for GRMF, L2 for GMF-L2
    :return:
    """
    if version != 'L1':
        tmp = least_square_L2(param[0], param[1], param[2], hyper_param)
        result = tmp.fit()[0]
    else:
        result = approx_LS_L1(param[0], param[1], param[2], hyper_param).fit()[0]
    with result_lock:
        result_dict[name] = result
    return


def multiprocess_V(U, V, data, V_init, hyper_param, version='L1'):
    """
    Processing V in parallel. Distribute each row of V to a core of CPU and recollect all the results
    :param U:
    :param V:
    :param data:
    :param V_init: Initialization of V
    :param hyper_param:
    :param version: L1 for GRMF, L2 for GMF-L2
    :return:
    """
    param_dict = {}
    for i in range(data.shape[1]):
        param_dict[str(i)] = [U, data[:, i], V_init[i, :]]
    pool = multiprocessing.Pool(num_cores)
    manager = multiprocessing.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()

    results = [pool.apply_async(train_on_parameter,
                                args=(name, param, hyper_param, managed_dict, managed_locker, version))
               for name, param in param_dict.items()]
    results = [p.get() for p in results]
    for k, v in managed_dict.items():
        V.T[:, int(k)] = v
    return V


def multiprocess_U(U, V, data, U_init, hyper_param, version='L1'):
    """
    Processing V in parallel. Distribute each row of U to a core of CPU and recollect all the results
    :param U:
    :param V:
    :param data:
    :param U_init:
    :param hyper_param:
    :param version: L1 for GRMF, L2 for GMF-L2
    :return:
    """
    param_dict = {}
    for i in range(data.shape[0]):
        param_dict[str(i)] = [V, data[i, :], U_init[i, :]]
    pool = multiprocessing.Pool(num_cores)
    manager = multiprocessing.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()

    results = [pool.apply_async(train_on_parameter,
                                args=(name, param, hyper_param, managed_dict, managed_locker, version))
               for name, param in param_dict.items()]
    results = [p.get() for p in results]
    for k, u in managed_dict.items():
        U[int(k), :] = u
    return U


def multiprocess_GRMF(data, rank, tol, max_iter=100, param=dict(), version='L1'):
    """

    :param data:
    :param rank:
    :param tol:
    :param max_iter:
    :param param:
    :param version: L1 for GRMF, L2 for GMF-L2
    :return:
    """
    loss = list()
    # initialize two low-rank matrices
    svd_u, svd_s, svd_vh = np.linalg.svd(data)
    U = svd_u[:, :rank] @ np.diag(np.sqrt(svd_s[:rank]))
    V = (np.diag(np.sqrt(svd_s[:rank])) @ svd_vh[:rank, :]).T
    U_init, V_init = U.copy(), V.copy()
    iter = 0
    while (abs(data - np.matmul(U, V.T))).sum() > tol:
        iter += 1
        # fix U to solve V
        V = multiprocess_V(U, V, data, V_init, param, version)
        # fix V to solve U
        U = multiprocess_U(U, V, data, U_init, param, version)
        if iter >= max_iter:
            break
    return U, V, loss


def GRMF(data, rank, tol, max_iter=10, param=dict()):
    """

    :param data:
    :param rank:
    :param tol:
    :param max_iter:
    :param param:
    :return:
    """
    loss = list()
    # initialize two low-rank matrices
    d, n = data.shape[0], data.shape[1]
    svd_u, svd_s, svd_vh = np.linalg.svd(data)
    U = svd_u[:, :rank] @ np.diag(np.sqrt(svd_s[:rank]))
    V = (np.diag(np.sqrt(svd_s[:rank])) @ svd_vh[:rank, :]).T
    U_init, V_init = U.copy(), V.copy()
    iter = 0
    while (abs(data - np.matmul(U, V.T))).sum() > tol:
        iter += 1
        # fix U to solve V
        for j in range(n):
            # solve the model of data[:, j] = U %*% V.T[:, j]
            V.T[:, j] = approx_LS_L1(U, data[:, j], V_init.T[:, j], param).fit()[0]
        # fix V to solve U
        for i in range(d):
            # solve the model of data[j, :] = U[j, :] %*% V.T
            U[i, :] = approx_LS_L1(V, data[i, :], U_init[i, :], param).fit()[0]
        loss.append((abs(data - np.matmul(U, V.T))).sum())
        if iter >= max_iter:
            break
    return U, V, loss


def GRMF_L2(data, rank, tol, max_iter=100, param=dict()):
    """

    :param data:
    :param rank:
    :param tol:
    :param max_iter:
    :param param:
    :return:
    """
    loss = list()
    # initialize two low-rank matrices
    d, n = data.shape[0], data.shape[1]
    svd_u, svd_s, svd_vh = np.linalg.svd(data)
    U = svd_u[:, :rank] @ np.diag(np.sqrt(svd_s[:rank]))
    V = (np.diag(np.sqrt(svd_s[:rank])) @ svd_vh[:rank, :]).T
    U_init, V_init = U.copy(), V.copy()
    iter = 0
    while (abs(data - np.matmul(U, V.T))).sum() > tol:
        iter += 1
        # fix U to solve V
        for j in range(n):
            # solve the model of data[:, j] = U %*% V.T[:, j]
            V.T[:, j] = least_square_L2(U, data[:, j], V_init.T[:, j], param).fit()[0]
        # fix V to solve U
        for i in range(d):
            # solve the model of data[j, :] = U[j, :] %*% V.T
            U[i, :] = least_square_L2(V, data[i, :], U_init[i, :], param).fit()[0]
        loss.append((abs(data - np.matmul(U, V.T))).sum())
        if iter >= max_iter:
            break
    return U, V, loss
