import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF


# PMF
class PMF_class(object):
    def __init__(self, data,
                 num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.data = data
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self):
        # mean subtraction
        num_user, num_item = self.data.shape
        pairs_tr = num_user * num_item  # length of training data
        train_vec = np.zeros((pairs_tr, 3))
        tmp = np.array([[i for i in range(num_item)] * num_user]).flatten()
        train_vec[:, 0] = tmp
        tmp = np.array([[i] * num_item for i in range(num_user)]).flatten()
        train_vec[:, 1] = tmp
        tmp = self.data.flatten()
        train_vec[:, 2] = tmp
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

        # 1-p-i, 2-m-c
        num_user, num_item = self.data.shape

        incremental = False  # 增量
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  # 检查迭代次数
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  # mean_inv subtracted # np.multiply对应位置元素相乘

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc
        return self.w_Item, self.w_User

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)


def PMF(data, rank):
    pmf = PMF_class(data)
    num_batches = int(data.shape[0]/10)
    batch_size = int(np.prod(data.shape) / num_batches)
    pmf.set_params({"num_feat": rank, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10,
                    "num_batches": num_batches, "batch_size": batch_size})
    U, V = pmf.fit()
    return U, V


# RPMF
def RPMF(data, rank, lambdaU, lambdaV, tol):
    np.random.seed(1)
    maxIter = 100
    m, n = data.shape
    U = np.random.randn(m, rank)
    V = np.random.randn(rank, n)
    lambdaY = 1
    eps = 1e-3
    r = abs(data - np.matmul(U, V))
    # replace the element in r that is smaller than eps
    r = np.where(r < eps, eps, r)
    r = lambdaY / r
    IS = np.identity(rank)
    c = 0
    while c < maxIter:
        old_r = r.copy()
        # Update V
        for i in range(n):
            T = np.matmul(U.T, np.diagflat(r[:, i]))
            V[:, i] = np.linalg.solve((np.matmul(T, U) + lambdaV * IS), np.matmul(T, data[:, i]))
            r[:, i] = abs(data[:, i] - np.matmul(U, V[:, i]))
            # replace the element in r that is smaller than eps
            r[:, i] = np.where(r[:, i] < eps, eps, r[:, i])
            r[:, i] = lambdaY / r[:, i]
        # Update U
        for i in range(m):
            T = np.matmul(V, np.diagflat(r[i, :]))
            U[i, :] = np.linalg.solve((np.matmul(T, V.T) + lambdaU * IS), np.matmul(T, data[i, :].T))
            r[i, :] = abs(data[i, :].T - np.matmul(V.T, U[i, :].T))
            # replace the element in r that is smaller than eps
            r[i, :] = np.where(r[i, :] < eps, eps, r[i, :])
            r[i, :] = lambdaY / r[i, :]
        c += 1
        if (abs(r - old_r).sum()) / old_r.sum() < tol:
            break
    return U, V.T


# GoDec+
def GoDec_plus(X, rank, sigma, epsilon, q):
    iter_max = 100
    # rel_err = list()
    X = X.T
    m, n = X.shape[0], X.shape[1]
    T = np.zeros(X.shape)
    L = X

    iter = 1
    Y2 = np.random.rand(n, rank)

    while True:
        e = T - T * np.exp(-T * T / sigma)
        X1 = X - e
        # update of L
        for i in range(q):
            Y1 = np.matmul(X1, Y2)
            Y2 = np.matmul(X1.T, Y1)
        Q, R = np.linalg.qr(Y2)
        base = np.matmul(X1, Q)
        L_new = np.matmul(base, Q.T)
        Y2 = Q

        T = X - L_new
        L_diff = L_new - L
        stop_cri = (np.linalg.norm(L_diff) / np.linalg.norm(L)) ** 2
        # rel_err.append(stop_cri)
        if stop_cri < epsilon or iter > iter_max:
            break
        L = L_new
        iter += 1
    L = L_new.T
    return Q, base


# Truncated svd
def truncated_svd(data, rank):
    u, s, vh = np.linalg.svd(data)
    U = u[:, :rank] @ np.diag(np.sqrt(s[:rank]))
    V = (np.diag(np.sqrt(s[:rank])) @ vh[:rank, :]).T
    return U, V


# RPCA
class R_pca:
    def __init__(self, D, rank, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        # self.mu_inv = 1 / self.mu
        self.mu_inv = 1/rank
        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            # if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
            #     print('iteration: {0}, error: {1}'.format(iter, err))
        self.L = Lk
        self.S = Sk
        return Lk, Sk
        # return Yk


def RPCA(data, rank, mu=None, lmbda=None):
    rpca = R_pca(data, rank, mu=None, lmbda=None)
    L, S = rpca.fit(max_iter=1000)
    # Y = rpca.fit(max_iter=1000)
    # return Y, np.identity(data.shape[0])
    return L+S, L


# Robust NMF
def robust_nmf(data, rank, beta=0, reg_val=1, sum_to_one=1, tol=1e-7, max_iter=1000,
               print_every=10, user_prov=None):
    '''
    This function performs the robust NMF algorithm.
    Input:
        1. data: data to be factorized. WIP: based on the data type of 'data',
        all computations performed at fp32 or fp64. fp64 implemented currently.
        2. rank: rank of the factorization/number of components.
        3. beta: parameter of the beta-divergence used.
            Special cases:
            beta = 2: Squared Euclidean distance (Gaussian noise assumption)
            beta = 1: Kullback-Leibler divergence (Poisson noise assumption)
            beta = 0: Itakura-Saito divergence (multiplicative gamma noise
            assumption)
        4. init: Initialization method used for robust NMF.
            init == 'random': Draw uniform random values (recommended).
            init == 'NMF': Uses a small run of regular NMF to get initial
            values and initializes outliers uniformly at random.
            init == 'bNMF': Uses a small run of beta NMF to get initial values
            and initializes outliers uniformly at random.
            init == 'nndsvdar': Uses Boutsidis' modified algorithm and
            initializes outliers uniformly at random.
            init == 'user': the user can provide their own initialization in
            the form of a python dictionary with the keys: 'basis', 'coeff' and
            'outlier'.
        5. reg_val: Weight of L-2,1 regularization.
        6. sum_to_one: flag indicating whether a sum-to-one constraint is to be
        applied on the factor matrices.
        7. tol: tolerance on the iterative optimization. Recommended: 1e-7.
        8. max_iter: maximum number of iterations.
        9. print_every: Number of iterations at which to show optimization
        progress.
    Output:
        1. basis: basis matrix of the factorization.
        2. coeff: coefficient matrix of the factorization.
        3. outlier: sparse outlier matrix.
        4. obj: objective function progress.
    NOTE: init == 'bNMF' applies the same beta parameter as required for rNMF,
    which is nice, but is slow due to multiplicative updates
    '''

    # Utilities:
    # Defining epsilon to protect against division by zero:
    eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Initialize rNMF:
    basis, coeff, outlier = initialize_rnmf(data, rank, 'NMF', beta,
                                            sum_to_one, user_prov)

    # Set up for the algorithm:
    # Initial approximation of the reconstruction:
    data_approx = basis@coeff + outlier + eps
    fit = np.zeros(max_iter+1)
    obj = np.zeros(max_iter+1)

    # Monitoring convergence:
    fit[0] = beta_divergence(data, data_approx, beta)
    obj[0] = fit[0] + reg_val*np.sum(np.sqrt(np.sum(outlier**2, axis=0)))

    # Print initial iteration:
    # print('Iter = 0; Obj = {}'.format(obj[0]))

    for iter in range(max_iter):
        # Update the outlier matrix:
        outlier = update_outlier(data, data_approx, outlier, beta, reg_val)
        data_approx = basis@coeff + outlier + eps  # Update reconstuction

        # Update the coefficient matrix:
        coeff = update_coeff(data, data_approx, beta, basis, coeff, sum_to_one)
        data_approx = basis@coeff + outlier + eps  # Update reconstruction

        # Update the basis matrix:
        basis = update_basis(data, data_approx, beta, basis, coeff)
        data_approx = basis@coeff + outlier + eps  # Update reconstruction

        # Monitor optimization:
        fit[iter+1] = beta_divergence(data, data_approx, beta)
        obj[iter+1] = fit[iter+1] +\
                      reg_val*np.sum(np.sqrt(np.sum(outlier**2, axis=0)))

        # Termination criterion:
        if np.abs((obj[iter]-obj[iter+1])/obj[iter]) <= tol:
            # print('Algorithm converged as per defined tolerance')
            break

    # In case the algorithm terminated early:
    # obj = obj[:iter]
    # fit = fit[:iter]

    return basis, coeff.T, outlier, obj


def initialize_rnmf(data, rank, alg, beta=2, sum_to_one=0, user_prov=None):
    '''
    This function retrieves factor matrices to initialize rNMF. It can do this
    via the following algorithms:
        1. 'random': draw uniform random values.
        2. 'NMF': initialize with 200 iterations of regular NMF.
        3. 'bNMF': initialize with 200 iterations of beta NMF.
        4. 'nndsvdar': initialize with Boutsidis' modified algorithm. (classic
        nndsvd will cause issues with division by zero)
        5. 'user': provide own initializations. Must be passed in 'user_prov'
        as a dictionary with the format:
            user_prov['basis'], user_prov['coeff'], user_prov['outlier']
    Input:
        1. data: data to be factorized.
        2. rank: rank of the factorization/number of components.
        3. alg: Algorithm to initialize factorization. Either 'random', 'NMF',
        or 'bNMF'. 'bNMF' is the slowest option.
        4. beta: parameter for beta-NMF. Ignored if not provided.
        5. sum_to_one: binary flag indicating whether a simplex constraint will
        be later applied on the coefficient matrix.
        6. user_prov: if alg == 'user', then this is the dictionary containing
        the user provided initial values to use. Mandatory keys: 'basis',
        'coeff', and 'outlier'.
    Output:
        1. basis: initial basis matrix.
        2. coeff: initial coefficient matrix.
        3. outlier: initial outlier matrix.
    This can use a small run of regular/beta NMF to initialize rNMF via 'alg'.
    If a longer run is desired, or other parameters of sklearn's NMF are
    desired, modify the code below in the else block. NMF itself is very
    initialization sensitive. Here, we use Boutsidis, et al.'s NNDSVD algorithm
    to initialize it.
    Empirically, random initializations work well for rNMF.
    This initializes the outlier matrix as uniform random values.
    '''

    # Utilities:
    # Defining epsilon to protect against division by zero:
    eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Initialize outliers with uniform random values:
    outlier = np.random.rand(data.shape[0], data.shape[1])

    # Initialize basis and coefficients:
    if alg == 'random':
        basis = np.random.rand(data.shape[0], rank)
        coeff = np.random.rand(rank, data.shape[1])

        # Rescale coefficients if they will have a simplex constraint later:
        if sum_to_one == 1:
            coeff = normalize(coeff, norm='l1', axis=0)

        return basis+eps, coeff+eps, outlier+eps

    elif alg == 'NMF':
        # print('Initializing rNMF with NMF.')
        model = NMF(n_components=rank, init='nndsvdar', verbose=False)
        basis = model.fit_transform(data)
        coeff = model.components_

        # Rescale coefficients if they will have a simplex constraint later:
        if sum_to_one == 1:
            coeff = normalize(coeff, norm='l1', axis=0)

        return basis+eps, coeff+eps, outlier+eps

    elif alg == 'bNMF':

        # NNDSVDar used to initialize beta-NMF as multiplicative algorithms do
        # not like zero values and regular NNDSVD causes sparsity.
        print('Initializing rNMF with beta-NMF.')
        model = NMF(n_components=rank, init='nndsvdar', beta_loss=beta,
                    solver='mu', verbose=False)
        basis = model.fit_transform(data)
        coeff = model.components_

        # Rescale coefficients if they will have a simplex constraint later:
        if sum_to_one == 1:
            coeff = normalize(coeff, norm='l1', axis=0)

        return basis + eps, coeff + eps, outlier + eps


def beta_divergence(mat1, mat2, beta):
    '''
    This follows the definition of the beta divergence used by Fevotte, et al.
    Another definition of the beta divergence used by Amari, et al. shifts the
    values of beta by one.
    Input:
        1. mat1, mat2: matrices between which to calculate the beta divergence
        2. beta: parameter of the beta divergence
    Output:
        1. beta_div: the beta-divergence between mat1 and mat2.
    Special cases of beta:
        1. beta = 2 : Squared Euclidean Distance (Gaussian noise assumption)
        2. beta = 1 : Kullback-Leibler Divergence (Poisson noise assumption)
        3. beta = 0 : Itakura-Saito Divergence (multiplicative gamma noise
        assumption)
    NOTE: If beta = 0, the data cannot contain any zero values. If beta = 1,
    Fevotte and Dobigeon explicitly work around zero values in their version of
    the KL-divergence as shown below. beta = 2 is just the squared Frobenius
    norm of the difference between the two matrices. With the squaring, it is
    no longer an actual distance metric.
    Beta values in between the above interpolate between assumptions.
    '''

    # Utilities:
    # Defining epsilon to protect against division by zero:
    eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Inline function for vectorizing arrays for readability:
    vec = lambda X: X.flatten()

    # Main section:
    # If/else through the special limiting cases of beta, otherwise use the
    # last option:

    if beta == 2:
        # Gaussian assumption.
        beta_div = 0.5*(np.linalg.norm(mat1 - mat2, ord='fro')**2)

    elif beta == 1:
        # Poisson assumption.

        # Identify indices in the arrays that would cause issues with division
        # by zero or with the log. There's probably a faster way of doing this:
        idx_zeros = np.flatnonzero(mat1 <= eps)
        idx_interest = np.ones(mat1.size, dtype=bool)
        idx_interest[idx_zeros] = False

        # Inline functions for readability:
        nonzero = lambda X: X.flatten()[idx_interest]
        zero = lambda X: X.flatten()[idx_zeros]

        beta_div = np.sum((nonzero(mat1) * np.log(nonzero(mat1)/nonzero(mat2)))
                          - nonzero(mat1) + nonzero(mat2)) + np.sum(zero(mat2))

    elif beta == 0:
        # Multiplicative gamma assumption.
        beta_div = np.sum(vec(mat1)/vec(mat2) - np.log(vec(mat1)/vec(mat2))) -\
                    len(vec(mat1))

    else:
        # General case.
        beta_div = np.sum(vec(mat1)**beta + (beta-1)*vec(mat2)**beta
                          - beta*vec(mat1)*(vec(mat2))**(beta-1))\
                          / (beta*(beta-1))

    return beta_div


def update_basis(data, data_approx, beta, basis, coeff):
    '''
    This function updates the basis vectors of the approximation.
    In the paper, this is the M matrix.
    Input:
        1. data: data matrix to be factorized.
        2. data_approx: current approximation of the model to the data.
        3. beta: parameter of the beta-divergence.
        4. basis: current estimate of the basis matrix.
        5. coeff: current estimate of the coefficent matrix.
    Output:
        Multiplicative update for basis matrix.
    '''
    return basis * ((data*(data_approx**(beta-2))@coeff.T) /
                    ((data_approx**(beta-1))@coeff.T))


def update_coeff(data, data_approx, beta, basis, coeff, sum_to_one):
    '''
    This function updates the coefficient matrix of the approximation.
    In the paper, this is the A matrix.
    Input:
        1. data: data matrix to be factorized.
        2. data_approx: current approximation of the model to the data.
        3. beta: parameter of the beta-divergence.
        4. basis: current estimate of the basis matrix.
        5. coeff: current estimate of the coefficent matrix.
        6. sum_to_one: binary flag indicating whether a simplex constraint is
        applied on the coefficents.
    Output:
        Multiplicative update for coefficient matrix.
    '''

    # Using inline functions for readability:
    bet1 = lambda X: X**(beta-1)
    bet2 = lambda X: X**(beta-2)

    if sum_to_one == 1:

        Gn = ((basis.T)@(data*bet2(data_approx)) +
              np.sum((basis@coeff)*bet1(data_approx), axis=0))
        Gp = ((basis.T)@bet1(data_approx) +
              np.sum((basis@coeff)*data*bet2(data_approx), axis=0))
        coeff = coeff*(Gn/Gp)

        return normalize(coeff, norm='l1', axis=0)

    elif sum_to_one == 0:
        return coeff * (((basis.T)@(data*bet2(data_approx))) /
                        ((basis.T)@bet1(data_approx)))


def update_outlier(data, data_approx, outlier, beta, reg_val):
    '''
    This function updates the outlier matrix within the approximation.
    In the paper, this is the R matrix.
    Input:
        1. data: data matrix to be factorized.
        2. data_approx: current approximation of the model to the data.
        3. outlier: current estimate of the outlier matrix.
        4. beta: parameter of the beta-divergence.
        5. reg_val: strength of L-2,1 regularization on outliers.
    Output:
        Multiplicative update for outlier matrix.
    '''
    # Utilities:
    # Defining epsilon to protect against division by zero:
    eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Using inline functions for readability:
    bet1 = lambda X: X**(beta-1)
    bet2 = lambda X: X**(beta-2)
    # This normalizes the columns of a matrix X by the 2-norm of the respective
    # columns. Using this instead of sklearn's normalize to explicitly handle
    # division by zero:
    l2n = lambda X: (X /
                     (np.sum(np.abs(X)**2, axis=0)**(0.5)
                      + eps).T[np.newaxis, :])

    return outlier * ((data*bet2(data_approx)) / (bet1(data_approx) +
                                                  reg_val*l2n(outlier)))
