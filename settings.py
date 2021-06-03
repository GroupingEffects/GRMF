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

# default parameter for GRMF
default_param_A = {
    TAU: 5,
    TAU_1: 1,
    TAU_2: 0.9,
    LAMBDA0: 0,
    LAMBDA1: 0.1,
    LAMBDA2: 0.001,
    LAMBDA3: 0,
    INNTER_TOL: 1e-4,
    OUTER_TOL: 1e-2,
    V_ALPHA: 1,
    V_INIT: 1.1
}

default_param_B = {
    TAU: 1,
    TAU_1: 1,
    TAU_2: 1,
    LAMBDA0: 0.1,
    LAMBDA1: 0.1,
    LAMBDA2: 0.001,
    LAMBDA3: 0.01,
    INNTER_TOL: 1e-4,
    OUTER_TOL: 1e-2,
    V_ALPHA: 1,
    V_INIT: 1.1
}