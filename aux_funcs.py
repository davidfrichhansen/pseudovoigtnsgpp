import autograd.numpy as np
from scipy.special import erf, erfinv
import tangent


##############################   NON-STATIONARY KERNEL   ##############################
def gibbs_kernel(x, l, sigma):
    W = len(x)
    K = np.zeros((W,W))
    for i in range(W):
        for j in range(W):
            prefactor = np.sqrt(2*l[i]*l[j] / (l[i]*l[i] + l[j]*l[j]))
            exponential = np.exp(- (x[i] - x[j])*(x[i] - x[j]) / (l[i]*l[i] + l[j]*l[j]))
            K[i,j] = sigma*sigma * prefactor * exponential
    K = K + 1e-4*np.eye(W)
    return K


def length_scale(c, gamma, steep, w, height, base=1e-6):
    K = len(c)
    W = len(w)
    l = base*np.ones(W)
    for idx,k in enumerate(range(0, 2*K-1, 2)):
        endpoints1 = c[idx] - gamma[idx]
        endpoints2 = c[idx] + gamma[idx]
        l = l + height * (np.tanh((w - endpoints1)*steep) - np.tanh((w-endpoints2)*steep))

    return l



############################## LINK FUNCTIONS ##############################

# H -> eta
def forward_exp_to_gauss(fh, pars):
    sigma = pars[0]
    lamb = pars[1]
    inner = -2*np.exp(-lamb*fh) + 1
    ei = erfinv(inner)
    h = np.sqrt(2)*sigma*ei
    return h

# H -> eta
def forward_rectgauss(fh, pars):
    sigma = pars[0]
    s = pars[1]

    inner = 2 * erf(fh / (np.sqrt(2)*s) - 1)
    h = np.sqrt(2) * sigma * erfinv(inner)

    return h

# eta -> H
def link_exp_to_gauss(h, pars):
    sigma = pars[0]
    lamb = pars[1]  # inverse scale
    # actual inverse link value
    inner = 1e-12 + .5 - .5*erf(h / (np.sqrt(2) * sigma))

    val = np.maximum((-1.0 / lamb) * np.log(inner), 0)
    val = np.nan_to_num(val)

    # elementwise derivative of inverse link
    grad = 1 / (np.sqrt(2*np.pi) * sigma * lamb) * np.exp(lamb*val - h*h / (2*sigma*sigma))
    return val, grad

# eta -> H
def link_rectgauss(h, pars):
    sigma = pars[0]  # diag of Sigma_h
    s = pars[1]  # "width" parameter
    # value of inverse link
    inner = .5 + .5 * erf(h / (np.sqrt(2) * sigma))
    val = np.sqrt(2) * s * erfinv(inner)

    # elementwise derivative of inverse link
    grad = (s / (2 * sigma)) * np.exp(val ** 2 / (2 * s * s) - h ** 2 / (2 * sigma ** 2))

    return val, grad


############################## Transformations ##############################

def logistic(x, L, k, x0=0):
    val = L / (1.0 + np.exp(-k*(x-x0)))
    try:
        N = len(x)
        grad = np.array(list(map(dlogisticdx, x, [L] * N, [k] * N, [x0] * N)))
    except TypeError:
        grad = dlogisticdx(x, L, k, x0)

    return val, grad


def inv_logistic(p, L, k, x0=0):
    val = 1 / k * (np.log(p) + k*x0 - np.log(L-p))

    try:
        N = len(p)
        grad = np.array(list(map(dinv_logisticdp, p, [L]*N, [k]*N, [x0]*N)))
    except TypeError:
        grad = dinv_logisticdp(p, L, k, x0)

    return val, grad



## Following is auto-generated from tangent
def dlogisticdx(x, L, k, x0=0, bval=1.0):
    x_minus_x0 = x - x0
    minus_k = -k
    _val3 = minus_k * x_minus_x0
    _val2 = np.exp(_val3)
    _val = 1.0 + _val2
    val = L / _val
    assert tangent.shapes_match(val, bval
        ), 'Shape mismatch between return value (%s) and seed derivative (%s)' % (
        np.shape(val), np.shape(bval))
    # Grad of: val = L / (1.0 + np.exp(-k * (x - x0)))
    _b_val = -bval * L / (_val * _val)
    b_val = _b_val
    _b_val2 = tangent.unbroadcast(b_val, _val2)
    b_val2 = _b_val2
    __val2 = _val2
    _b_val3 = __val2 * b_val2
    b_val3 = _b_val3
    _bx_minus_x0 = tangent.unbroadcast(b_val3 * minus_k, x_minus_x0)
    bx_minus_x0 = _bx_minus_x0
    _bx = tangent.unbroadcast(bx_minus_x0, x)
    bx = _bx
    return bx

def dinv_logisticdp(p, L, k, x0=0, bval=1.0):
    L_minus_p = L - p
    _val3 = np.log(L_minus_p)
    k_times_x0 = k * x0
    np_log_p = np.log(p)
    _val2 = np_log_p + k_times_x0
    _val = _val2 - _val3
    _1_over_k = 1 / k
    val = _1_over_k * _val
    assert tangent.shapes_match(val, bval
        ), 'Shape mismatch between return value (%s) and seed derivative (%s)' % (
        np.shape(val), np.shape(bval))
    # Grad of: val = 1 / k * (np.log(p) + k * x0 - np.log(L - p))
    _b_val = tangent.unbroadcast(bval * _1_over_k, _val)
    b_val = _b_val
    _b_val2 = tangent.unbroadcast(b_val, _val2)
    _b_val3 = -tangent.unbroadcast(b_val, _val3)
    b_val2 = _b_val2
    b_val3 = _b_val3
    _bnp_log_p = tangent.unbroadcast(b_val2, np_log_p)
    bnp_log_p = _bnp_log_p
    _bp2 = bnp_log_p / p
    bp = _bp2
    _bL_minus_p = b_val3 / L_minus_p
    bL_minus_p = _bL_minus_p
    _bp = -tangent.unbroadcast(bL_minus_p, p)
    bp = tangent.add_grad(bp, _bp)
    return bp


def pseudo_voigt(w, c, gamma, eta):
    # w is array
    W = len(w)
    K = len(c)
    V = np.zeros((W, K))

    for k in range(K):
        L = gamma[k] * np.pi**(-1) / ((w - c[k])*(w - c[k]) + gamma[k]*gamma[k])
        G = 1.0 / (np.sqrt(2*np.pi)*gamma[k])*np.exp(-(w-c[k])**2 / (2*gamma[k]*gamma[k]))
        V[:,k] = eta[k] * L + (1 - eta[k]) * G


    # With this construction, the entire W x N matrix can be expressed as V@alpha, where alpha is a K x N matrix of coefs
    # The W x N matrix I can then be computed by I = V@alpha + np.outer(B,beta)

    return V

