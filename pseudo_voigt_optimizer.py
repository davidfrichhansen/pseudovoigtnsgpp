import autograd.numpy as np
from scipy.stats import truncnorm, norm
from scipy.io import loadmat
import implementation.aux_funcs as fs
import os
from autograd import grad, multigrad_dict
from autograd import elementwise_grad as egrad

dlogistic_dx = egrad(fs.logistic, 0)

def log_posterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w):

    W, N = X.shape



    ## Hyperparameters
    alpha0 = 1
    mu_c = W / 2
    tau_c = 1
    gamma0 = 1
    beta0 = 1
    a_tau = 0.5
    b_tau = 0.3

    sigma = 1

    # chol rotate B
    # alpha_t = log(alpha) ~ exp(alpha0)
    # c_tilde = gen logistic(c, W, 0.25) ~ trunc norm(mu_c, tau_c)
    # steep_tilde = gen logistic(steep, 50 , 0.25) ~ trunc norm(25, 5)
    # gamma_t = log(gamma) ~ exp(gamma0)
    # beta_t = log(beta) ~ exp(beta0)
    # tau_t = log(tau) ~ gamma(a_tau, b_tau)
    # eta_t = logistic(eta) ~ uniform(0,1)
    # height_t = 10*logistic(height) ~ uniform(0,10)


    # Transform parameters

    alpha = np.exp(alpha_t)
    gamma = np.exp(gamma_t)
    eta = fs.logistic(eta_t,1,1)
    beta = np.exp(beta_t)
    c = fs.logistic(c_t, W, 0.25)
    tau = np.exp(tau_t)
    height = fs.logistic(height_t, 10, 0.5)
    steep = fs.logistic(steep_t, 50, 0.25)

    ls = fs.length_scale(c, gamma, steep, w, height)
    cov_B = fs.gibbs_kernel(w, ls, sigma)
    chol_B = np.linalg.cholesky(cov_B)
    # rotate B_t to B
    B = np.dot(chol_B.T,B_t)


    ######

    # likelihood

    V = fs.pseudo_voigt(w, c, gamma, eta)
    I = np.dot(V,alpha) + np.outer(B,beta)

    assert I.shape == X.shape


    ll = np.sum(np.sum(norm.logpdf(X,  loc=I, scale=1/tau))) # maybe tau should be squared


    # priors

    prior_alpha = np.sum(np.sum(-alpha0*alpha + alpha_t)) # exponential prior
    prior_gamma = np.sum(np.sum(-gamma0*gamma + gamma_t)) # exponential prior
    prior_beta = np.sum(-beta0*beta + beta_t)     # exponential prior
    prior_tau = (a_tau-1)*tau_t - tau / b_tau + tau_t # gamma prior

    prior_eta = np.sum(np.log(dlogistic_dx(eta_t, 1,1)))  # uniform prior
    prior_height = np.sum(np.log(dlogistic_dx(height_t, 10, 0.5))) # uniform prior

    prior_B = 0.5*np.dot(B_t, B_t)                         # GP prior

    prior_c = np.sum(truncnorm.logpdf(c, 0, W, mu_c, 1.0 / tau_c) + np.log(dlogistic_dx(c_t, W, 0.25)))
    prior_steep = (truncnorm.logpdf(steep, 0, 50, 25, 5) + np.log(dlogistic_dx(steep_t, 50, 0.25)))

    logpost = ll + prior_alpha + prior_gamma + prior_beta + prior_tau + prior_eta + \
              prior_height + prior_B + prior_c + prior_steep

    return logpost

grad_logposterior = multigrad_dict(log_posterior)

def logpost_wrapper(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w):
    val = log_posterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w)
    grad_dict = grad_logposterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w)

    return val, grad_dict



if __name__ == '__main__':
    print(os.getcwd())
    mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/simulated.mat')
    X = mats['X'].T
    W,N = X.shape
    K = 3

    np.random.seed(1)

    eta_t = np.random.uniform(-10,10, size=(K))

    alpha_t = np.random.exponential(5, size=(K,N))
    c_t = truncnorm.rvs(0,W,W/2,100, size=(K))
    gamma_t = np.random.exponential(5, size=(K))
    beta_t = np.random.exponential(5, size=(N))
    B_t = np.random.normal(0,1,size=W)
    tau_t = np.random.gamma(5,3)
    height_t = np.random.uniform(0.01,10)
    steep_t = truncnorm.rvs(0,50,25,10)

    w = np.array(list(range(W)), dtype=float)

    ll = log_posterior(X,eta_t,alpha_t,c_t,gamma_t,beta_t,B_t, tau_t, height_t, steep_t, w)

    ll, grad_dict = logpost_wrapper(X,eta_t,alpha_t,c_t,gamma_t,beta_t,B_t, tau_t, height_t, steep_t, w)