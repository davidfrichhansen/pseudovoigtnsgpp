import autograd.numpy as np
from autograd.scipy.stats import norm
from scipy.stats import truncnorm
from scipy.io import loadmat
from scipy.optimize import minimize
import implementation.aux_funcs as fs
import os
from autograd import value_and_grad
import funcsigs
from autograd import elementwise_grad as egrad

np.seterr('raise')

dlogistic_dx = egrad(fs.logistic, 0)

def log_posterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w):
    # TODO: Fix shapes
    # TODO: Truncnorm strange outside range

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

    alpha = np.reshape(np.exp(alpha_t), (K,N))
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
    ll = np.sum(np.sum(norm.logpdf(X,  I, 1/tau * np.ones((W,N))))) # maybe tau should be squared



    # priors

    prior_alpha = np.sum(np.sum(-alpha0*alpha + np.reshape(alpha_t, (K,N)))) # exponential prior
    prior_gamma = np.sum(np.sum(-gamma0*gamma + gamma_t)) # exponential prior
    prior_beta = np.sum(-beta0*beta + beta_t)     # exponential prior
    prior_tau = (a_tau-1)*tau_t - tau / b_tau + tau_t # gamma prior

    prior_eta = np.sum(np.log(dlogistic_dx(eta_t, 1,1)))  # uniform prior
    prior_height = np.sum(np.log(dlogistic_dx(height_t, 10, 0.5))) # uniform prior

    prior_B = 0.5*np.dot(B_t, B_t)                         # GP prior

    prior_c = np.sum(fs.truncated_normal_logpdf(c, 0, W, mu_c, 1.0 / tau_c) + np.log(dlogistic_dx(c_t, W, 0.25)))
    prior_steep = (fs.truncated_normal_logpdf(steep, 0, 50, 25, 5) + np.log(dlogistic_dx(steep_t, 50, 0.25)))

    logpost = ll + prior_alpha + prior_gamma + prior_beta + prior_tau + prior_eta + \
              prior_height + prior_B + prior_c + prior_steep

    return logpost


def logpost_grads(eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t):

    names = funcsigs.signature(logpost_grads).parameters.keys()

    return {name : value_and_grad(log_posterior, i+1) for i,name in enumerate(names)} # extremely hacky. should be changed

def objective_wrap(opt_val, opt_str, kwargs):
    par_names = funcsigs.signature(log_posterior).parameters.keys()

    par_dict = {k : None for k in par_names}
    for name in par_names:
        if name == opt_str:
            par_dict[name] = opt_val
        else:
            par_dict[name] = kwargs[name]

    val_grad_fun = grad_funs[opt_str]
    val, grad = val_grad_fun(par_dict['X'], par_dict['eta_t'], par_dict['alpha_t'], par_dict['c_t'], par_dict['gamma_t'],
                        par_dict['beta_t'], par_dict['B_t'], par_dict['tau_t'], par_dict['height_t'], par_dict['steep_t'], par_dict['w'])

    return -val, -grad



if __name__ == '__main__':
    mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/simulated.mat')
    X = mats['X'].T
    W,N = X.shape
    K = 3

    np.random.seed(1)
    eta_t = np.random.uniform(-10,10, size=(K))
    alpha_t = np.random.exponential(5, size=(K*N))
    a, b = (0 - W/2) / 100, (W - W/2) / 100
    c_t = truncnorm.rvs(a,b,100, size=(K))
    gamma_t = np.random.exponential(5, size=(K))
    beta_t = np.random.exponential(5, size=(N))
    B_t = np.random.normal(0,1,size=W)
    tau_t = np.random.gamma(0.2,0.3)
    height_t = np.random.uniform(0.01,10)
    a, b = (0 - 25) / 100, (50 - 25) / 10
    steep_t = truncnorm.rvs(a,b,25,10)

    w = np.array(list(range(W)), dtype=float)

    #ll = log_posterior(X,eta_t,alpha_t,c_t,gamma_t,beta_t,B_t, tau_t, height_t, steep_t, w)

    grad_funs = logpost_grads(eta_t,alpha_t,c_t,gamma_t,beta_t,B_t, tau_t, height_t, steep_t)

    par_dict = {
        'eta_t' : eta_t,
        'alpha_t' : alpha_t,
        'c_t' : c_t,
        'gamma_t' : gamma_t,
        'beta_t' : beta_t,
        'B_t' : B_t,
        'tau_t' : tau_t,
        'height_t' : height_t,
        'steep_t' : steep_t,
        'X' : X,
        'w' : w
    }


    max_iter = 1000


    for i in range(max_iter):
        if i % 10 == 0:
            print(f"Iteration {i} of {max_iter}\n")
        # iterate over variables to be optimized over
        for opt_str in grad_funs.keys():
            #print(opt_str, par_dict[opt_str].shape)
            opt_res = minimize(objective_wrap, par_dict[opt_str], args=(opt_str, par_dict), method='L-BFGS-B',
                               jac=True)
            if not opt_res.success:
                print(opt_str)
                print(opt_res.message)
            # update value
            par_dict[opt_str] = opt_res.x

    #opt_res = minimize(objective_wrap, alpha_t, args=('alpha_t', par_dict), method='L-BFGS-B', jac=True)

