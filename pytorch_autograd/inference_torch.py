import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import implementation.pytorch_autograd.aux_funcs_torch as fs
from scipy.io import loadmat


def log_posterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w, K):

    # TODO: SIGNS OF PRIORS MAY BE WRONG!!


    W, N = X.size()

    alpha0 = torch.tensor(1.0)
    mu_c = torch.tensor(W/2.)
    tau_c = torch.tensor(1.)
    gamma0 = torch.tensor(1.)
    beta0 = torch.tensor(1.)
    a_tau = torch.tensor(0.5)
    b_tau = torch.tensor(0.3)

    sigma = torch.tensor(1.)


    # parameter transformations
    alpha = torch.exp(alpha_t).reshape(K,N)
    gamma = torch.exp(gamma_t)
    eta = fs.general_sigmoid(eta_t, 1, 1)
    beta = torch.exp(beta_t)
    c = fs.general_sigmoid(c_t, W, 0.25) # <- maybe 0.25 should be changed
    tau = torch.exp(tau_t)
    height = fs.general_sigmoid(height_t, 10, 0.5)
    steep = fs.general_sigmoid(steep_t, 75, 0.25)

    l = fs.length_scale(c,gamma,steep,w,height)
    covB = fs.gibbs_kernel(w.double(),l.double(),sigma.double())
    cholB = torch.cholesky(covB)

    B = torch.mv(cholB, B_t)

    # likelihood
    V = fs.pseudo_voigt(w.double(),c.double(),gamma.double(),eta.double())
    I = torch.mm(V,alpha) + torch.ger(B,beta)

    ll = torch.distributions.normal.Normal(I.double(), 1/tau.double()).log_prob(X.double()).sum()

    prior_alpha = torch.distributions.exponential.Exponential(alpha0).log_prob(alpha).sum() - alpha_t.sum()
    prior_gamma = torch.distributions.exponential.Exponential(gamma0).log_prob(gamma).sum() - gamma_t.sum()
    prior_beta = torch.distributions.exponential.Exponential(beta0).log_prob(beta).sum() - beta_t.sum()
    prior_tau = torch.distributions.gamma.Gamma(a_tau,b_tau).log_prob(tau).sum() + tau_t.sum()

    prior_eta = torch.log(fs.dgen_sigmoid(eta_t, 1,1)).sum()
    prior_height = torch.log(fs.dgen_sigmoid(height_t, 10,0.5)).sum()
    prior_steep = torch.distributions.normal.Normal(25,5).log_prob(steep).sum() + torch.log(fs.dgen_sigmoid(steep_t, 75, 0.25)).sum()

    prior_B = -0.5 * torch.dot(B_t,B_t)

    prior_c = torch.distributions.normal.Normal(mu_c, 1 / tau_c).log_prob(c).sum() + torch.log(fs.dgen_sigmoid(c_t, W, 0.25)).sum()


    logpost = ll + prior_alpha + prior_gamma + prior_beta + prior_tau + prior_eta + \
              prior_height + prior_B + prior_c + prior_steep

    return logpost






if __name__ == '__main__':

    # TODO: REQUIRE GRADS

    mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/simulated.mat')
    X = torch.from_numpy(mats['X'].T).float()
    W, N = X.size()
    K = 3

    np.random.seed(1)
    eta_t = torch.from_numpy(np.random.uniform(-10, 10, size=(K))).requires_grad_(True)
    alpha_t = torch.from_numpy(np.random.exponential(5, size=(K * N))).requires_grad_(True)
    a, b = (0 - W / 2) / 100, (W - W / 2) / 100
    c_t = torch.from_numpy(truncnorm.rvs(a, b, 100, size=(K))).requires_grad_(True)
    gamma_t = torch.from_numpy(np.random.exponential(5., size=(K))).requires_grad_(True)
    beta_t = torch.from_numpy(np.random.exponential(5., size=(N))).requires_grad_(True)
    B_t = torch.from_numpy(np.random.normal(0, 1, size=W)).requires_grad_(True)
    tau_t = torch.tensor(np.random.gamma(0.2, 0.3)).requires_grad_(True).double()
    height_t = torch.tensor(np.random.uniform(0.01, 10)).requires_grad_(True).double()

    a, b = (0 - 25) / 100, (50 - 25) / 10
    steep_t = torch.tensor(truncnorm.rvs(a, b, 25, 10)).requires_grad_(True).double()

    w = torch.from_numpy(np.array(list(range(W))))



    # test
    ll = log_posterior(X,eta_t,alpha_t,c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w, K)
    ll.backward()
