import torch
from torch.autograd import grad

import funcsigs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import implementation.pytorch_autograd.aux_funcs_torch as fs
from scipy.io import loadmat, savemat
from implementation.pytorch_autograd.nuts import NUTS, Metropolis


### FUNCTIONS

def positive_logpost_wrap(par_value, par_name, other_pars):
    # wraps the objective function for par_name
    names = funcsigs.signature(log_posterior).parameters.keys()
    par_dict = {n: None for n in names}
    par_tensor = torch.from_numpy(par_value).requires_grad_(True)
    # forward pass
    for n in names:
        if n == par_name:
            par_dict[n] = par_tensor
        else:
            par_dict[n] = other_pars[n]

    ll = log_posterior(par_dict['X'], par_dict['eta_t'], par_dict['alpha_t'], par_dict['c_t'], par_dict['gamma_t'],
                       par_dict['beta_t'], par_dict['B_t'], par_dict['tau_t'], par_dict['height_t'],
                       par_dict['steep_t'], par_dict['w'], par_dict['K'], par_dict['l_base'])

    # backprop
    par_grad = grad(ll, par_tensor)[0]  # par_value is tensor, which is why this works
    ll_detach = ll.detach()
    grad_detach = par_grad.detach()
    return ll_detach.numpy(), grad_detach.numpy()


def log_posterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w, K, l_base=torch.tensor(5.0).double()):

    W, N = X.size()
    mean_TB = torch.tensor([0.0]).double()
    #mean_TB = torch.median(X*X)
    alpha0 = torch.tensor(0.2)
    mu_c = torch.tensor(W/2.)
    tau_c = torch.tensor(0.005)

    #beta0 = torch.tensor(1.0)
    a_tau = torch.tensor(7.5)
    b_tau = torch.tensor(1.0)

    #l_base = torch.tensor(5).double().requires_grad_(False)




    # parameter transformations8
    alpha = torch.exp(alpha_t).reshape(K,N)
    gamma = torch.exp(gamma_t)
    eta = fs.general_sigmoid(eta_t, 1, 1)
    beta = torch.exp(beta_t)

    c = fs.general_sigmoid(c_t, W, 0.025)
    tau = torch.exp(tau_t)
    height = fs.general_sigmoid(height_t, 1000, 0.007)
    steep = fs.general_sigmoid(steep_t, 2.0, 1.0)
    l = fs.length_scale(c, 5*gamma,steep,w,height, base=l_base)

    sigma = 1/tau


    covB = fs.gibbs_kernel(w,l,sigma)

    cholB = torch.cholesky(covB)


    B = torch.mv(cholB, B_t) + mean_TB


    # likelihood
    V = fs.pseudo_voigt(w,c,gamma,eta)
    I = torch.mm(V,alpha) + torch.ger(B,beta)

    ll = torch.distributions.normal.Normal(I, 1/tau).log_prob(X).sum()

    prior_alpha = torch.distributions.exponential.Exponential(alpha0).log_prob(alpha).sum() + alpha_t.sum()
    #prior_alpha = fs.truncated_normal_lpdf(alpha, torch.tensor(5.0).double(), torch.tensor(1.5).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum() + \
    #alpha_t.sum()
    prior_gamma = fs.truncated_normal_lpdf(gamma, torch.tensor(10.).double(), torch.tensor(1.0/6.0).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum() + \
        gamma_t.sum()

    prior_beta = fs.truncated_normal_lpdf(beta, torch.tensor(0.5), torch.tensor(0.02), 0, torch.tensor(float('Inf')).double()).sum() + beta_t.sum()
    prior_tau = torch.distributions.gamma.Gamma(a_tau,b_tau).log_prob(tau).sum() + tau_t
    prior_eta = torch.log(fs.dgen_sigmoid(eta_t, 1,1)).sum()
    #  torch.distributions.normal.Normal(torch.tensor(20.0), torch.tensor(5.0)).log_prob(height) +
    prior_height = torch.log(fs.dgen_sigmoid(height_t, 1000,0.007)).sum()
    prior_steep = fs.truncated_normal_lpdf(steep, torch.tensor(0.2).double(),torch.tensor(.5).double(),
                                           torch.tensor(0.).double(),torch.tensor(5.).double()) + torch.log(fs.dgen_sigmoid(steep_t, 2.0, 1.0))
    prior_B = -0.5 * torch.dot(B_t,B_t)
    prior_c = fs.truncated_normal_lpdf(c, mu_c, 1.0 / tau_c, 0, torch.tensor(W).double()).sum() + torch.log(fs.dgen_sigmoid(c_t, W, 0.025)).sum()
    #prior_c = torch.log(fs.dgen_sigmoid(c_t, W, 0.025)).sum()

    logpost = ll + prior_alpha + prior_gamma + prior_beta + prior_tau + prior_eta + \
              prior_height + prior_B + prior_c + prior_steep

    return logpost

#### SETUP
mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/25x25x300_K1_2hot_lessnoisy.mat')
X = torch.from_numpy(mats['X'].T).double()
gen = mats['gendata']
W, N = X.size()
K = 1

X.shape

K = 1

true_alpha = gen['A'][0][0]
true_vp = gen['vp'][0][0]
true_c = gen['c'][0][0]
true_eta = gen['eta_voigt'][0][0]
true_gamma = gen['gamma'][0][0]
true_B = gen['B'][0][0]
true_beta = gen['b'][0][0]
true_sigma = gen['sig'][0][0]


plt.figure()
plt.plot(true_vp.T)
plt.title('True Voigt')
plt.show()

plt.figure()
plt.plot(true_alpha)
plt.title('True alpha')
plt.show()

plt.figure()
plt.plot(true_B)
plt.title('True background')
plt.show()

plt.figure()
plt.plot(true_beta)
plt.title('True beta')
plt.show()

print(f"True c: {true_c}")
print(f"True gamma: {true_gamma}")
print(f"True eta: {true_eta}")
print(f"True noise: {true_sigma}")

# convert to tensors

ta = torch.from_numpy(true_alpha.T).double()
tgamma = torch.from_numpy(true_gamma[0]).double()
tc = torch.from_numpy(true_c[0]).double()
teta = torch.from_numpy(true_eta[0]).double()
#tsig = 1.0 /torch.from_numpy(true_sigma[0]).double()
tsig = torch.tensor(1.0).double()
tB = torch.from_numpy(true_B.ravel()).double()
tbeta = torch.from_numpy(true_beta.ravel()).double()
w = torch.arange(X.shape[0]).double()


tV = torch.from_numpy(true_vp.T)

alpha_t = torch.log(ta)
gamma_t = torch.log(tgamma)
c_t = fs.inv_gen_sigmoid(tc, W, 0.025)
eta_t = fs.inv_gen_sigmoid(teta, 1, 1)
tau_t = torch.log(tsig)
beta_t = torch.log(tbeta)
height = torch.tensor(100).double()
height_t = torch.unsqueeze(fs.inv_gen_sigmoid(height, 1000, 0.007), 0).double()
steep = torch.tensor(.1).double()
steep_t = torch.unsqueeze(fs.inv_gen_sigmoid(steep, 2, 1), 0).double()

l_base = torch.tensor(8).double()
lt = fs.length_scale(tc, 5 * tgamma, steep, w, height, base=l_base)


plt.figure()
plt.plot(lt.detach().numpy())

covB = fs.gibbs_kernel(w, lt, tsig)



cholB = torch.cholesky(covB) # not the same in inference test
cholInv = torch.inverse(cholB)
mean_tB = torch.tensor(0).double()

GP = torch.distributions.multivariate_normal.MultivariateNormal(mean_tB.unsqueeze(0), covariance_matrix=covB.double())
print(GP.log_prob(tB))

plt.figure()
plt.plot(GP.sample([5]).numpy().T)
plt.title('samples')
plt.axvline((tc - 5*tgamma).numpy())
plt.axvline((tc + 5*tgamma).numpy())


plt.figure()
plt.plot(tB.numpy())
plt.title('True background')


B_t = torch.mv(cholInv, (tB-mean_tB))
plt.figure()
plt.plot(B_t.numpy())
plt.title('True background transformed')
plt.axvline((tc - 5*tgamma).numpy())
plt.axvline((tc + 5*tgamma).numpy())


plt.figure()
plt.plot((torch.mv(cholB, B_t) + mean_tB).numpy())
plt.title('hard coded')

# prior for transformed variable is prop to -0.5*(B_t - mu)^T* (B_t-mu)!

print(-0.5*torch.dot(B_t - mean_tB, (B_t - mean_tB)))

par_dict = {
    'eta_t': eta_t,
    'alpha_t': alpha_t,
    'c_t': c_t,
    'gamma_t': gamma_t,
    'beta_t': beta_t,
    #'B_t': B_t,
    'B_t' : torch.randn(W).double(),
    'tau_t': tau_t,
    'height_t': height_t,
    'steep_t': steep_t,
    'X': X,
    'w': w,
    'K': K,
    'l_base' : l_base
}
par_dict_orig = par_dict

#%%  INFERENCE
par_dict = par_dict_orig


def logp_height(h, par_dict):
    h_t = fs.inv_gen_sigmoid(torch.from_numpy(np.array(h)).double(), 1000, 0.007).double().detach().numpy()
    val, _ = positive_logpost_wrap(h_t, 'height_t', par_dict)
    return val


def prop_h(h):
    #return np.random.normal(h,10)
    return np.random.uniform(0,200)


def prop_c(c):
    #return np.random.uniform(0,W,size=len(c))
    return np.random.multivariate_normal(c, 100*np.eye(len(c)))



def logp_c(c, par_dict):
    c_t = fs.inv_gen_sigmoid(torch.from_numpy(c), W, 0.025).detach().numpy()
    val, _ = positive_logpost_wrap(c_t, 'c_t', par_dict)
    return val


def logp_gamma(g, par_dict):
    g_t = np.log(g)
    val,_ = positive_logpost_wrap(g_t, 'gamma_t', par_dict)
    return val

def prop_gamma(g):
    # a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    a,b = (0.01 - g) / 10, (100 - g) / 10
    return truncnorm.rvs(a,b,g,10, size=len(g))

def logp_eta(eta, par_dict):
    eta_t = fs.inv_gen_sigmoid(torch.from_numpy(eta), 1,1).double().numpy()
    val, _ = positive_logpost_wrap(eta_t, 'eta_t', par_dict)
    return val

def prop_eta(eta):
    return np.random.uniform(1e-8,1, size=len(eta))




# DUAL AVERAGING
M_adapt = 30
NUTS_alpha = NUTS(positive_logpost_wrap, M_adapt+1,M_adapt, par_dict['alpha_t'].numpy().ravel(), 'alpha_t', par_dict)
NUTS_B = NUTS(positive_logpost_wrap, M_adapt+1, M_adapt,par_dict['B_t'].detach().numpy(), 'B_t', par_dict)
NUTS_beta = NUTS(positive_logpost_wrap, M_adapt+1, M_adapt, par_dict['beta_t'].numpy(), 'beta_t', par_dict)
#NUTS_gamma = NUTS(positive_logpost_wrap, M_adapt+1, M_adapt, par_dict['gamma_t'].numpy(), 'gamma_t', par_dict)
#NUTS_gamma.sample()
NUTS_alpha.sample()
NUTS_B.sample()
NUTS_beta.sample()

eps_alpha = np.mean(NUTS_alpha.eps_list[-20])
eps_B = np.mean(NUTS_B.eps_list[-20])
eps_beta = np.mean(NUTS_beta.eps_list[-20])
#eps_gamma = np.mean(NUTS_gamma.eps_list[-20])

#%% SAMPLES
par_dict = par_dict_orig

num_samples = 500
live_plot = True

metropolisC = Metropolis(logp_c, np.array([83.0]), prop_c, par_dict)

metropolisGamma = Metropolis(logp_gamma, np.array([10.0]), prop_gamma, par_dict)

metropolisH = Metropolis(logp_height, np.array([30.0]), prop_h, par_dict)

metropolisEta = Metropolis(logp_eta, np.array([0.5]), prop_eta, par_dict)

samples_dict = {'c':np.zeros((num_samples,K)),
                'gamma':np.zeros((num_samples,K)),
                'B':np.zeros((num_samples,W)),
                'height':np.zeros((num_samples)),
                'alpha' : np.zeros((num_samples, K*N)),
                'beta' : np.zeros((num_samples, N)),
                'eta' : np.zeros((num_samples, K))}


# initial sample
#NUTS_gamma = NUTS(positive_logpost_wrap, 2,0,par_dict['gamma_t'].numpy(), 'gamma_t', par_dict, start_eps=0.5)
#NUTS_gamma.sample(override_M=2, override_Madapt=0)
NUTS_B = NUTS(positive_logpost_wrap, 2,0,par_dict['B_t'].detach().numpy(), 'B_t', par_dict, start_eps=eps_B)
NUTS_B.sample(override_M=2, override_Madapt=0)
NUTS_alpha = NUTS(positive_logpost_wrap, 2,0, par_dict['alpha_t'].numpy().ravel(), 'alpha_t', par_dict, start_eps=eps_alpha)
NUTS_alpha.sample()

NUTS_beta = NUTS(positive_logpost_wrap, 2, 0, par_dict['beta_t'].numpy(), 'beta_t', par_dict, start_eps=eps_beta)
NUTS_beta.sample()
#NUTS_c = NUTS(positive_logpost_wrap, 2,0,fs.inv_gen_sigmoid(torch.tensor([1.0]).double(), W, 0.025).numpy(),'c_t', par_dict, start_eps=eps_c)
#NUTS_c.sample()
metropolisC.sample(override_M=1)

metropolisH.sample(override_M=1)
metropolisGamma.sample(override_M=1)
metropolisEta.sample(override_M=1)

samples_dict['c'][0,:] = metropolisC.samples[0]
#samples_dict['c'][0,:] = fs.general_sigmoid(torch.from_numpy(NUTS_c.samples[1,:]), W,0.025)
samples_dict['height'][0] = metropolisH.samples[0]
samples_dict['B'][0,:] = NUTS_B.samples[1,:]
samples_dict['gamma'][0,:] = metropolisGamma.samples[0,:]
#samples_dict['gamma'][0,:] = np.exp(NUTS_gamma.samples[1,:])
samples_dict['alpha'][0,:] = NUTS_alpha.samples[1,:]
samples_dict['beta'][0,:] = NUTS_beta.samples[1,:]
samples_dict['eta'][0,:] = metropolisEta.samples[0,:]
if live_plot:
    plt.figure()
    #plt.plot(X[:,67].numpy()) # spectrum 67 has alot of signal - just for plotting
    #V_plot = plt.plot(fs.pseudo_voigt(w,torch.from_numpy(samples_dict['c'][0,:]).double(),torch.from_numpy(samples_dict['gamma'][0,:]), torch.from_numpy(samples_dict['eta'][0,:])).numpy())
    plt.plot((ta[:, 492] * fs.pseudo_voigt(w, tc.double(),
                                           tgamma,
                                           teta)).numpy())
    V_plot = plt.plot((ta[:,492]*fs.pseudo_voigt(w, tc.double(),
                                      tgamma,
                                      teta)).numpy())
    plt.pause(0.5)

for s in range(1,num_samples):
    #NUTS_c =  NUTS(positive_logpost_wrap, 2,0,fs.inv_gen_sigmoid(torch.from_numpy(samples_dict['c'][s-1,:]).double(),W,0.025).numpy(),'c_t', par_dict, start_eps=eps_c)
    #NUTS_c.sample()

    metropolisC = Metropolis(logp_c, samples_dict['c'][s-1,:], prop_c, par_dict)
    print("SAMPLE C\n\n")
    metropolisC.sample(override_M=1)
    NUTS_B = NUTS(positive_logpost_wrap, 2, 0, samples_dict['B'][s-1,:], 'B_t', par_dict, start_eps=eps_B)
    NUTS_alpha = NUTS(positive_logpost_wrap, 2, 0, samples_dict['alpha'][s-1,:], 'alpha_t', par_dict,
                      start_eps=eps_alpha)
    NUTS_beta = NUTS(positive_logpost_wrap, 2, 0, samples_dict['beta'][s-1,:], 'beta_t', par_dict, start_eps=eps_beta)
    NUTS_beta.sample()

    #NUTS_gamma = NUTS(positive_logpost_wrap, 2,0, np.log(samples_dict['gamma'][s-1,:]), 'gamma_t', par_dict, start_eps=0.5)
    #NUTS_gamma.sample()
    print("SAMPLE GAMMA \n\n")
    metropolisGamma = Metropolis(logp_gamma, samples_dict['gamma'][s-1,:], prop_gamma, par_dict)
    metropolisGamma.sample(override_M=1)
    NUTS_B.sample()
    NUTS_alpha.sample()

    metropolisEta = Metropolis(logp_eta, samples_dict['eta'][s - 1, :], prop_eta, par_dict)
    metropolisH = Metropolis(logp_height, np.array([samples_dict['height'][s-1]]), prop_h, par_dict)


    print("SAMPLE H\n\n")
    metropolisH.sample(override_M=1)
    print("SAMPLE ETA\n\n")
    metropolisEta.sample(override_M=1)

    #samples_dict['c'][s, :] = fs.general_sigmoid(torch.from_numpy(NUTS_c.samples[1,:]),W,0.025).numpy()
    samples_dict['c'][s,:] = metropolisC.samples[0,:]
    print(samples_dict['c'][s,:])
    samples_dict['height'][s] = metropolisH.samples[0]
    samples_dict['B'][s, :] = NUTS_B.samples[1, :]
    samples_dict['gamma'][s,:] = metropolisGamma.samples[0, :]
    #samples_dict['gamma'] [s,:] = np.exp(NUTS_gamma.samples[1,:])
    samples_dict['alpha'][s, :] = NUTS_alpha.samples[1,:]
    samples_dict['beta'][s,:] = NUTS_beta.samples[1,:]
    samples_dict['eta'][s,:] = metropolisEta.samples[0,:]
    if live_plot:
        V_plot[0].remove()
        V = fs.pseudo_voigt(w,torch.from_numpy(samples_dict['c'][s,:]),torch.from_numpy((samples_dict['gamma'][s,:])), torch.from_numpy(samples_dict['eta'][s,:]))
        V_plot = plt.plot((torch.exp(torch.tensor(samples_dict['alpha'][s,492]))*V).numpy(),color='C1')
        print(f"alpha diff: {ta[:,492].numpy() - samples_dict['alpha'][s,492]}\n")
       # V_plot = plt.plot((V).numpy(),color='C1')
        plt.draw()
        plt.pause(0.001)


    par_dict['c_t'] = fs.inv_gen_sigmoid(torch.from_numpy(samples_dict['c'][s,:]), W, 0.025)
    par_dict['gamma_t'] = torch.log(torch.from_numpy(samples_dict['gamma'][s,:]).double())
    #par_dict['gamma'] = torch.from_numpy(samples_dict['gamma'][s,:])
    par_dict['height_t'] = fs.inv_gen_sigmoid(torch.from_numpy(np.array(samples_dict['height'][s])), 1000, 0.007).double()
    par_dict['B_t'] = torch.from_numpy(samples_dict['B'][s,:])
    par_dict['alpha_t'] = torch.from_numpy(samples_dict['alpha'][s,:])
    par_dict['beta_t'] = torch.from_numpy(samples_dict['beta'][s,:])
    par_dict['eta_t'] = fs.general_sigmoid(torch.from_numpy(samples_dict['eta'][s,:]),1,1)


#%%
B_samples = np.zeros((num_samples, W))
c_samples = np.zeros((num_samples,K))
g_samples = np.zeros((num_samples, K))
h_samples = np.zeros(num_samples)
beta_samples = np.zeros((num_samples, N))
alpha_samples = np.zeros((num_samples, K*N))

# transform samples back
for s in range(num_samples):
    cur_h = torch.from_numpy(np.array([samples_dict['height'][s]])).double()
    cur_c = torch.from_numpy(samples_dict['c'][s,:]).double()
    cur_Bt = torch.from_numpy(samples_dict['B'][s,:]).double()
    #cur_gt = torch.from_numpy(samples_dict['gamma'][s,:]).double()

    cur_g = torch.from_numpy(samples_dict['gamma'][s,:]).double()

    cur_l = fs.length_scale(cur_c, cur_g, steep, w, cur_h, l_base)
    cur_cov = fs.gibbs_kernel(w,cur_l,tsig)

    cur_chol = torch.cholesky(cur_cov)
    cur_B = torch.mv(cur_chol, cur_Bt)

    B_samples[s,:] = cur_B.numpy()
    c_samples[s,:] = cur_c.numpy()
    g_samples[s,:] = cur_g.numpy()
    h_samples[s] = cur_h.numpy()

alpha_samples = np.exp(samples_dict['alpha'])
beta_samples = np.exp(samples_dict['beta'])

