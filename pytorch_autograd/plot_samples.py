import torch
from torch.autograd import grad
import time
import funcsigs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import implementation.pytorch_autograd.aux_funcs_torch as fs
from scipy.io import loadmat
from scipy.optimize import minimize
from implementation.pytorch_autograd.nuts import NUTS, Metropolis
from implementation.pytorch_autograd.inference_torch import positive_logpost_wrap

l_base = torch.tensor(5.0).double()
sample_dict = np.load('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/samples.npy').item()

## Visualize samples
def transform_samples(sample_dict):
    W = sample_dict['B_t'].shape[1]
    alpha_samples = np.exp(sample_dict['alpha_t'])
    num_samples = alpha_samples.shape[0]

    gamma_samples = np.exp(sample_dict['gamma_t'])
    c_samples = fs.general_sigmoid(torch.from_numpy(sample_dict['c_t']).double(), W, 0.025).numpy()
    eta_samples = fs.general_sigmoid(torch.from_numpy(sample_dict['eta_t']).double(), 1, 1).numpy()
    beta_samples = np.exp(sample_dict['beta_t'])
    steep_samples = fs.general_sigmoid(torch.from_numpy(sample_dict['steep_t']).double(), 2.0, 1.0).numpy()
    eta_samples = fs.general_sigmoid(torch.from_numpy(sample_dict['eta_t']).double(), 1, 1).numpy()
    height = 500



    l_samples = np.zeros((num_samples, sample_dict['B_t'].shape[1]))
    B_samples = np.zeros((num_samples, W))

    for i in range(num_samples):
        l_samples[i ,:] = fs.length_scale(torch.from_numpy(c_samples[i]), 5* torch.from_numpy(gamma_samples[i, :]),
                                          torch.from_numpy(steep_samples[i, :]), torch.arange(W).double(),
                                          torch.tensor(height).double(), base=l_base)
        covB = fs.gibbs_kernel(torch.arange(W).double(), torch.tensor(l_samples[i, :]), torch.tensor(0.1))

        cholB = torch.cholesky(covB)

        B_samples[i, :] = torch.mv(cholB, torch.from_numpy(sample_dict['B_t'][i, :]).double())



    return locals()


"""
sampled_pars = transform_samples(sample_dict)

B_samples = sampled_pars['B_samples']
c_samples = sampled_pars['c_samples']
ix = 50
plt.plot(B_samples[-ix,:])
plt.axvline(c_samples[-ix,:], ls='--', lw=2)

plt.show()

"""

optimize = False

mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/1x1x300_K1_2hot_noisy.mat')
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
"""

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
"""
print(f"True c: {true_c}")
print(f"True gamma: {true_gamma}")
print(f"True eta: {true_eta}")
print(f"True noise: {true_sigma}")

# convert to tensors

ta = torch.from_numpy(true_alpha.T).double()
tgamma = torch.from_numpy(true_gamma[0]).double()
tc = torch.from_numpy(true_c[0]).double()
teta = torch.from_numpy(true_eta[0]).double()
tsig = torch.from_numpy(true_sigma[0]).double()
tB = torch.from_numpy(true_B.ravel()).double()
tbeta = torch.from_numpy(true_beta.ravel()).double()
w = torch.arange(X.shape[0]).double()
# X = torch.from_numpy(X).double()
tV = torch.from_numpy(true_vp.T)

alpha_t = torch.log(ta)
gamma_t = torch.log(tgamma)
c_t = fs.inv_gen_sigmoid(tc, W, 0.025)
eta_t = fs.inv_gen_sigmoid(teta, 1, 1)
tau_t = torch.log(tsig)
beta_t = torch.log(tbeta)

height_t = torch.unsqueeze(fs.inv_gen_sigmoid(torch.tensor(500.0), 1000, 0.007), 0).double()
steep_t = torch.unsqueeze(fs.inv_gen_sigmoid(torch.tensor(0.2), 2, 1), 0).double()
# delta_t = torch.log(torch.tensor(15.0))

lt = fs.length_scale(tc, 5 * tgamma, torch.tensor(0.2), w, torch.tensor(20.0))
covB = fs.gibbs_kernel(w, lt, tsig)

cholB = torch.cholesky(covB)

cholInv = torch.inverse(cholB)

B_t = torch.mv(cholInv, tB)

w = torch.from_numpy(np.array(list(range(W)))).double()

par_dict = {
    'eta_t': eta_t,
    'alpha_t': alpha_t,
    'c_t': c_t,
    'gamma_t': gamma_t,
    'beta_t': beta_t,
    'B_t': B_t,
    'tau_t': tau_t,
    'height_t': height_t,
    'steep_t': steep_t,
    'X': X,
    'w': w,
    'K': K
}


def prop_c(c):
    return np.random.uniform(0,W,size=len(c))


def logp_c(c):
    c_t = fs.inv_gen_sigmoid(torch.from_numpy(c), W, 0.025).detach().numpy()
    val, _ = positive_logpost_wrap(c_t, 'c_t', par_dict)
    return val
metropolisC = Metropolis(logp_c, fs.general_sigmoid(c_t, W, 0.025).numpy(), prop_c)

plt.figure()
num_samples = 1
plt.plot(X.numpy())
V_plot = plt.plot( fs.pseudo_voigt(w,tc,tgamma, teta).numpy())
for i in range(num_samples):
    metropolisC.sample(override_M=1)
    sample = metropolisC.samples

    if metropolisC.acc_rate > 0:
        V_plot[0].remove()
        V = fs.pseudo_voigt(w,torch.from_numpy(sample[0,:]).double(),tgamma, teta)
        V_plot = plt.plot(V.numpy())
    plt.draw()
    plt.pause(0.01)

