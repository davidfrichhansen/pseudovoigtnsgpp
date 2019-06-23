import torch
from torch.autograd import grad
import time
import funcsigs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import implementation.pytorch_autograd.aux_funcs_torch as fs
from scipy.io import loadmat, savemat
from scipy.optimize import minimize
from implementation.pytorch_autograd.nuts import NUTS, Metropolis
from implementation.pytorch_autograd.inference_torch import positive_logpost_wrap

sample_dict = np.load('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/samples.npy').item()
l_base = torch.tensor(5.0).double()

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


mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/25x25x300_K1_2hot.mat')
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

tV = torch.from_numpy(true_vp.T)

alpha_t = torch.log(ta)
gamma_t = torch.log(tgamma)
c_t = fs.inv_gen_sigmoid(tc, W, 0.025)
eta_t = fs.inv_gen_sigmoid(teta, 1, 1)
tau_t = torch.log(tsig)
beta_t = torch.log(tbeta)
height = torch.tensor(20).double()
height_t = torch.unsqueeze(fs.inv_gen_sigmoid(height, 1000, 0.007), 0).double()
steep = torch.tensor(.1).double()
steep_t = torch.unsqueeze(fs.inv_gen_sigmoid(steep, 2, 1), 0).double()

l_base = torch.tensor(5).double()
lt = fs.length_scale(tc, 2 * tgamma, steep, w, height, base=l_base)


plt.figure()
plt.plot(lt.detach().numpy())

covB = fs.gibbs_kernel(w, lt, tsig)

#savemat('plotcov.mat', {'covB' : covB.numpy()})


cholB = torch.cholesky(covB) # not the same in inference test
cholInv = torch.inverse(cholB)
#mean_tB = torch.mean(tB)
mean_tB = torch.tensor(0).double()
GP = torch.distributions.multivariate_normal.MultivariateNormal(mean_tB.unsqueeze(0), covariance_matrix=covB.double())
print(GP.log_prob(tB))

plt.figure()
plt.plot(GP.sample([5]).numpy().T)
plt.title('samples')
plt.axvline((tc - 2*tgamma).numpy())
plt.axvline((tc + 2*tgamma).numpy())


plt.figure()
plt.plot(tB.numpy())
plt.title('True background')


B_t = torch.mv(cholInv, (tB-mean_tB))
plt.figure()
plt.plot(B_t.numpy())
plt.title('True background transformed')
plt.axvline((tc - 2*tgamma).numpy())
plt.axvline((tc + 2*tgamma).numpy())


plt.figure()
plt.plot(torch.mv(cholB, (B_t+mean_tB)).numpy())
plt.title('hard coded')

# prior for transformed variable is prop to -0.5*(B_t - mu)^T* (B_t-mu)!

print(-0.5*torch.dot(B_t - mean_tB, (B_t - mean_tB)))

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
    #return np.random.multivariate_normal(c, 500*np.eye(len(c)))


def logp_c(c):
    c_t = fs.inv_gen_sigmoid(torch.from_numpy(c), W, 0.025).detach().numpy()
    val, _ = positive_logpost_wrap(c_t, 'c_t', par_dict)
    return val

logpc_val = logp_c(tc.numpy())


c_arr = torch.arange(1,W).double()
logpc = np.zeros(W-1)

for i in range(W-1):
    logpc[i] = logp_c(c_arr[i].unsqueeze(0).numpy())

plt.figure()
plt.plot(c_arr.numpy(),logpc)
plt.xlabel('c')
plt.ylabel('log prob')
plt.title('p(c|X, alpha, beta, ...)')


metropolisC = Metropolis(logp_c, np.array([150.0]), prop_c)
plt.figure()
num_samples = 500
plt.plot(X[:,67].numpy()) # spectrum 67 has alot of signal - just for plotting
V_plot = plt.plot( fs.pseudo_voigt(w,tc,tgamma, teta).numpy())
c_samples = []
for i in range(num_samples):

    metropolisC.sample(override_M=1)
    sample = metropolisC.samples
    c_samples.append(sample[0][0])
    if metropolisC.acc_rate > 0:
        print('Accept!')
        V_plot[0].remove()
        V = fs.pseudo_voigt(w,torch.from_numpy(sample[0,:]).double(),tgamma, teta)
        V_plot = plt.plot(V.numpy())
    plt.draw()
    plt.pause(0.01)

"""
def naive_likelihood(alpha, beta, c, gamma, eta, B, W=300):
    V = fs.pseudo_voigt(torch.arange(W).double(), c, gamma, eta)
    I = torch.mm(V, alpha) + torch.ger(B, beta)

    return torch.distributions.normal.Normal(I, 1 / 0.1).log_prob(X).sum() + torch.log(fs.dgen_sigmoid(fs.inv_gen_sigmoid(c, W, 0.025), W, 0.025))

c_arr = torch.linspace(1e-3,W,1000).double()
ll_c = torch.zeros(len(c_arr))


for i in range(len(c_arr)):
    ll_c[i] = naive_likelihood(ta, tbeta, c_arr[i].unsqueeze(0), tgamma, teta, tB)
"""