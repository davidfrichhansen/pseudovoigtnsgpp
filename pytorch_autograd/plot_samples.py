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
height = torch.tensor(10).double()
height_t = torch.unsqueeze(fs.inv_gen_sigmoid(height, 1000, 0.007), 0).double()
steep = torch.tensor(.1).double()
steep_t = torch.unsqueeze(fs.inv_gen_sigmoid(steep, 2, 1), 0).double()

l_base = torch.tensor(5).double()
lt = fs.length_scale(tc, 5 * tgamma, steep, w, height, base=l_base)


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

#%%

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
num_samples = 0
#plt.plot(X[:,67].numpy()) # spectrum 67 has alot of signal - just for plotting
plt.plot(tV.numpy())
V_plot = plt.plot(fs.pseudo_voigt(w,tc,tgamma, teta).numpy())
c_samples = [np.array([150.0])]
for i in range(1,num_samples):

    metropolisC.sample(override_M=1, override_theta0=c_samples[i-1])
    sample = metropolisC.samples
    c_samples.append(sample[0])
    if metropolisC.acc_rate > 0:
        print('Accept!')
        V_plot[0].remove()
        V = fs.pseudo_voigt(w,torch.from_numpy(sample[0,:]).double(),tgamma, teta)
        V_plot = plt.plot(V.numpy())
        plt.draw()
    plt.pause(0.001)

plt.figure()
plt.hist([c_samples[i][0] for i in range(len(c_samples))])

#%%
B_0 = torch.randn(W)
num_samples = 1000
NUTS_B = NUTS(positive_logpost_wrap, 1000,0,par_dict['B_t'].numpy(), 'B_t', par_dict, start_eps=0.55)
NUTS_B.sample()
B_samples = torch.zeros(1000, 300)

for idx,s in enumerate(NUTS_B.samples):
    B_samples[idx, :] = torch.mv(cholB, torch.from_numpy(s))


#%%
num_samples = 2000
NUTS_alpha = NUTS(positive_logpost_wrap, num_samples, 100, par_dict['alpha_t'].numpy().ravel(), 'alpha_t', par_dict)

NUTS_alpha.sample()
alpha_samples = torch.zeros(num_samples, N)

for idx, s in enumerate(NUTS_alpha.samples):
    alpha_samples[idx,:] = torch.exp(torch.from_numpy(s))

#%%
num_samples = 1000
NUTS_beta = NUTS(positive_logpost_wrap, num_samples, 100, par_dict['beta_t'].numpy(), 'beta_t', par_dict)
NUTS_beta.sample()
beta_samples = torch.zeros(num_samples-100, N)

for idx, s in enumerate(NUTS_beta.samples):
    beta_samples[idx, :] = torch.exp(torch.from_numpy(s))

#%%

eta_arr = torch.linspace(0,1,500).double()

def logp_eta(eta):
    eta_t = fs.inv_gen_sigmoid(eta,1,1).unsqueeze(0).detach().numpy()

    val, _ = positive_logpost_wrap(eta_t, 'eta_t', par_dict)
    return val

def prop_eta(eta):
    return np.random.uniform(0,1,size=len(eta))

logpeta = np.zeros(len(eta_arr))

for idx, eta in enumerate(eta_arr):
    logpeta[idx] = logp_eta(eta)


metropolisEta = Metropolis(logp_eta, np.array([0.5]), prop_eta)



#%% c, gamma, B, height

def logp_height(h):
    h_t = fs.inv_gen_sigmoid(torch.from_numpy(np.array(h)).double(), 1000, 0.007).double().detach().numpy()
    val, _ = positive_logpost_wrap(h_t, 'height_t', par_dict)
    return val


def prop_h(h):
    #return np.random.normal(h,10)
    return np.random.uniform(0,1000)

metropolisH = Metropolis(logp_height, np.array([30]), prop_h)

NUTS_gamma = NUTS(positive_logpost_wrap, 2,0, par_dict['gamma_t'].detach().numpy(), 'gamma_t', par_dict, start_eps=0.053)

NUTS_gamma.sample()


num_samples = 2000
samples_dict = {'c':np.zeros((num_samples,K)),'gamma':np.zeros((num_samples,K)), 'B':np.zeros((num_samples,W)), 'height':np.zeros((num_samples))}

# initial sample
NUTS_gamma.sample(override_M=2, override_Madapt=0)
NUTS_B.sample(override_M=2, override_Madapt=0)
metropolisC.sample(override_M=1)
metropolisH.sample(override_M=1)

samples_dict['c'][0,:] = metropolisC.samples[0]
samples_dict['height'][0] = metropolisH.samples[0]
samples_dict['B'][0,:] = NUTS_B.samples[1,:]
samples_dict['gamma'][0,:] = NUTS_gamma.samples[1,:]


for s in range(1,num_samples):
    NUTS_gamma.sample(override_M=2, override_Madapt=0, override_theta0=samples_dict['gamma'][s,:])
    NUTS_B.sample(override_M=2, override_Madapt=0, override_theta0=samples_dict['B'][s,:])
    metropolisC.sample(override_M=1, override_theta0=samples_dict['c'][s,:])
    metropolisH.sample(override_M=1, override_theta0=np.array([samples_dict['height'][s]]))

    samples_dict['c'][s, :] = metropolisC.samples[0]
    samples_dict['height'][s] = metropolisH.samples[0]
    samples_dict['B'][s, :] = NUTS_B.samples[1, :]
    samples_dict['gamma'][s,:] = NUTS_gamma.samples[1, :]

    par_dict['c_t'] = fs.inv_gen_sigmoid(torch.from_numpy(samples_dict['c'][s,:]), W, 0.025)
    par_dict['gamma_t'] = torch.from_numpy(samples_dict['gamma'][s,:]).double()
    par_dict['height_t'] = fs.inv_gen_sigmoid(torch.from_numpy(np.array(samples_dict['height'][s])), 1000, 0.007).double()
    par_dict['B_t'] = torch.from_numpy(samples_dict['B'][s,:])
#gamma_arr = torch.linspace(1e-8,30,1000).double()
#logpgamma = np.zeros(len(gamma_arr))

#for idx, g in enumerate(gamma_arr):
#    val, _ = positive_logpost_wrap(torch.log(g.unsqueeze(0)).numpy(), 'gamma_t', par_dict)
#    logpgamma[idx] = val