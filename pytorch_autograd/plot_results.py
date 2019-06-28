import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import implementation.pytorch_autograd.aux_funcs_torch as fs

# load samples and data
samples_dict = np.load('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/base60_250samples_goodnonoise.npy').item()
mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/25x25x300_K1_2hot.mat')
X = torch.from_numpy(mats['X'].T).double()
W, N = X.size()

gen = mats['gendata']

K = 1

w = torch.arange(X.shape[0]).double()

true_alpha = gen['A'][0][0]
true_c = gen['theta'][0][0][0][0]
true_eta = gen['theta'][0][0][0][2]
true_gamma = gen['theta'][0][0][0][1]
true_B = gen['B'][0][0]
true_beta = gen['b'][0][0]
true_sigma = gen['sig'][0][0]
true_vp = fs.pseudo_voigt(w, torch.tensor([true_c]).double(), torch.tensor([true_gamma]).double(), torch.tensor([true_eta]).double())

num_samples = samples_dict['height'].shape[0]

K = 1
steep = torch.tensor(0.1).double()
l_base = torch.tensor(60).double()
tsig = torch.tensor(1.0).double()

B_samples = np.zeros((num_samples, W))
c_samples = np.zeros((num_samples,K))
g_samples = np.zeros((num_samples, K))
h_samples = np.zeros(num_samples)

alpha_samples = np.exp(samples_dict['alpha'])
beta_samples = np.exp(samples_dict['beta'])
eta_samples = samples_dict['eta']


V_samples = np.zeros((num_samples, W))

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

    V_samples[s,:] = fs.pseudo_voigt(w,cur_c, cur_g, torch.tensor(np.array(eta_samples[s,:]))).numpy().ravel()


idx = [-5, -10,-50,-100,-150, -200]

plt.figure()

for i in idx:
    plt.plot(B_samples[idx].T)
    plt.title('Samples of background')
    plt.xlabel('Wavenumber')


plt.figure()
plt.plot(true_vp.numpy(), '--', lw=3)
for i in idx:
    plt.plot(V_samples[i])
    plt.title('Recovered Pseudo-voigt component')
    plt.xlabel('Wavenumber')

plt.figure()
plt.plot(true_alpha, '--', lw=3)
for i in idx:
    plt.plot(alpha_samples[i,:])
    plt.title('Samples of amplitudes')
    plt.xlabel('Observation number')

plt.figure()
plt.plot(true_beta, '--', lw=3)
plt.title('Samples of background contributions')
plt.xlabel('Observation number')
for i in [-50]:
    plt.plot(beta_samples[i,:])


# plot single observation voigt
most_sig = np.argmax(true_alpha)
plt.figure()
plt.plot(X[:,most_sig].numpy())
plt.title('Observation with most signal and samples and alpha * V')
plt.xlabel('Wavenumber')
for i in idx:
    plt.plot(alpha_samples[i,most_sig] * V_samples[i,:])