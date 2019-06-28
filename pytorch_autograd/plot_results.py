import torch
import numpy as np
import matplotlib.pyplot as plt


B_samples = np.zeros((num_samples, W))
c_samples = np.zeros((num_samples,K))
g_samples = np.zeros((num_samples, K))
h_samples = np.zeros(num_samples)
beta_samples = np.zeros((num_samples, N))
alpha_samples = np.zeros((num_samples, K*N))

steep
l_base

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

