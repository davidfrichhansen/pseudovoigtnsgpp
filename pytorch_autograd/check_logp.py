import numpy as np
import torch
from implementation.pytorch_autograd.inference_torch import log_posterior
import implementation.pytorch_autograd.aux_funcs_torch as fs
from scipy.io import loadmat
import matplotlib.pyplot as plt



mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/25x25x300_K1_2hot.mat')

X = mats['X'].T
gen = mats['gendata']
W,N = X.shape

K = 1

true_alpha = gen['A'][0][0]
true_vp = gen['vp'][0][0]
true_c = gen['c'][0][0]
true_eta = gen['eta_voigt'][0][0]
true_gamma = gen['gamma'][0][0]
true_B = gen['B'][0][0]
true_beta = gen['b'][0][0]
true_sigma = gen['sig'][0][0]

plt.plot(true_vp.T)
plt.title('True Voigt')
plt.show()

plt.plot(true_alpha)
plt.title('True alpha')
plt.show()

plt.plot(true_B)
plt.title('True background')
plt.show()

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
X = torch.from_numpy(X).double()
tV = torch.from_numpy(true_vp.T)


#my_vp = fs.pseudo_voigt(w, tc, tgamma, teta)
#my_vp = my_vp / torch.max(my_vp)

true_I = torch.mm(tV, ta) + torch.ger(tB,tbeta)

my_I = torch.mm(fs.pseudo_voigt(w,tc,tgamma,teta), ta) + torch.ger(tB,tbeta)

#plt.imshow(my_I)
#plt.title('I')

#plt.axis('square')
#plt.show()
#plt.axis('square')
plt.imshow(X)
plt.title('X')
plt.show()

plt.imshow(true_I)
plt.title('True I')
plt.show()


residuals = my_I - X ### hmmm.... => Normalization

plt.imshow(my_I)
plt.title('Computed I')
plt.show()

plt.imshow(residuals)
plt.title('Residuals on I')
plt.show()

tat = torch.log(ta)
tgammat = torch.log(tgamma)
tct = fs.inv_gen_sigmoid(tc, W, 0.025)
tetat = fs.inv_gen_sigmoid(teta, 1,1)
taut = torch.log(tsig)
tbetat = torch.log(tbeta)


heightt = fs.inv_gen_sigmoid(torch.tensor(70.0), 100, 0.07)
steept = fs.inv_gen_sigmoid(torch.tensor(50.0), 75, 0.25)
deltat = torch.log(torch.tensor(15.0))

lt = fs.length_scale(tc, tgamma, torch.tensor(50.0), w, torch.tensor(70.0))
covB = fs.gibbs_kernel(w, lt, tsig)

cholB = torch.cholesky(covB)

cholInv = torch.inverse(cholB)

tBt = torch.mv(cholInv, tB)

ll = -log_posterior(X, tetat, tat,tct,tgammat,tbetat,tBt, taut, heightt, steept, deltat, w, K)

