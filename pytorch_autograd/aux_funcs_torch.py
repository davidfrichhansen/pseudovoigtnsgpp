import torch
import numpy as np



def gibbs_kernel(x,l,sigma):
    W = x.size()[0]

    lij = torch.ger(l,l) # outer product
    ll = l*l
    l_sums_1 = ll.view(-1,1).repeat(1,W).view(W,W) # torch equivalent to ll.repeat(W,axis=0).reshape(W,W)
    l_sums = l_sums_1 + torch.t(l_sums_1)

    x_rep = x.view(-1,1).repeat(1,W).view(W,W)

    ximj = x_rep - torch.t(x_rep)
    ximjsq = ximj*ximj

    prefactor = torch.sqrt(2 * lij / l_sums)
    exponential = torch.exp(-ximjsq / l_sums)

    K = sigma*sigma*prefactor*exponential + 1e-6*torch.eye(W).double()

    return K



def length_scale(c,gamma,steep,w,height,base=1e-6):
    K = c.size()[0]
    W = w.size()[0]

    l = base*torch.ones(W).double()

    for idx, k in enumerate(range(0,2*K-1, 2)):
        endpoint1 = c[idx] - gamma[idx]
        endpoint2 = c[idx] + gamma[idx]
        l = l + height * (torch.tanh((w - endpoint1)*steep) - torch.tanh((w-endpoint2)*steep))

    return l


def pseudo_voigt(w,c,gamma,eta):
    W = w.size()[0]
    K = c.size()[0]

    wc = (w.view(-1,1).repeat(1,K).view(W,K) - c)**2

    gamma_arr = gamma.view(-1,1).repeat(1,W).view(W,K)

    L = gamma_arr * np.pi**(-1) / (wc + gamma_arr*gamma_arr)
    G = 1.0 / (np.sqrt(2 * np.pi) * gamma_arr) * torch.exp(-wc / (2 * gamma_arr * gamma_arr))
    V = eta * L + (1-eta) * G

    return V


def general_sigmoid(x,L,k):
    val = L*torch.sigmoid(k*x)
    return val

def dgen_sigmoid(x,L,k):
    grad = torch.exp(-k*x)*k*L / ((1+torch.exp(-k*x))**2)
    return grad
