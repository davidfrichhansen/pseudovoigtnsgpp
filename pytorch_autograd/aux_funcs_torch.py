import torch
import numpy as np


##### NON STATIONARY KERNEl
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

    K = sigma*sigma*prefactor*exponential + 1e-4*torch.eye(W).double()

    return K

def length_scale(c,gamma,steep,w,height,base=1e-6):
    K = c.size()[0]
    W = w.size()[0]

    l = base*torch.ones(W).double()

    for idx, k in enumerate(range(0,2*K-1, 2)):
        endpoint1 = (c[idx] - gamma[idx]).double()
        endpoint2 = (c[idx] + gamma[idx]).double()
        l = l + height.double() * (torch.tanh((w.double() - endpoint1)*steep.double()) - torch.tanh((w.double()-endpoint2)*steep.double()))

    return l



##### PSEUDO VOIGT
def pseudo_voigt(w,c,gamma,eta):
    W = w.size()[0]
    K = c.size()[0]

    wc = (w.view(-1,1).repeat(1,K).view(W,K) - c)**2

    gamma_arr = gamma.view(-1,1).repeat(1,W).view(W,K)

    L = gamma_arr * np.pi**(-1) / (wc + gamma_arr*gamma_arr)
    G = 1.0 / (np.sqrt(2 * np.pi) * gamma_arr) * torch.exp(-wc / (2 * gamma_arr * gamma_arr))
    V = eta * L + (1-eta) * G

    return V



##### DISTRIBUTIONS
def truncated_normal_lpdf(x, mu, sigma, a, b):
    dist = torch.distributions.Normal(mu.float(),sigma.float())
    assert a < b
    if torch.isinf(b.double()):
        b_tmp = torch.tensor(1.0).double()
    else:
        b_tmp = dist.cdf(b)

    probs =  dist.log_prob(x) - (torch.log(sigma) + torch.log(b_tmp - dist.cdf(a)))
    probs[x.double() >= b] = torch.log(torch.tensor(0.0))
    probs[x.double() <= a] = torch.log(torch.tensor(0.0))
    return probs.double()

def truncated_normal_pdf(x,mu,sigma,a,b):
    num = torch.rsqrt(torch.tensor(2*np.pi))*torch.exp(-0.5 * ((x-mu) / sigma)**2)
    dist = torch.distributions.Normal(mu.float(), sigma.float())
    den = sigma*(dist.cdf(b)-dist.cdf(a))
    probs = num / den
    probs[x.double() <= a] = torch.tensor(0.0)
    probs[x.double() >= b] = torch.tensor(0.0)
    return probs

##### TRANSFORMATIONS

def general_sigmoid(x,L,k):
    val = L*torch.sigmoid(k*x)
    return val

def dgen_sigmoid(x,L,k):
    grad = torch.exp(-k*x)*k*L / ((1+torch.exp(-k*x))**2)
    return grad


#### LINK FUNCTIONS
def exp_to_gauss(h,sigma, lambda_):

    inner = 1e-12 + .5 - .5*torch.erf(h / (np.sqrt(2) *sigma))

    val = torch.max((-1.0 / lambda_) * torch.log(inner), torch.tensor(0).double())
    val[val != val] = 0 # Remove nans
    #grad =

    return val

