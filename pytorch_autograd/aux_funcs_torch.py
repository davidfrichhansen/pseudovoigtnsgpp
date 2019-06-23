import torch
import numpy as np
import matplotlib.pyplot as plt


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

    K = sigma*sigma*prefactor*exponential + 1e-8*torch.eye(W).double()

    return K

def length_scale(c,gamma,steep,w,height,base=1e-6):
    K = c.size()[0]
    W = w.size()[0]

    l = base*torch.ones(W).double()

    for idx, k in enumerate(range(0,2*K-1, 2)):
        endpoint1 = (c[idx] - gamma[idx])
        endpoint2 = (c[idx] + gamma[idx])
        l = l + (height) * (torch.sigmoid((w - endpoint1)*steep) - torch.sigmoid((w-endpoint2)*steep))

    return l


def discont_length_scale(c,gamma,w,height,base=1e-6):
    K = c.size()[0]
    W = w.size()[0]

    l = base*torch.ones(W).double()

    for idx,k in enumerate(range(0,2*K-1,2)):
        cond1 = w.double() >= (c[idx] - gamma[idx])
        cond2 =w.double() <= (c[idx] + gamma[idx])
        l[cond1 & cond2] = height

    return l

l = discont_length_scale(torch.tensor([150,250]).double(), torch.tensor([10,20]).double(), torch.arange(300).double(), torch.tensor(50.0), torch.tensor(5.0))
def plot_gp_samples(w,c,gamma,steep,height,base,sigma):
    l = length_scale(c,gamma,steep,w,height,base)

    K = gibbs_kernel(w,l,sigma)

    plt.plot(l.numpy())
    plt.title('Length scale')
    plt.show()

    #samples = torch.cholesky(K).numpy() @ np.random.randn(w.size()[0], 1)
    samples = np.random.multivariate_normal(np.zeros(300), K.numpy(), size=3)

    plt.plot(samples.T)
    plt.axvline(c - gamma, lw=2, ls='--', c='k')
    plt.axvline(c + gamma, lw=2, ls='--', c='k')
    plt.title('Prior samples')
    plt.show()


##### PSEUDO VOIGT
def pseudo_voigt(w,c,gamma,eta):
    W = w.size()[0]
    K = c.size()[0]

    #wc = (w.view(-1,1).repeat(1,K).view(W,K) - c) / gamma.view(-1,1).repeat(1,W).view(W,K)
    xdata = w.repeat(K,1)
    c_arr = torch.unsqueeze(c,1).repeat(1,W)
    gamma_arr = torch.unsqueeze(gamma,1).repeat(1,W)
    eta_arr = torch.unsqueeze(eta,1).repeat(1,W)

    diff = xdata - c_arr
    kern = diff / gamma_arr

    diff2 = kern*kern

    L = 1.0 / (1.0 + diff2)

    G_kern = -torch.log(torch.tensor(2.0)) * diff2

    G = torch.exp(G_kern)

    V = eta_arr * L + (1-eta_arr) * G

    return torch.t(V)



##### DISTRIBUTIONS
def truncated_normal_lpdf(x, mu, sigma, a, b):
    dist = torch.distributions.Normal(mu.float(),sigma.float())
    assert a < b
    if torch.isinf(b):
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

def inv_gen_sigmoid(x,L,k):
    val = 1/ k * (torch.log(x) - torch.log(L-x))
    return val

def dinv_gen_sigmoid(x,L,k):
    val = L / (k*x*(L-x))
    return val

"""

def inv_logistic(p, L, k, x0=0):
    val = 1 / k * (np.log(p) + k*x0 - np.log(L-p))
    return val
"""


#### LINK FUNCTIONS
def exp_to_gauss(h,sigma, lambda_):

    inner = 1e-12 + .5 - .5*torch.erf(h / (np.sqrt(2) *sigma))

    val = torch.max((-1.0 / lambda_) * torch.log(inner), torch.tensor(0).double())
    val[val != val] = 0 # Remove nans
    #grad =

    return val

