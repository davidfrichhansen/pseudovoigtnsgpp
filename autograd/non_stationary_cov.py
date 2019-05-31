import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic


def gibbs_kernel(x, l, sigma):
    W = len(x)
    K = np.zeros((W,W))
    for i in range(W):
        for j in range(W):
            prefactor = np.sqrt(2*l[i]*l[j] / (l[i]*l[i] + l[j]*l[j]))
            exponential = np.exp(- (x[i] - x[j])*(x[i] - x[j]) / (l[i]*l[i] + l[j]*l[j]))
            K[i,j] = sigma*sigma * prefactor * exponential
    K += 1e-6*np.eye(N)
    return K


def length_scale(c, gamma, steep, w, height, base=1e-6):
    K = len(c)
    W = len(w)
    l = base*np.ones(W)
    for idx,k in enumerate(range(0, 2*K-1, 2)):
        endpoints1 = c[idx] - gamma[idx]
        endpoints2 = c[idx] + gamma[idx]
        l = l + height * (np.tanh((w - endpoints1)*steep) - np.tanh((w-endpoints2)*steep))

    return l



N = 300
x = np.linspace(-3, 3, N)

sigma = 1


steep = 25
height = 2
base = 0.05

c = [-2,-0.5,0.5,2]
gamma = [0.5,0.25,0.25,0.5]



l = length_scale(c,gamma,steep,x,height,base)
plt.plot(x,l)
plt.show()


cov = gibbs_kernel(x,l,sigma)

L = np.linalg.cholesky(cov)

samples = L@np.random.normal(size=(N,5))

plt.imshow(cov)
plt.show()

plt.plot(x,samples)

plt.show()

