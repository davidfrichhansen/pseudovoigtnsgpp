
def prop_c(c):
    return np.random.uniform(0,W,size=len(c))
    #return np.random.multivariate_normal(c, 500*np.eye(len(c)))


def logp_c(c, par_dict):
    c_t = fs.inv_gen_sigmoid(torch.from_numpy(c), W, 0.025).detach().numpy()
    val, _ = positive_logpost_wrap(c_t, 'c_t', par_dict)
    return val

logpc_val = logp_c(tc.numpy(), par_dict)


c_arr = torch.arange(1,W).double()
logpc = np.zeros(W-1)

for i in range(W-1):
    logpc[i] = logp_c(c_arr[i].unsqueeze(0).numpy(),par_dict)

plt.figure()
plt.plot(c_arr.numpy(),logpc)
plt.xlabel('c')
plt.ylabel('log prob')
plt.title('p(c|X, alpha, beta, ...)')


metropolisC = Metropolis(logp_c, np.array([150.0]), prop_c, par_dict)

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



