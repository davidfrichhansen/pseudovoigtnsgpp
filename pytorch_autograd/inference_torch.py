import torch
from torch.autograd import grad
import time
import funcsigs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import implementation.pytorch_autograd.aux_funcs_torch as fs
from scipy.io import loadmat
from scipy.optimize import minimize
from implementation.pytorch_autograd.nuts import NUTS, Metropolis


def logpost_wrap(par_value, par_name, other_pars):
    # wraps the objective function for par_name
    names = funcsigs.signature(log_posterior).parameters.keys()
    par_dict = {n : None for n in names}
    par_tensor = torch.from_numpy(par_value).requires_grad_(True)
    # forward pass
    for n in names:
        if n == par_name:
            par_dict[n] = par_tensor
        else:
            par_dict[n] = other_pars[n]

    ll = log_posterior(par_dict['X'], par_dict['eta_t'], par_dict['alpha_t'], par_dict['c_t'], par_dict['gamma_t'],
                        par_dict['beta_t'], par_dict['B_t'], par_dict['tau_t'], par_dict['height_t'],
                       par_dict['steep_t'], par_dict['w'], par_dict['K'])

    # backprop
    par_grad = grad(ll, par_tensor)[0] # par_value is tensor, which is why this works
    ll_detach = ll.detach()
    grad_detach = par_grad.detach()
    return -ll_detach.numpy(), -grad_detach.numpy()

def logpost_wrap_all(pars_np, dims, pars_dict):
    """
    DEPRECATED!
    :param pars_np:
    :param dims:
    :param pars_dict:
    :return:
    """


    # must be in this order
    # opt_pars = ['beta_t', 'B_t', 'steep_t', 'height_t', 'gamma_t', 'alpha_t', 'eta_t','c_t']
    pars = unpack_pars(pars_np, dims)
    #print(pars[4])
    ll = log_posterior(X, pars[6], pars[5], pars[7], pars[4], pars[0], pars[1], pars_dict['tau_t'], pars[3], pars[2], w, K)
    grads_split = grad(ll, pars)
    grads = torch.cat(grads_split, 0)
    #print(torch.max(torch.abs(grads)))



    return ll.detach().numpy(), grads.detach().numpy()


def unpack_pars(pars_np, dims):
    """
    DEPRECATED
    :param pars_np:
    :param dims:
    :return:
    """
    # takes a long array of parameters and returns a tuple of tensors
    # assumes 'dims' is a global list of shapes of the parameters in order
    pars = list()

    i = 0

    for idx,shapes in enumerate(dims):
        param_len = int(np.product(shapes))
        pars_tmp = pars_np[i:(i+param_len)]

        pars.append(torch.from_numpy(pars_tmp).double().requires_grad_(True))

        i += param_len

    return tuple(pars)


def positive_logpost_wrap(par_value, par_name, other_pars):
    # wraps the objective function for par_name
    names = funcsigs.signature(log_posterior).parameters.keys()
    par_dict = {n: None for n in names}
    par_tensor = torch.from_numpy(par_value).requires_grad_(True)
    # forward pass
    for n in names:
        if n == par_name:
            par_dict[n] = par_tensor
        else:
            par_dict[n] = other_pars[n]

    ll = log_posterior(par_dict['X'], par_dict['eta_t'], par_dict['alpha_t'], par_dict['c_t'], par_dict['gamma_t'],
                       par_dict['beta_t'], par_dict['B_t'], par_dict['tau_t'], par_dict['height_t'],
                       par_dict['steep_t'], par_dict['w'], par_dict['K'])

    # backprop
    par_grad = grad(ll, par_tensor)[0]  # par_value is tensor, which is why this works
    ll_detach = ll.detach()
    grad_detach = par_grad.detach()
    return ll_detach.numpy(), grad_detach.numpy()


#l_base = torch.tensor(5).double().requires_grad_(False)

def log_posterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w, K, ):

    W, N = X.size()
    mean_TB = torch.tensor([0.0]).double()
    alpha0 = torch.tensor(0.2)
    mu_c = torch.tensor(W/2.)
    tau_c = torch.tensor(0.005)

    #beta0 = torch.tensor(1.0)
    a_tau = torch.tensor(7.5)
    b_tau = torch.tensor(1.0)

    l_base = torch.tensor(5).double().requires_grad_(False)




    # parameter transformations8
    alpha = torch.exp(alpha_t).reshape(K,N)
    gamma = torch.exp(gamma_t)
    eta = fs.general_sigmoid(eta_t, 1, 1)
    beta = torch.exp(beta_t)

    c = fs.general_sigmoid(c_t, W, 0.025)
    tau = torch.exp(tau_t)
    height = fs.general_sigmoid(height_t, 1000, 0.007)
    steep = fs.general_sigmoid(steep_t, 2.0, 1.0)
    l = fs.length_scale(c, 5*gamma,steep,w,height, base=l_base)

    sigma = tau


    covB = fs.gibbs_kernel(w,l,sigma)

    cholB = torch.cholesky(covB)


    B = torch.mv(cholB, B_t) + mean_TB
    #plt.figure()
    #plt.plot(B.detach().numpy())
    #plt.title('inference')
    #plt.show()



    # likelihood
    V = fs.pseudo_voigt(w,c,gamma,eta)
    I = torch.mm(V,alpha) + torch.ger(B,beta)

    ll = torch.distributions.normal.Normal(I, 1/tau).log_prob(X).sum()

    prior_alpha = torch.distributions.exponential.Exponential(alpha0).log_prob(alpha).sum() + alpha_t.sum()
    #prior_alpha = fs.truncated_normal_lpdf(alpha, torch.tensor(5.0).double(), torch.tensor(1.5).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum() + \
    #alpha_t.sum()
    prior_gamma = fs.truncated_normal_lpdf(gamma, torch.tensor(10.).double(), torch.tensor(1.0/3.0).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum() + \
        gamma_t.sum()

    prior_beta = fs.truncated_normal_lpdf(beta, torch.tensor(0.5), torch.tensor(1.0), 0, torch.tensor(float('Inf')).double()).sum() + beta_t.sum()
    prior_tau = torch.distributions.gamma.Gamma(a_tau,b_tau).log_prob(tau).sum() + tau_t
    prior_eta = torch.log(fs.dgen_sigmoid(eta_t, 1,1)).sum()
    #  torch.distributions.normal.Normal(torch.tensor(20.0), torch.tensor(5.0)).log_prob(height) +
    prior_height = torch.log(fs.dgen_sigmoid(height_t, 1000,0.007)).sum()
    prior_steep = fs.truncated_normal_lpdf(steep, torch.tensor(0.2).double(),torch.tensor(.5).double(),
                                           torch.tensor(0.).double(),torch.tensor(5.).double()) + torch.log(fs.dgen_sigmoid(steep_t, 2.0, 1.0))
    prior_B = -0.5 * torch.dot(B_t,B_t)
    prior_c = fs.truncated_normal_lpdf(c, mu_c, 1.0 / tau_c, 0, torch.tensor(W).double()).sum() + torch.log(fs.dgen_sigmoid(c_t, W, 0.025)).sum()
    #prior_c = torch.log(fs.dgen_sigmoid(c_t, W, 0.025)).sum()

    logpost = ll + prior_alpha + prior_gamma + prior_beta + prior_tau + prior_eta + \
              prior_height + prior_B + prior_c + prior_steep

    return logpost



if __name__ == '__main__':


    optimize = False



    mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/25x25x300_K1_2hot_noisy.mat')
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
    #X = torch.from_numpy(X).double()
    tV = torch.from_numpy(true_vp.T)


    alpha_t = torch.log(ta)
    gamma_t = torch.log(tgamma)
    c_t = fs.inv_gen_sigmoid(tc, W, 0.025)
    eta_t = fs.inv_gen_sigmoid(teta, 1, 1)
    tau_t = torch.log(tsig)
    beta_t = torch.log(tbeta)

    height_t = torch.unsqueeze(fs.inv_gen_sigmoid(torch.tensor(500.0), 1000, 0.007),0).double()
    steep_t = torch.unsqueeze(fs.inv_gen_sigmoid(torch.tensor(0.2),2, 1), 0).double()
    #delta_t = torch.log(torch.tensor(15.0))

    lt = fs.length_scale(tc, 5*tgamma, torch.tensor(0.2), w, torch.tensor(20.0))
    covB = fs.gibbs_kernel(w, lt, tsig)

    cholB = torch.cholesky(covB)

    cholInv = torch.inverse(cholB)

    B_t = torch.mv(cholInv, tB)


    w = torch.from_numpy(np.array(list(range(W)))).double()




    par_dict = {
        'eta_t' : eta_t,
        'alpha_t' : alpha_t,
        'c_t' : c_t,
        'gamma_t' : gamma_t,
        'beta_t' : beta_t,
        'B_t' : B_t,
        'tau_t' : tau_t,
        'height_t' : height_t,
        'steep_t' : steep_t,
        'X' : X,
        'w' : w,
        'K' : K
    }

    opt_pars = ['beta_t','B_t', 'steep_t', 'height_t', 'gamma_t', 'alpha_t', 'eta_t', 'tau_t','c_t']



    if optimize:
        max_iter = 25
        count = 0
        for i in range(max_iter):
            print(f"\n\nIteration {i+1} of {max_iter}\n")
            # iterate over variables to be optimized over
            for opt_str in opt_pars:
                print(f"optimizing {opt_str}")

                opt_res = minimize(logpost_wrap, par_dict[opt_str].detach().numpy(), args=(opt_str, par_dict), method='L-BFGS-B',
                                   jac=True)
                if not opt_res.success:
                    count += 1
                    print(f"Error when optimizing {opt_str}:\n Failed with message: {opt_res.message}\n")
                else:
                    # update value
                    par_dict[opt_str] = torch.from_numpy(opt_res.x).requires_grad_(True)
            print(torch.exp(par_dict['alpha_t']).max())

        print(f"Done with {count} errors\n")


        # Transform parameters back
        # parameter transformations
        alpha = torch.exp(par_dict['alpha_t']).reshape(K, N)
        gamma = torch.exp(par_dict['gamma_t'])
        eta = fs.general_sigmoid(par_dict['eta_t'], 1, 1)
        beta = torch.exp(par_dict['beta_t'])
        c = fs.general_sigmoid(par_dict['c_t'], W, 0.025)
        #c = par_dict['c_t']
        tau = torch.exp(par_dict['tau_t'])
        height = fs.general_sigmoid(par_dict['height_t'], 1000, 0.007)
        steep = fs.general_sigmoid(par_dict['steep_t'], 2.0, 1.0)

        l = fs.length_scale(c, 5*gamma, steep, w, height, base=l_base)
        covB = fs.gibbs_kernel(w.double(), l.double(), tau)
        cholB = torch.cholesky(covB)

        B = torch.mv(cholB, par_dict['B_t'])


        V = fs.pseudo_voigt(w, c, gamma, eta)
        I = torch.mm(V,alpha) + torch.ger(B, beta)

        X_prob = torch.distributions.Normal(I, torch.tensor(1/tau)).log_prob(X)


        # Plots

        plt.matshow(I.detach().numpy())
        plt.colorbar()
        plt.title('I')
        plt.show()

        plt.matshow(X.numpy())
        plt.colorbar()
        plt.title('X')
        plt.show()

        plt.plot(V.detach().numpy())
        plt.plot(tV.numpy())
        plt.legend(['Learned', 'True'])
        plt.title('Pseudo voigt')
        plt.xlabel('Wavenumber')
        plt.show()

        plt.plot(alpha.detach().numpy().T)
        plt.title('Learned amplitudes (alpha)')
        plt.xlabel('Wavenumber')
        plt.show()

        plt.plot(l.detach().numpy())
        plt.title('Learned length scale')
        plt.show()

        plt.plot(B.detach().numpy())
        plt.plot(tB.numpy())
        plt.legend(['Learned', 'True'])
        plt.title('Background')
        plt.axvline((c - 5*gamma).detach().numpy(), lw=2, ls='--')
        plt.axvline((c + 5*gamma).detach().numpy(), lw=2, ls='--')
        plt.xlabel('Wavenumber')
        plt.show()


        plt.plot(beta.detach().numpy())
        plt.title('Learned betas')
        plt.show()
    else: # sample

        # TODO: Sample one at a time with NUTS, c with M-H
        """  
        0.5) Make new wrapper
        1) Make epsilon dictionary - for this to work, we must assume that we are already in an area of high prob for initial guess -> problems with burnin
        2) Do dual averaging for each variable independently
        3) Store in epsilon dictionary
        4) Draw 1 (!!) sample for each variable with their respective epsilon
        5) Save samples in array
        6) M-H on c, eta and height (uniform proposals)
        7) Do not learn tau
        """

        """
        Accept (c, gamma, alpha) as a tuple
        Look at proposals for c in metropolis
    
        
        """


        if True:
            ### DUAL AVERAGING
            eps_dict = {name : None for name in opt_pars if name != 'c_t' and name != 'height_t' and name != 'eta_t' and name != 'tau_t'}

            M_adapt = 250
            t1 = time.time()
            for name in eps_dict.keys():
                print(f"\n\n ---- DUAL AVERAGING FOR {name} ----\n\n")
                sampler = NUTS(positive_logpost_wrap, 1,1, par_dict[name].detach().numpy().ravel(), name, par_dict)
                # run dual averaging
                sampler.sample(override_M=M_adapt+1, override_Madapt=M_adapt, plot_eps=False)
                eps_dict[name] = sampler.epsilon
                print(eps_dict[name])

            print(f"---- FINISHED DUAL AVERAGING IN {time.time() - t1} SECONDS ---- \n\n")


            ### SAMPLING
            num_samples = 2000
            sample_dict = {name : np.zeros((num_samples, len(par_dict[name].detach().numpy().ravel()))) for name in opt_pars if name != 'tau_t' and name != 'height_t'}
            NUTS_dict = {name : NUTS(positive_logpost_wrap, 2, 0, par_dict[name].detach().numpy().ravel(), name, par_dict, start_eps=eps_dict[name]) for name in eps_dict.keys()}



            def logp_c(c):
                c_t = fs.inv_gen_sigmoid(torch.from_numpy(c), W, 0.025).detach().numpy()
                val, _ = positive_logpost_wrap(c_t, 'c_t', par_dict)
                return val

            def prop_c(c):
                #return np.random.normal(tc.numpy(),0.001,size=len(c))
                return np.random.uniform(0,W, size=len(c))

            def logp_eta(eta):
                eta_t = fs.inv_gen_sigmoid(torch.from_numpy(eta), 1,1).detach().numpy()
                val, _ = positive_logpost_wrap(eta_t, 'eta_t', par_dict)

                return val

            def prop_eta(eta):
                return np.random.uniform(0,1, size=len(eta))

            metropolisC = Metropolis(logp_c, fs.general_sigmoid(par_dict['c_t'], W, 0.025).detach().numpy(), prop_c)
            metropolisEta = Metropolis(logp_eta, fs.general_sigmoid(par_dict['eta_t'], 1,1).detach().numpy(), prop_eta)


            # initial sample
            for name, sampler in NUTS_dict.items():
                sampler.sample()
                sample_dict[name][0,:] = sampler.samples[1,:]

            metropolisC.sample(override_M=1)
            metropolisEta.sample(override_M=1)

            sample_dict['c_t'][0,:] = fs.inv_gen_sigmoid(torch.from_numpy(metropolisC.samples[0,:]), W, 0.025).numpy()
            sample_dict['eta_t'][0,:] = fs.inv_gen_sigmoid(torch.from_numpy(metropolisEta.samples[0,:]),1,1).numpy()

            # remaining samples
            for s in range(1,num_samples):
                # NUTS
                for name, sampler in NUTS_dict.items():
                    sampler.sample(override_theta0=sample_dict[name][s,:])
                    sample_dict[name][s,:] = sampler.samples[1,:]
                    par_dict[name] = torch.tensor(sampler.samples[1,:]).double()

                metropolisC.sample(override_M=1, override_theta0=fs.general_sigmoid(torch.from_numpy(sample_dict['c_t'][s,:]), W, 0.025).detach().numpy())
                metropolisEta.sample(override_M=1, override_theta0=fs.general_sigmoid(torch.from_numpy(sample_dict['eta_t'][s,:]),1,1).detach().numpy())
                c_sample = fs.inv_gen_sigmoid(torch.from_numpy(metropolisC.samples[0, :]), W, 0.025).numpy()
                print(f"c: {c_sample}")
                sample_dict['c_t'][s, :] = c_sample
                sample_dict['eta_t'][s, :] = fs.inv_gen_sigmoid(torch.from_numpy(metropolisEta.samples[0, :]), 1, 1).numpy()
                par_dict['c_t'] = torch.tensor(c_sample).double()
                par_dict['eta_t'] = fs.inv_gen_sigmoid(torch.from_numpy(metropolisEta.samples[0, :]), 1, 1)



            # save samples to file
            np.save('samples2.npy', sample_dict)
            np.save('epsilons2.npy', eps_dict)

"""
plt.axis([-50, 50, 0, 10000])
plt.ion()
plt.show()

x = np.arange(-50, 51)
for pow in range(1, 5):  # plot x^1, x^2, ..., x^4
    y = [Xi ** pow for Xi in x]
    myp = plt.plot(x, y)
    plt.draw()
    plt.pause(1)
    myp[0].remove()    
"""