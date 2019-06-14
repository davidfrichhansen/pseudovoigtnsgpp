import torch
from torch.autograd import grad
from torch.optim import LBFGS
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

def logpost_wrap_all(pars_np):
    pars = unpack_pars(pars_np, dims)

    ll = log_posterior(X, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], w, K)
    grads_split = grad(ll, pars)
    grads = torch.cat(grads_split, 0)



    return ll.detach().numpy(), grads.detach().numpy()


def unpack_pars(pars_np, dims):
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



l_base = torch.tensor(5).double().requires_grad_(False)

def log_posterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, w, K):

    W, N = X.size()

    alpha0 = torch.tensor(0.2)
    mu_c = torch.tensor(W/2.)
    tau_c = torch.tensor(0.05)

    #beta0 = torch.tensor(1.0)
    a_tau = torch.tensor(7.5)
    b_tau = torch.tensor(1.0)




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


    B = torch.mv(cholB, B_t)




    # likelihood
    V = fs.pseudo_voigt(w,c,gamma,eta)
    I = torch.mm(V,alpha) + torch.ger(B,beta)

    ll = torch.distributions.normal.Normal(I, 1/tau).log_prob(X).sum()

    prior_alpha = torch.distributions.exponential.Exponential(alpha0).log_prob(alpha).sum() + alpha_t.sum()
    #prior_alpha = fs.truncated_normal_lpdf(alpha, torch.tensor(5.0).double(), torch.tensor(1.5).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum() + \
    #alpha_t.sum()
    prior_gamma = fs.truncated_normal_lpdf(gamma, torch.tensor(10.).double(), torch.tensor(1.0/10.0).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum() + \
        gamma_t.sum()

    prior_beta = fs.truncated_normal_lpdf(beta, torch.tensor(0.5), torch.tensor(0.2), 0, torch.tensor(float('Inf')).double()).sum() + beta_t.sum()
    prior_tau = torch.distributions.gamma.Gamma(a_tau,b_tau).log_prob(tau).sum() + tau_t
    prior_eta = torch.log(fs.dgen_sigmoid(eta_t, 1,1)).sum()
    #  torch.distributions.normal.Normal(torch.tensor(20.0), torch.tensor(5.0)).log_prob(height) +
    prior_height = torch.log(fs.dgen_sigmoid(height_t, 1000,0.007)).sum()
    prior_steep = fs.truncated_normal_lpdf(steep, torch.tensor(0.2).double(),torch.tensor(.5).double(),
                                           torch.tensor(0.).double(),torch.tensor(5.).double()) + torch.log(fs.dgen_sigmoid(steep_t, 2.0, 1.0))
    prior_B = -0.5 * torch.dot(B_t,B_t)
    prior_c = fs.truncated_normal_lpdf(c, mu_c, 1.0 / tau_c, 0, torch.tensor(W).double()).sum() + torch.log(fs.dgen_sigmoid(c_t, W, 0.025)).sum()

    #prior_delta = fs.truncated_normal_lpdf(delta, torch.tensor(10.0).double(), torch.tensor(1.0/10.0).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum()+\
    #   delta_t.sum()

    logpost = ll + prior_alpha + prior_gamma + prior_beta + prior_tau + prior_eta + \
              prior_height + prior_B + prior_c + prior_steep #+ prior_delta

    return logpost



if __name__ == '__main__':
    # TODO: Check om den løsning vi gerne vil have har bedre logpost end den lærte
    # TODO: Hold parametre konstante enkeltvis - det er meget følsomt overfor startgæt i c - baseline begynder at forklare peaks
    #

    optimize = False



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
    #X = torch.from_numpy(X).double()
    tV = torch.from_numpy(true_vp.T)


    alpha_t = torch.log(ta)
    gamma_t = torch.log(tgamma)
    c_t = fs.inv_gen_sigmoid(tc, W, 0.025)
    eta_t = fs.inv_gen_sigmoid(teta, 1, 1)
    tau_t = torch.log(tsig)
    beta_t = torch.log(tbeta)

    height_t = fs.inv_gen_sigmoid(torch.tensor(500.0), 1000, 0.007)
    steep_t = fs.inv_gen_sigmoid(torch.tensor(0.2),2, 1)
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

    #opt_pars = ['eta_t', 'alpha_t', 'c_t', 'gamma_t', 'B_t', 'beta_t', 'tau_t', 'height_t', 'steep_t']
    opt_pars = ['beta_t','B_t', 'steep_t', 'height_t', 'gamma_t', 'alpha_t', 'eta_t']


    dims = [par_dict[name].shape for name in opt_pars]

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
        # pack parameters
        """
        pars = np.array([par_dict[name].detach().numpy() for name in opt_pars])
        pars_cat = np.array([])
        for par in pars:
            try:
                pars_cat = np.concatenate((pars_cat, par))
            except ValueError:  # ugh
                pars_cat = np.concatenate((pars_cat, [par]))

        """
        # try metropolis on c
        def prop_c(c):
            return np.random.uniform(0, W, size=K)


        def logp_c(c):
            c_t = fs.inv_gen_sigmoid(torch.from_numpy(c), W, 0.025).detach().numpy()

            val, _ = logpost_wrap(c_t, 'c_t', par_dict)

            return -val

        c_sampler = Metropolis(logp_c, torch.tensor([W/2.]).double().numpy(), prop_c, M=10000)
        c_sampler.sample()



