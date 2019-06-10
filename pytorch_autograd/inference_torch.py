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
from implementation.pytorch_autograd.nuts import NUTS


def log_posterior(X, eta_t, alpha_t, c_t, gamma_t, beta_t, B_t, tau_t, height_t, steep_t, delta_t, w, K):

    W, N = X.size()

    alpha0 = torch.tensor(15.0)
    mu_c = torch.tensor(W/2.)
    tau_c = torch.tensor(.1)

    beta0 = torch.tensor(.5)
    a_tau = torch.tensor(7.5)
    b_tau = torch.tensor(1.0)



    l_base = torch.tensor(1e-6).double().requires_grad_(False)

    # parameter transformations8
    alpha = torch.exp(alpha_t).reshape(K,N)
    gamma = torch.exp(gamma_t)
    eta = fs.general_sigmoid(eta_t, 1, 1)
    beta = torch.exp(beta_t)
    c = fs.general_sigmoid(c_t, W, 0.025)
    tau = torch.exp(tau_t)
    height = fs.general_sigmoid(height_t, 100, 0.07)
    steep = fs.general_sigmoid(steep_t, 75, 0.25)
    delta = torch.exp(delta_t)

    l = fs.length_scale(c,gamma,steep,w,height, base=l_base)

    sigma = tau


    covB = fs.gibbs_kernel(w,l,sigma)
    cholB = torch.cholesky(covB)


    B = torch.mv(cholB, B_t)




    # likelihood
    V = fs.pseudo_voigt(w,c,gamma,eta)
    I = torch.mm(V,alpha) + torch.ger(B,beta)

    ll = torch.distributions.normal.Normal(I, 1/tau).log_prob(X).sum()

    prior_alpha = torch.distributions.exponential.Exponential(alpha0).log_prob(alpha).sum() + alpha_t.sum()
    prior_gamma = fs.truncated_normal_lpdf(gamma, torch.tensor(10.).double(), torch.tensor(1.0/10.0).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum() + \
        gamma_t.sum()

    prior_beta = torch.distributions.exponential.Exponential(beta0).log_prob(beta).sum() + beta_t.sum()
    prior_tau = torch.distributions.gamma.Gamma(a_tau,b_tau).log_prob(tau).sum() + tau_t
    prior_eta = torch.log(fs.dgen_sigmoid(eta_t, 1,1)).sum()
    prior_height = torch.log(fs.dgen_sigmoid(height_t, 100,0.07)).sum()
    prior_steep = fs.truncated_normal_lpdf(steep, torch.tensor(25.0).double(),torch.tensor(5.).double(),
                                           torch.tensor(0.).double(),torch.tensor(75.).double()) + torch.log(fs.dgen_sigmoid(steep_t, 75, 0.25))
    prior_B = -0.5 * torch.dot(B_t,B_t)
    prior_c = fs.truncated_normal_lpdf(c, mu_c, 1.0 / tau_c, 0, torch.tensor(W).double()).sum() + torch.log(fs.dgen_sigmoid(c_t, W, 0.025)).sum()

    prior_delta = fs.truncated_normal_lpdf(delta, torch.tensor(10.0).double(), torch.tensor(1.0/10.0).double(), torch.tensor(0.0).double(), torch.tensor(float('Inf')).double()).sum()+\
        delta_t.sum()

    logpost = ll + prior_alpha + prior_gamma + prior_beta + prior_tau + prior_eta + \
              prior_height + prior_B + prior_c + prior_steep + prior_delta

    return logpost

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
                       par_dict['steep_t'], par_dict['delta_t'], par_dict['w'], par_dict['K'])

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



    return -ll.detach().numpy(), -grads.detach().numpy()


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



if __name__ == '__main__':
    # TODO: Check om den løsning vi gerne vil have har bedre logpost end den lærte
    # TODO: Hold parametre konstante enkeltvis - det er meget følsomt overfor startgæt i c - baseline begynder at forklare peaks
    #

    optimize = True
    torch_optim = False


    mats = loadmat('/home/david/Documents/Universitet/5_aar/PseudoVoigtMCMC/implementation/data/25x25x300_K1_2hot.mat')
    X = torch.from_numpy(mats['X'].T).double()
    W, N = X.size()
    K = 1

    np.random.seed(5)
    eta_t = torch.from_numpy(np.random.uniform(-10, 10, size=(K))).double().requires_grad_(True)
    alpha_t = torch.from_numpy(np.random.exponential(1, size=(K * N))).double().requires_grad_(True)
    a, b = (0 - W / 2) / 100, (W - W / 2) / 100
    c_t = torch.from_numpy(truncnorm.rvs(a, b, 100, size=(K))).requires_grad_(True)
    #c_t = torch.tensor([260]).double().requires_grad_(True)
    gamma_t = torch.from_numpy(np.random.exponential(.75, size=(K))).double().requires_grad_(True)
    beta_t = torch.from_numpy(np.random.exponential(5., size=(N))).double().requires_grad_(True)
    B_t = torch.from_numpy(np.random.normal(0, 1, size=W)).double().requires_grad_(True)
    tau_t = torch.tensor(np.random.gamma(0.2, 0.3)).double().requires_grad_(True)
    height_t = torch.tensor(np.random.uniform(0.01, 10)).double().requires_grad_(True)
    delta_t = torch.distributions.Exponential(.75).sample((K,)).double().requires_grad_(True)

    a, b = (0 - 25) / 100, (50 - 25) / 10
    steep_t = torch.tensor(truncnorm.rvs(a, b, 25, 10)).double().requires_grad_(True)

    w = torch.from_numpy(np.array(list(range(W)))).double()


    # test forward pass

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
        'delta_t' : delta_t,
        'X' : X,
        'w' : w,
        'K' : K
    }

    opt_pars = ['eta_t', 'alpha_t', 'c_t', 'gamma_t', 'beta_t','B_t', 'tau_t', 'height_t', 'steep_t', 'delta_t']



    dims = [par_dict[name].shape for name in opt_pars]

    if optimize:
        max_iter = 100
        count = 0
        for i in range(max_iter):
            print(f"\n\nIteration {i+1} of {max_iter}\n")
            # iterate over variables to be optimized over
            for opt_str in opt_pars:
                print(f"optimizing {opt_str}")

                opt_res = minimize(logpost_wrap, par_dict[opt_str].detach().numpy(), args=(opt_str, par_dict), method='L-BFGS-B',
                                   jac=True)
                #print(par_dict['c_t'])
                if not opt_res.success:
                    count += 1
                    print(f"Error when optimizing {opt_str}:\n Failed with message: {opt_res.message}\n")
                else:
                    # update value
                    par_dict[opt_str] = torch.from_numpy(opt_res.x).requires_grad_(True)
        print(f"Done with {count} errors\n")

        # Transform parameters back
        # parameter transformations
        alpha = torch.exp(par_dict['alpha_t']).reshape(K, N)
        gamma = torch.exp(par_dict['gamma_t'])
        eta = fs.general_sigmoid(par_dict['eta_t'], 1, 1)
        beta = torch.exp(par_dict['beta_t'])
        c = fs.general_sigmoid(par_dict['c_t'], W, 0.025)
        #c = par_dict['c_t']
        tau = torch.exp(par_dict['tau_t']).detach().numpy()
        height = fs.general_sigmoid(par_dict['height_t'], 100, 0.07)
        steep = fs.general_sigmoid(par_dict['steep_t'], 75, 0.25)
        #steep = par_dict['steep_t']
        l = fs.length_scale(c, gamma, steep, w, height)
        covB = fs.gibbs_kernel(w.double(), l.double(), torch.tensor(1).double())
        cholB = torch.cholesky(covB)

        B = torch.mv(cholB, par_dict['B_t'])

        delta = torch.exp(par_dict['delta_t'])

        V = fs.pseudo_voigt(w.double(), c, gamma, eta)
        I = torch.mm(V,alpha) + torch.ger(B, beta)

        X_prob = torch.distributions.Normal(I, torch.tensor(1/tau)).log_prob(X)
        plt.plot(V.detach().numpy())
        plt.title('Pseudo voigt component')
        plt.show()

        plt.plot(l.detach().numpy())
        plt.title('Learned length scale')
        plt.show()

        plt.plot(B.detach().numpy())
        plt.title('Learned background')
        plt.show()
    elif torch_optim:
        pars = np.array([par_dict[name].detach().numpy() for name in opt_pars])
        pars_cat = np.array([])
        for par in pars:
            try:
                pars_cat = np.concatenate((pars_cat, par))
            except ValueError: # ugh
                pars_cat = np.concatenate((pars_cat, [par]))


        # do sampling
        #sampler = NUTS(logpost_wrap_all, 3000, 1500, pars_cat)

        #ll, grads = logpost_wrap_all(pars_cat)
        #sampler.sample()

        optimizer = LBFGS([par_dict['eta_t'], par_dict['alpha_t'], par_dict['c_t'], par_dict['gamma_t'],
                        par_dict['beta_t'], par_dict['B_t'], par_dict['tau_t'], par_dict['height_t'],
                       par_dict['steep_t'], par_dict['delta_t']], lr=2)

        opt_tensors = [par_dict['eta_t'], par_dict['alpha_t'], par_dict['c_t'], par_dict['gamma_t'],
                        par_dict['beta_t'], par_dict['B_t'], par_dict['tau_t'], par_dict['height_t'],
                       par_dict['steep_t'], par_dict['delta_t']]
        for i in range(20):
            print(f"Iteration {i}\n")
            def closure():
                optimizer.zero_grad()
                ll = -log_posterior(par_dict['X'], par_dict['eta_t'], par_dict['alpha_t'], par_dict['c_t'], par_dict['gamma_t'],
                        par_dict['beta_t'], par_dict['B_t'], par_dict['tau_t'], par_dict['height_t'],
                       par_dict['steep_t'], par_dict['delta_t'], par_dict['w'], par_dict['K'])
                #print(ll)
                #print(fs.general_sigmoid(par_dict['c_t'], W, 0.025))
                ll.backward()
                #print([tensor.grad for tensor in opt_tensors])
                return ll
            optimizer.step(closure)

        alpha = torch.exp(par_dict['alpha_t']).reshape(K, N)
        gamma = torch.exp(par_dict['gamma_t'])
        eta = fs.general_sigmoid(par_dict['eta_t'], 1, 1)
        beta = torch.exp(par_dict['beta_t'])
        c = fs.general_sigmoid(par_dict['c_t'], W, 0.025)
        # c = par_dict['c_t']
        tau = torch.exp(par_dict['tau_t']).detach().numpy()
        height = fs.general_sigmoid(par_dict['height_t'], 100, 0.07)
        steep = fs.general_sigmoid(par_dict['steep_t'], 75, 0.25)
        # steep = par_dict['steep_t']
        l = fs.length_scale(c, gamma, steep, w, height)
        covB = fs.gibbs_kernel(w.double(), l.double(), torch.tensor(1).double())
        cholB = torch.cholesky(covB)

        B = torch.mv(cholB, par_dict['B_t'])

        delta = torch.exp(par_dict['delta_t'])

        V = fs.pseudo_voigt(w.double(), c, gamma, eta)
        I = torch.mm(V, alpha) + torch.ger(B, beta)

        plt.plot(V.detach().numpy())
        plt.title('Pseudo voigt component')
        plt.show()

        plt.plot(l.detach().numpy())
        plt.title('Learned length scale')
        plt.show()

        plt.plot(B.detach().numpy())
        plt.title('Learned background')
        plt.show()
    else:
        pars = np.array([par_dict[name].detach().numpy() for name in opt_pars])
        pars_cat = np.array([])
        for par in pars:
            try:
                pars_cat = np.concatenate((pars_cat, par))
            except ValueError:  # ugh
                pars_cat = np.concatenate((pars_cat, [par]))

        opt_res = minimize(logpost_wrap_all, pars_cat, method='L-BFGS-B', jac=True)
        assert opt_res.success
        unpacked_pars = unpack_pars(opt_res.x, dims)


        unpacked_pars_dict = {name : unpacked_pars[idx] for idx, name in enumerate(opt_pars)}

        alpha = torch.exp(unpacked_pars_dict['alpha_t']).reshape(K, N)
        gamma = torch.exp(unpacked_pars_dict['gamma_t'])
        eta = fs.general_sigmoid(unpacked_pars_dict['eta_t'], 1, 1)
        beta = torch.exp(unpacked_pars_dict['beta_t'])
        c = fs.general_sigmoid(unpacked_pars_dict['c_t'], W, 0.025)
        tau = torch.exp(unpacked_pars_dict['tau_t'])
        height = fs.general_sigmoid(unpacked_pars_dict['height_t'], 100, 0.07)
        steep = fs.general_sigmoid(unpacked_pars_dict['steep_t'], 75, 0.25)

        l = fs.length_scale(c, gamma, steep, w, height)
        covB = fs.gibbs_kernel(w, l, torch.tensor(1).double())
        cholB = torch.cholesky(covB)

        B = torch.mv(cholB, unpacked_pars_dict ['B_t'])

        delta = torch.exp(unpacked_pars_dict ['delta_t'])

        V = fs.pseudo_voigt(w, c, gamma, eta)
        I = torch.mm(V, alpha) + torch.ger(B, beta)

        plt.plot(V.detach().numpy())
        plt.title('Pseudo voigt component')
        plt.show()

        plt.plot(l.detach().numpy())
        plt.title('Learned length scale')
        plt.show()

        plt.plot(B.detach().numpy())
        plt.title('Learned background')
        plt.show()

        plt.matshow(I.detach().numpy())
        plt.title('I')
        plt.show()


