import numpy as np
import matplotlib.pyplot as plt
np.seterr('raise')

class NUTS:
    """
    Implements the efficient NUTS sampler with dual averaging (algorithm 6) from Hoffman and Gelman (2014)
    """

    def __init__(self, logp, M, M_adapt, theta0, *logp_args, delta=0.63, debug=False, delta_max=1000.0, start_eps=None):
        """
        Initialize
        :param logp:        Callable that takes parameters of target distribution and return log probability and
                            the gradient in that point in the form
                            logprob, grad = f(theta)
        :param M:           Number of desired samples
        :param M_adapt:     Amount of samples in which the dual averaging for estimating step size, espilon, should be run
        :param theta0:      Initial parameter value
        :param delta:       Desired acceptance rate for the sampler, used in the dual averaging
        :param debug:       Boolean determining verbosity - for debugging purposes
        :param delta_max:   Parameter determining precision. Default: 1000 as recommended by Hoffman and Gelman
        """
        self.logp = logp
        self.logparr = np.zeros((M,))
        self.M = M
        self.M_adapt = M_adapt
        self.theta0 = theta0
        self.delta = delta
        self.debug = debug
        self.delta_max = delta_max
        self.accepted = 0
        self.eps_list = np.zeros(M_adapt + 1)
        self.start_eps = start_eps
        self.logpargs = logp_args
        self.epsilon = None

        # maybe do some smart stuff, pickling, sqlite etc.
        self.samples = np.zeros((M, len(theta0)))
        self.samples[0, :] = theta0

    def leapfrog(self, theta, r, epsilon, grad):
        """
        Performs leapfrog integration
        :param theta:       Current position
        :param r:           Current momentum
        :param epsilon:     Step length
        :param grad:        Gradient evaluated in theta
        :return: new positition, new momentum, new gradient and new function value
        """
        f = self.logp

        r_bar = r + 0.5 * epsilon * grad
        theta_bar = theta + epsilon * r_bar
        # recompute gradient

        logp_new, grad_new = f(theta_bar, *self.logpargs)
        r_bar = r_bar + 0.5 * epsilon * grad_new

        return theta_bar, r_bar, grad_new, logp_new

    def epsilon_heuristic(self, theta, old_logp, old_grad):
        """
        Heuristic for finding initial value of step length epsilon
        :param theta:   Initial parameter value
        :param old_logp f(theta)
        :param old_grad gradient of f(theta)
        :return:        Reasonable value for initial epsilon
        """
        if self.debug:
            print("Enter reasonable epsilon")
        dim = len(theta)
        epsilon = 1
        r = np.random.randn(dim)
        # initial leapfrog
        _, r_prime, _, new_logp = self.leapfrog(theta, r, epsilon, old_grad)
        logp_ratio = new_logp - old_logp - 0.5 * r_prime.T @ r_prime + 0.5 * r.T @ r

        criteria = logp_ratio > np.log(.5)

        a = 1.0 if criteria else -1.0

        while a * logp_ratio > -a * np.log(2.0):
            epsilon *= 2.0 ** a
            _, r_prime, _, new_logp = self.leapfrog(theta, r, epsilon, old_grad)
            logp_ratio = new_logp - old_logp - 0.5 * r_prime.T @ r_prime + 0.5 * r.T @ r

        print("Find reasonable epsilon: %.4lf\n" % epsilon)

        return epsilon

    def build_tree(self, theta, r, log_u, v, j, epsilon, theta0, r0, old_logp, grad):
        """
        Implicitly builds balanced search tree recursively of visited (theta, r)-positions
        This is one of the main enhancements from regular HMC
        :param theta:       Latest parameter value
        :param r:           Latest momentum value
        :param log_u        log of slice variable
        :param v:           Direction (-1 or 1)
        :param j:           height of tree
        :param epsilon:     step length
        :param r0:          Initial momentum value
        :param old_logp:    self.f(theta0)
        :param grad:        gradient of self.f(theta)
        :return:            A bunch
        """

        delta_max = self.delta_max
        # base case - tree height is 0
        if j == 0:

            theta_prime, r_prime, grad_new, new_logp = self.leapfrog(theta, r, v * epsilon, grad)

            n_prime = 1 if log_u < new_logp - 0.5 * r_prime.T @ r_prime else 0

            s_prime = 1 if log_u < delta_max + new_logp - 0.5 * r_prime.T @ r_prime else 0

            return (theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime,
                    min(1, np.exp(new_logp - 0.5 * r_prime.T @ r_prime - old_logp + 0.5 * r0.T @ r0)),
                    1, grad_new, grad_new, grad_new)
        else:
            # main recursion
            (theta_m, r_m, theta_p, r_p,
             theta_prime, n_prime, s_prime, alpha_prime, n_alpha, grad_p, grad_m, grad_prime) = self.build_tree(theta,
                                                                                                                r,
                                                                                                                log_u,
                                                                                                                v,
                                                                                                                j - 1,
                                                                                                                epsilon,
                                                                                                                theta0,
                                                                                                                r0,
                                                                                                                old_logp,
                                                                                                                grad)
            if s_prime == 1:
                if v == -1:
                    (theta_m, r_m, _, _, theta_pp, n_pp,
                     s_pp, alpha_pp, n_alphapp, _, grad_m, grad_pp) = self.build_tree(theta_m, r_m, log_u, v, j - 1,
                                                                                      epsilon, theta0, r0,
                                                                                      old_logp, grad_m)
                else:
                    (_, _, theta_p, r_p, theta_pp, n_pp,
                     s_pp, alpha_pp, n_alphapp, grad_p, _, grad_pp) = self.build_tree(theta_p, r_p, log_u, v, j - 1,
                                                                                      epsilon, theta0, r0,
                                                                                      old_logp, grad_p)

                # draw bernoulli sample
                try:
                    bern = np.random.rand() < n_pp / (n_prime + n_pp)
                except ZeroDivisionError:
                    bern = False
                if bern:
                    theta_prime = theta_pp
                    grad_prime = grad_pp

                alpha_prime += alpha_pp
                n_alpha += n_alphapp

                s_prime = s_pp if ((theta_p - theta_m) @ r_m >= 0 and (theta_p - theta_m) @ r_p >= 0) else 0

                if self.debug:
                    print("s' = %d" % s_prime)

                n_prime += n_pp

            return theta_m, r_m, theta_p, r_p, theta_prime, n_prime, s_prime, alpha_prime, n_alpha, grad_p, grad_m, grad_prime

    def sample(self, override_M=None, override_Madapt=None, override_theta0=None, kappa=0.75, t0=10, plot_eps=True):
        """
        Run the NUTS sampling algorithm with dual averaging. Does not return samples but saves them in self.samples
        :param kappa    Parameter to be used in dual averaging
        :param t0       Initial value of t0 as described in paper
        :param plot_eps Determines if epsilon convergence should be plotted
        :return:        Nothing - sets self.samples directly
        """
        f = self.logp
        if override_M is not None:
            self.samples = np.zeros((override_M, len(self.theta0)))
            self.logparr = np.zeros(override_M)

        if override_Madapt is not None:
            self.eps_list = np.zeros(override_Madapt + 1)

        M = self.M if override_M is None else override_M
        M_adapt = self.M_adapt if override_Madapt is None else override_Madapt
        theta0 = self.theta0 if override_theta0 is None else override_theta0
        logp, grad = f(theta0, *self.logpargs)
        if self.start_eps is None:
            epsilon = self.epsilon_heuristic(theta0, logp, grad)
        else:
            print(f"Using fixed stepsize: {self.start_eps}\n")
            epsilon = self.start_eps

        mu = np.log(10 * epsilon)
        logeps_bar = 0
        H_bar = 0
        gamma = 0.05
        for m in range(1, M):
            if m % 10 == 0:
                print("%d iterations completed\n" % m)
                print("likelihood : %f" % logp)
                print(f"Epsilon: {epsilon}\n\n")
            # resample momentum
            r0 = np.random.randn(len(theta0))
            # evaluate probability
            logp, grad = f(self.samples[m - 1, :], *self.logpargs)
            # logp(theta, r)
            joint = logp - 0.5 * r0.T @ r0
            # if u ~ uniform(0, z), then log(u) ~ log(z) - exp(1), cf. Slice sampling paper by R.M. Neal
            log_u = joint - np.random.exponential(1, size=1)
            # initialize
            theta_m = self.samples[m - 1, :]
            theta_p = self.samples[m - 1, :]
            grad_m = grad_p = grad
            r_m = r0
            r_p = r0
            j = 0
            # proposed value of parameters
            theta_prop = self.samples[m - 1, :]
            n = s = 1

            while s:
                # choose direction uniformly
                v = np.random.choice([-1, 1])
                # Begin initial recursion
                if v == -1:
                    (theta_m, r_m, _, _, theta_prime, n_prime,
                     s_prime, alpha, n_alpha, _, grad_m, grad_prime) = self.build_tree(theta_m, r_m, log_u, v, j,
                                                                                       epsilon,
                                                                                       self.samples[m - 1, :], r0, logp,
                                                                                       grad_m)
                else:
                    (_, _, theta_p, r_p, theta_prime, n_prime,
                     s_prime, alpha, n_alpha, grad_p, _, grad_prime) = self.build_tree(theta_p, r_p, log_u, v, j,
                                                                                       epsilon,
                                                                                       self.samples[m - 1, :], r0, logp,
                                                                                       grad_p)
                if s_prime:
                    # accept sample
                    bern = np.random.rand() < min(1, n_prime / n)
                    if bern:
                        theta_prop = theta_prime
                        grad = grad_prime

                # stopping criteria
                n += n_prime
                s = s_prime if (theta_p - theta_m) @ r_m >= 0 and (theta_p - theta_m) @ r_p >= 0 else 0
                j = j + 1

            # dual averaging
            if m <= (M_adapt-1) and self.start_eps is None:
                H_bar = (1 - 1.0 / (m + t0)) * H_bar + 1 / (m + t0) * (self.delta - alpha / n_alpha)
                logeps = mu - np.sqrt(m) / gamma * H_bar
                epsilon = np.exp(logeps)
                self.eps_list[m] = epsilon
                logeps_bar = m ** (-kappa) * logeps + (1 - m ** (-kappa)) * logeps_bar

            else:
                if self.start_eps is None:
                    epsilon = np.exp(logeps_bar)

            if plot_eps and m == M_adapt and self.start_eps is None:
                plt.plot(self.eps_list)
                plt.title("Epsilon convergence during adaptation")
                plt.xlabel("Iteration")
                plt.ylabel("Epsilon")
                plt.show()

            if m == M_adapt:
                self.epsilon = epsilon
            # add proposal to sample list
            self.samples[m, :] = theta_prop
            self.logparr[m] = logp

        print("Epsilon was %lf" % epsilon)

        # remove burnin
        self.samples = self.samples[M_adapt:, :]
        print("Sampling finished!")
        print()
        print()


class Metropolis:

    def __init__(self, logp, theta0, proposal_dist, *logp_args,  M=1000, burn=0, thin=0):
        self.logp = logp
        self.proposal_dist = proposal_dist
        self.M = M
        self.theta0 = theta0
        self.thin = thin
        self.samples = np.zeros((M, len(theta0)))
        self.burn = burn
        self.acc_rate = 0
        self.logp_args = logp_args



    def sample(self, override_theta0=None, override_M=None, *prop_args):
        f = self.logp
        prop_dist = self.proposal_dist

        current = self.theta0 if override_theta0 is None else override_theta0
        M = self.M if override_M is None else override_M
        if override_M is not None:
            self.samples = np.zeros((override_M, len(current)))

        accept = 0
        for m in range(M):
            if m % 100 == 0:
                print(f"Iteration {m} of {M}")
            # current logp
            logp_old = f(current, *self.logp_args)

            # proposal
            proposal = prop_dist(current, *prop_args)

            logp_new = f(proposal, *self.logp_args)

            a = min(0, logp_new - logp_old)

            if np.random.rand() < np.exp(a):
                self.samples[m,:] = proposal
                current = proposal
                accept += 1
            else:
                self.samples[m,:] = current

            # remove burnin
            if self.burn != 0:
                self.samples = self.samples[self.burn:, :]

            if self.thin != 0:
                self.samples[::self.thin, :]

            self.acc_rate = accept / M



if __name__ == '__main__':
    # quick metropolis test
    a = 0
    b = 300

    A = np.array([
        [4.0,2],
        [2,3]
    ])
    mu = np.array([1,1])
    Ai = np.linalg.inv(A)
    def logp(x):
        return -0.5*(x-mu).T@Ai@(x-mu)

    def logp_grad(x):
        val = -0.5*(x-mu)@Ai@(x-mu)
        grad = -Ai@(x-mu)

        return val,grad

    def proposal(x):
        return np.random.uniform(-10,10,size=len(x))


    sampler = Metropolis(logp, np.array([0,0]).T, proposal, M=100000)

    sampler.sample()
    plt.scatter(sampler.samples[:,0], sampler.samples[:,1])
    plt.title('Metropolis')
    plt.show()


    sampler = NUTS(logp_grad, M=100000, M_adapt=10000, theta0=np.array([0,0]).T, start_eps=0.8)
    sampler.sample()

    plt.scatter(sampler.samples[:,0], sampler.samples[:,1])
    plt.title('NUTS')
    plt.show()
