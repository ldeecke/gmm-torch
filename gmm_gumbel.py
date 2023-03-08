import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR

from gmm import GaussianMixture


class GaussianMixtureGumbel(GaussianMixture):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, n_components, n_features, covariance_type="full", eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self.var = None
        self.mu = None
        self.initialize_params()
        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(
            1. / self.n_components)
        self.params_fitted = False

    def e_step_old(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        # self.estimate_log_prob is (n, k, 1)
        # self.pi is (1, k, 1)
        # weighted_log_prob is (n, k, 1)
        weighted_log_prob = self.estimate_log_prob(x) + torch.log(self.pi)

        # import ipdb; ipdb.set_trace()
        # log_prob_norm is (n, 1, 1) --> a separate "Z" for each datapoint
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)

        # Normalize by Z on a per-datapoint basis
        log_resp = weighted_log_prob - log_prob_norm

        # log_resp is (n, k, 1)
        return torch.mean(log_prob_norm), log_resp

    def e_step(self, x):
        x = self.check_size(x)
        logP_x_G_z_t = self.estimate_log_prob(x)
        logP_x0_G_z_t = logP_x_G_z_t[0]

        # Need to do dim=0 because now logP_x0_G_z_t is (k, 1)
        # Also need to include the prior!
        # import ipdb; ipdb.set_trace()
        true_logP_x0_G_t = torch.logsumexp(logP_x0_G_z_t + torch.log(self.pi).reshape(-1, 1), dim=0)
        print("True marginal for first datapoint:", true_logP_x0_G_t)
        V_lr = 1e-1
        V = torch.rand(1, requires_grad=True)
        V_optim = torch.optim.AdamW([V], lr=V_lr)
        #scheduler1 = StepLR(V_optim, 500, gamma=0.9)
        num_samples = 10000

        for iter_ in range(30000):
            # sample z
            # import ipdb; ipdb.set_trace()
            # self.pi is (1, k, 1)
            # NOTE: this sampling might be cheating because we normalize the probs (using the prior)
            z_index = torch.multinomial(self.pi[0, :, 0], num_samples=num_samples, replacement=True)
            # import ipdb; ipdb.set_trace()
            # (num_samples,)
            # import ipdb; ipdb.set_trace()
            V_optim.zero_grad()
            logPxGz = logP_x0_G_z_t[z_index].reshape(num_samples)
            loss = self.gumbel_stable_loss(logPxGz, V, beta=1, clip=None)
            # print("Loss:", loss)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(V, 1)
            V_optim.step()

            # import ipdb; ipdb.set_trace()
            # print("gradient", V.grad)

            #scheduler1.step()

            # Calculate difference from true log marginal
            # import ipdb; ipdb.set_trace()
            print("V:", V.data.numpy()[0], "grad: ", V.grad.data.numpy()[0])
            # print("grad:", V.grad)
            # print("Difference:", torch.abs(V - true_logP_x0_G_t))

    def gumbel_stable_loss(self, alpha, V, beta=1, clip=None):
        # alpha is (num_samples, )
        # V is (1,)
        # z is (num_samples,)
        z = (alpha - V) / beta
        #
        # if clip is not None:
        #     z = torch.clamp(z, max=clip)

        # max_z = torch.max(z)
        # # Clamp negative values to -1
        # max_z = torch.where(max_z < -10.0, torch.tensor(-10.0, dtype=torch.double), max_z)
        # max_z = max_z.detach()  # Detach the gradients

        # L = e^(alpha - V)/beta + V/beta - 1
        # scale by e^-max_z
        # loss = torch.exp(z - max_z) + (V / beta) * torch.exp(-max_z) - torch.exp(-max_z)

        # torch.exp(z) is (num_samples,)
        # import ipdb; ipdb.set_trace()
        loss = torch.exp((alpha - V)/beta) + (V / beta) - 1
        # print("Loss: ", loss.mean())
        return loss.mean()







