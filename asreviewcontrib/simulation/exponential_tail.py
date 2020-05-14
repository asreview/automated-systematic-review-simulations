
from scipy.stats import norm
import numpy as np


class PowerTailNorm():
    def __init__(self, mu, sigma, tail_surface, alpha, t_p):
        self.mu = mu
        self.sigma = sigma

        self.tail_surface = tail_surface

        self.alpha = alpha
        t_p = t_p

        transition_x = mu-t_p*sigma
        base_surf_norm = norm.cdf(transition_x, loc=mu, scale=sigma)
        norm_prefac = (1-tail_surface)/(1-base_surf_norm)
        transition_y = norm_prefac*norm.pdf(mu-t_p*sigma, loc=mu, scale=sigma)

        d = transition_x - (tail_surface * (1-alpha))/transition_y
        self.c = transition_y/(d-transition_x)**-alpha
        self.d = d
        self.transition_x = transition_x
        self.norm_prefac = norm_prefac

    def pdf(self, x):
        power_idx = np.where(x < self.transition_x)[0]
        norm_idx = np.where(x >= self.transition_x)[0]

        ret_pdf = np.zeros(x.shape)
        ret_pdf[power_idx] = self.c*(self.d-x[power_idx])**-self.alpha
        ret_pdf[norm_idx] = self.norm_prefac*norm.pdf(x[norm_idx], loc=self.mu,
                                                      scale=self.sigma)
        return ret_pdf


class ExpTailNorm():
    def __init__(self, mu, sigma, transition_point):
        self.mu = mu
        self.sigma = sigma

        transition_x = -transition_point
        base_surf_norm = norm.cdf(transition_x, loc=0, scale=1)
        transition_y_prime = norm.pdf(transition_x, loc=0, scale=1)

        dnorm = -transition_y_prime*transition_x

        tau = transition_y_prime/dnorm
        c = transition_y_prime*np.exp(-transition_x/tau)
        tail_surface = tau*c*np.exp(transition_x/tau)
        norm_prefac = 1/(tail_surface+(1-base_surf_norm))
        c *= norm_prefac

        self.c = c
        self.tau = tau
        self.transition_x = transition_x
        self.norm_prefac = norm_prefac
        self.tail_surface = tail_surface*norm_prefac

    def pdf(self, x):
        xt = (x-self.mu)/self.sigma
        exp_idx = np.where(xt < self.transition_x)[0]
        norm_idx = np.where(xt >= self.transition_x)[0]

        ret_pdf = np.zeros(xt.shape)
        ret_pdf[exp_idx] = self.c*np.exp(xt[exp_idx]/self.tau)
        if len(np.where(ret_pdf == np.inf)[0]):
            print(self.tau)
        ret_pdf[norm_idx] = self.norm_prefac*norm.pdf(xt[norm_idx])
        return ret_pdf/self.sigma

    def cdf(self, x):
        xt = (x-self.mu)/self.sigma
        exp_idx = np.where(xt < self.transition_x)[0]
        norm_idx = np.where(xt >= self.transition_x)[0]

        ret_cdf = np.zeros(xt.shape)
        ret_cdf[exp_idx] = self.c*self.tau*np.exp(xt[exp_idx]/self.tau)

        kwargs = {"loc": 0, "scale": 1}
        norm_cdf = norm.cdf(xt[norm_idx], **kwargs) - norm.cdf(
            self.transition_x, **kwargs)
        ret_cdf[norm_idx] = self.tail_surface + self.norm_prefac * norm_cdf

        return ret_cdf
