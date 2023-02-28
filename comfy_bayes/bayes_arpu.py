import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ARPUTest:
    def __init__(self,
                 revenues_a, converted_players_a, total_players_a,
                 revenues_b, converted_players_b, total_players_b,
                 sample_size=1_000_000):

        self.evaluated = False

        self.revenues_a = revenues_a
        self.converted_players_a = converted_players_a
        self.total_players_a = total_players_a

        self.revenues_b = revenues_b
        self.converted_players_b = converted_players_b
        self.total_players_b = total_players_b

        self.sample_size = sample_size

        # measured
        self.conversion_a = converted_players_a / total_players_a
        self.conversion_b = converted_players_b / total_players_b

        self.arpu_a = revenues_a / total_players_a
        self.arpu_b = revenues_b / total_players_b

        self.arppu_a = revenues_a / converted_players_a
        self.arppu_b = revenues_b / converted_players_b

        self.alpha_a = converted_players_a + 1
        self.alpha_b = converted_players_b + 1

        self.beta_a = total_players_a - converted_players_a + 1
        self.beta_b = total_players_b - converted_players_b + 1

        self.k_a = converted_players_a + 1
        self.k_b = converted_players_b + 1

        self.theta_a = revenues_a + 1
        self.theta_b = revenues_b + 1

    def evaluate(self):
        lambda_a = np.random.beta(self.alpha_a, self.beta_a, self.sample_size)
        lambda_b = np.random.beta(self.alpha_b, self.beta_b, self.sample_size)

        omega_a = np.random.gamma(self.k_a, 1/self.theta_a, self.sample_size)
        omega_b = np.random.gamma(self.k_b, 1/self.theta_b, self.sample_size)

        self.sampled_arpu_a_ = (lambda_a / omega_a)
        self.sampled_arpu_b_ = (lambda_b / omega_b)

        self.sampled_arppu_a_ = (1 / omega_a)
        self.sampled_arppu_b_ = (1 / omega_b)
        
        self.sampled_conversion_a_ = lambda_a
        self.sampled_conversion_b_ = lambda_b
        
        self.sampled_conversion_diff_ = (lambda_b - lambda_a)
        self.sampled_arppu_diff_ = (1/omega_b - 1/omega_a)
        self.sampled_arpu_diff_ = ((lambda_b/omega_b) - (lambda_a/omega_a))

        self.hdi_arpu_ = self.hdi_of_mcmc(self.sampled_arpu_diff_)

        # arpu
        self.probability_arpu_b_better_than_a_ = (self.sampled_arpu_b_[self.sampled_arpu_b_ > self.sampled_arpu_a_].size
                                                  / self.sample_size)

        self.expected_arpu_uplift_b_over_a_ = (self.sampled_arpu_diff_[self.sampled_arpu_diff_ > 0].sum()
                                               / self.sample_size)

        self.expected_arpu_loss_b_over_a_ = (-self.sampled_arpu_diff_[-self.sampled_arpu_diff_ > 0].sum()
                                             / self.sample_size)
        
        # arppu
        self.probability_arppu_b_better_than_a_ = (self.sampled_arppu_b_[self.sampled_arppu_b_ > self.sampled_arppu_a_].size / self.sample_size)
        self.expected_arppu_uplift_b_over_a_ = (self.sampled_arppu_diff_[self.sampled_arppu_diff_ > 0].sum() / self.sample_size)
        self.expected_arppu_loss_b_over_a_ = (-self.sampled_arppu_diff_[-self.sampled_arppu_diff_ > 0].sum() / self.sample_size)
 
        # conversion
        self.probability_conversion_b_better_than_a_ = (self.sampled_conversion_b_[self.sampled_conversion_b_ > self.sampled_conversion_a_].size / self.sample_size)
        self.expected_conversion_uplift_b_over_a_ = (self.sampled_conversion_diff_[self.sampled_conversion_diff_ > 0].sum() / self.sample_size)
        self.expected_conversion_loss_b_over_a_ = (-self.sampled_conversion_diff_[-self.sampled_conversion_diff_ > 0].sum() / self.sample_size)
        
        return self

    @staticmethod
    def hdi_of_mcmc(chain, cred_mass=.95):
        """https://stats.stackexchange.com/questions/252988/highest-density-interval-in-stan"""
        # sort chain using the first axis which is the chain
        chain.sort()
        # how many samples did you generate?
        samples_size = chain.size
        # how many samples must go in the HDI?
        nSampleCred = int(np.ceil(samples_size * cred_mass))
        # number of intervals to be compared
        nCI = samples_size - nSampleCred
        # width of every proposed interval
        width = np.array(list(chain[i+nSampleCred-1] - chain[i] for  i in range(nCI)))
        # index of lower bound of shortest interval (which is the HDI)
        best  = width.argmin()
        # put it in a dictionary
        hdi = {'lower': chain[best], 'upper': chain[best + nSampleCred], 'width': width.min()}
        return hdi


    def stats():
        pass

    # @staticmethod
    def draw_sampled_distribution_of_arpu_difference(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        x = self.sampled_arpu_diff_
        hdi = self.hdi_of_mcmc(self.sampled_arpu_diff_)

        N, bins, patches = ax.hist(x, 100, lw=1, edgecolor="white", );
        ax.axvline(0, c="k", lw=2, alpha=.5)

        for i in range(len(bins[bins <= hdi["lower"]])):
            patches[i].set_alpha(.33)

        for i in range(len(bins[bins >= hdi["upper"]])):
            patches[-i-1].set_alpha(.33)

        return ax

    # @staticmethod
    def draw_probability_distributions(ab_test, metric="arpu", ax=None, bins=100):
        if ax is None:
            fig, ax = plt.subplots()

        if metric == "arpu":
            prop_a = ab_test.sampled_arpu_a_
            prop_b = ab_test.sampled_arpu_b_
        else:
            raise ValueError('metric not recognised')

        ha = np.histogram(prop_a, bins=bins)
        hb = np.histogram(prop_b, bins=bins)

        xa, ya = ha[1][1:], ha[0]
        xb, yb = hb[1][1:], hb[0]

        ax.plot(xa, ya)
        ax.plot(xb, yb)

        hdi_bounds_a = ab_test.hdi_of_mcmc(ab_test.sampled_arpu_a_)
        hdi_a = np.logical_and(hdi_bounds_a["lower"] <= xa, xa <= hdi_bounds_a["upper"])

        hdi_bounds_b = ab_test.hdi_of_mcmc(ab_test.sampled_arpu_b_)
        hdi_b = np.logical_and(hdi_bounds_b["lower"] <= xb, xb <= hdi_bounds_b["upper"])

        ax.fill_between(xa, y1=0, y2=ya, where=hdi_a, alpha=.5)
        ax.fill_between(xb, y1=0, y2=yb, where=hdi_b, alpha=.5)

        return ax

    def draw_sampled_distribution_of_arpu_difference(ab_test, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        x = ab_test.sampled_arpu_diff_
        hdi = ab_test.hdi_of_mcmc(ab_test.sampled_arpu_diff_)

        N, bins, patches = ax.hist(x, 100, lw=1, edgecolor="white", );
        ax.axvline(0, c="k", lw=2, alpha=.5)

        for i in range(len(bins[bins <= hdi["lower"]])):
            patches[i].set_alpha(.33)

        for i in range(len(bins[bins >= hdi["upper"]])):
            patches[-i-1].set_alpha(.33)

        return ax
