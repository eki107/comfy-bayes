import numpy as np

import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt

from typing import Sequence

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

        self.sampled_conversion_diff_ = (lambda_b - lambda_a)
        self.sampled_arppu_diff_ = (1/omega_b - 1/omega_a)
        self.sampled_arpu_diff_ = ((lambda_b/omega_b) - (lambda_a/omega_a))

        self.hdi_arpu_ = self.hdi_of_mcmc(self.sampled_arpu_diff_)

        self.probability_arpu_b_better_than_a_ = (self.sampled_arpu_b_[self.sampled_arpu_b_ > self.sampled_arpu_a_].size
                                                  / self.sample_size)

        self.expected_arpu_uplift_b_over_a_ = (self.sampled_arpu_diff_[self.sampled_arpu_diff_ > 0].sum()
                                               / self.sample_size)

        self.expected_arpu_loss_b_over_a_ = (-self.sampled_arpu_diff_[-self.sampled_arpu_diff_ > 0].sum()
                                             / self.sample_size)

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
        ax = ax or plt.subplots()[1]

        ax.hist(self.sampled_arpu_diff_, 100, lw=1, edgecolor="white");
        ax.axvline(0, c="k", lw=2, alpha=.5)

    # @staticmethod
    def draw_arpu_probability_distributions(self, ax=None):
        ax = ax or plt.subplots()[1]

        ax.hist(self.sampled_arpu_a_, 100, lw=1, alpha=0.5);
        ax.hist(self.sampled_arpu_b_, 100, lw=1, alpha=0.5);


    def draw_probability_distributions(self, metric="arpu", ax=None):
        ax = ax or plt.subplots()[1]

        prop_a = f"sampled_{metric}_a"
        prop_b = f"sampled_{metric}_b"

        ax.hist(self[prop_a], 100, lw=1, alpha=0.5);
        ax.hist(self[prop_b], 100, lw=1, alpha=0.5);
