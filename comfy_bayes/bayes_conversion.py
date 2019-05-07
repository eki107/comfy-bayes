import numpy as np
from scipy.stats import beta

from comfy_bayes.utils import lbeta

"""
resources:
- Vidov blog
- viacero A/B blog-veci
- evan miller 
"""


class ConversionTest:
    hdi_95_lower_a_: float
    hdi_95_upper_a_: float

    hdi_95_lower_b_: float
    hdi_95_upper_b_: float

    b_better_than_a_: float

    loss_: float
    uplift_: float

    def __init__(self, successes_a, total_a, successes_b, total_b):
        self.successes_a = successes_a
        self.successes_b = successes_b

        self.total_a = total_a
        self.total_b = total_b

    def evaluate(self):
        """Main function that evaluates the test"""
        alpha_param_a = self.successes_a + 1
        beta_param_a = self.total_a - self.successes_a + 1
        alpha_param_b = self.successes_b + 1
        beta_param_b = self.total_b - self.successes_b + 1

        # probability of B better than A
        self.b_better_than_a_ = self.calculate_b_better_than_a_probability(alpha_param_a,
                                                                           beta_param_a,
                                                                           alpha_param_b,
                                                                           beta_param_b)

        # 95% HDI for both A and B
        self.hdi_95_lower_a_, self.hdi_95_upper_a_ = self.calculate_hdi(self.successes_a, self.total_a)
        self.hdi_95_lower_b_, self.hdi_95_upper_b_ = self.calculate_hdi(self.successes_b, self.total_b)

        # expected loss and uplifts
        self.loss_ = self.expected_loss_b_over_a(alpha_param_a, beta_param_a, alpha_param_b, beta_param_b)
        self.uplift_ = self.expected_uplift_b_over_a(alpha_param_a, beta_param_a, alpha_param_b, beta_param_b)

        return self

    @staticmethod
    def calculate_b_better_than_a_probability(alpha_param_a: int, beta_param_a: int,
                                              alpha_param_b: int, beta_param_b: int) -> float:
        """
        Calculates probability for B better than A, according to
        https://www.evanmiller.org/bayesian-ab-testing.html#binary_ab_equivalent
        This implementation follows the formula equivalent marked 7th, which gave the best results for most of the cases
        :param alpha_param_a: α param for group A, number of successes in group A (minus one)
        :param beta_param_a: β param for group A, number of failures in group A (minus one)
        :param alpha_param_b: α param for group B, number of successes in group B  (minus one)
        :param beta_param_b: α param for group B, number of successes in group B (minus one)
        :return: probability of B better than A
        """
        total = 1.0
        for i in range(alpha_param_a):
            total -= np.exp(lbeta(alpha_param_b + i, beta_param_b + beta_param_a) - np.log(beta_param_a + i)
                            - lbeta(1 + i, beta_param_a) - lbeta(alpha_param_b, beta_param_b))

        return total

    @staticmethod
    def calculate_hdi(successes, total, alpha=.05):
        """
        Calculates HDI for beta function derived from the observation
        """
        lower_bound, upper_bound = alpha / 2, 1 - (alpha / 2)
        beta_func = beta(a=successes + 1, b=total - successes + 1)
        hdi_lower, hdi_upper = beta_func.ppf(lower_bound), beta_func.ppf(upper_bound)

        return hdi_lower, hdi_upper

    @staticmethod
    def expected_loss_b_over_a(alpha_param_a: int, beta_param_a: int, alpha_param_b: int, beta_param_b: int) -> float:
        """
        Calculates expected loss if the A/B test probability when choosing B over A (also works vice-versa)
        # sometimes the probability is so close to 100%,
        # that it yields P=1 which that ln of 0 gives -inf, this causes python to error
        :param alpha_param_a: α param for group A, number of successes in group A (minus one)
        :param beta_param_a: β param for group A, number of failures in group A (minus one)
        :param alpha_param_b: α param for group B, number of successes in group B  (minus one)
        :param beta_param_b: α param for group B, number of successes in group B (minus one)
        :return: expected loss
        """
        a_plus_one_better_then_b = ConversionTest.calculate_b_better_than_a_probability(alpha_param_a + 1,
                                                                                        beta_param_a,
                                                                                        alpha_param_b,
                                                                                        beta_param_b)

        a_better_than_b_plus_one = ConversionTest.calculate_b_better_than_a_probability(alpha_param_a,
                                                                                        beta_param_a,
                                                                                        alpha_param_b + 1,
                                                                                        beta_param_b)

        numerator = (
                lbeta(alpha_param_a + 1, beta_param_a)
                - lbeta(alpha_param_a, beta_param_a)
                + np.log(1 - a_plus_one_better_then_b)
        )

        denominator = (
                lbeta(alpha_param_b + 1, beta_param_b)
                - lbeta(alpha_param_b, beta_param_b)
                + np.log(1 - a_better_than_b_plus_one)
        )

        with np.errstate(divide="ignore"):
            return np.exp(numerator) - np.exp(denominator)

    @staticmethod
    def expected_uplift_b_over_a(alpha_param_a, beta_param_a, alpha_param_b, beta_param_b):
        return ConversionTest.expected_loss_b_over_a(alpha_param_b, beta_param_b, alpha_param_a, beta_param_a)
