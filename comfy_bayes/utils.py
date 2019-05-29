from math import lgamma, log
from comfy_bayes.bayes_conversion import ConversionTest


def lbeta(alpha, beta):
    """lbeta distribution approx. through lgamma"""
    return lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)



