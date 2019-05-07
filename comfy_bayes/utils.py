from math import lgamma, log


def lbeta(alpha, beta):
    """lbeta distribution approx. through lgamma"""
    return lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)

