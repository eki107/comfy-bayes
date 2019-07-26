import numpy as np
from comfy_bayes.bayes_conversion import ConversionTest
import matplotlib.pyplot as plt
from seaborn.palettes import color_palette

cpal = color_palette("Set2")


def draw_probability_distributions(test: ConversionTest, ax=None):
    ax = ax or plt.gca()

    # get beta functions as pmf of our observations
    a_beta_func = test.beta_func_a
    b_beta_func = test.beta_func_b

    # get better ranges so our chart is on point
    top_x = max(a_beta_func.ppf(0.99999), b_beta_func.ppf(0.99999))
    bottom_x = min(a_beta_func.ppf(0.00001),  b_beta_func.ppf(0.00001))

    # get values for our chart
    x = np.linspace(bottom_x, top_x, 1000)
    hdi_a = np.logical_and(a_beta_func.ppf(.025) <= x, x <= a_beta_func.ppf(.975))
    hdi_b = np.logical_and(b_beta_func.ppf(.025) <= x, x <= b_beta_func.ppf(.975))

    # plot distribution for group A
    dplot = ax.plot(x, a_beta_func.pdf(x), color=cpal[0], label="A")
    ax.fill_between(x, y1=0, y2=a_beta_func.pdf(x), where=hdi_a, facecolor=cpal[0], alpha=.2, )

    # plot distribution for group B
    ax.plot(x, b_beta_func.pdf(x), c=cpal[1], label="B")
    ax.fill_between(x, y1=0, y2=b_beta_func.pdf(x), where=hdi_b, facecolor=cpal[1], alpha=.2, )

    # some other cosmetic options: hide Y axis which we do not need, set title and reformat % on X axis
    ax.yaxis.set_visible(False)
    ax.set_title("Conversion probability distributions of A and B, with 95% HDI", loc="right")
    ax.set_xticklabels(["{:,.2%}".format(tick) for tick in ax.get_xticks()])

    # add legend
    ax.legend()

    return dplot


def draw_approximate_distribution_of_difference(test, samples=1_000_000, alpha=.05, ax=None):
    ax = ax or plt.gca()

    i = alpha / 2, (1 - alpha / 2)

    data = test.beta_func_b.rvs(samples) - test.beta_func_a.rvs(samples)

    x = np.sort(data)
    n = len(x)
    N, bins, patches = ax.hist(x=x, bins=100)
    dplot = ax.axvline(0, lw=4, c='pink')

    hdi = x[int(i[0] * n) - 1], x[int(i[1] * n) - 1]

    for i in range(len(bins[bins <= hdi[0]])):
        patches[i].set_alpha(.33)

    for i in range(len(bins[bins >= hdi[1]])):
        patches[-i-1].set_alpha(.33)

    ax.yaxis.set_visible(False)
    ax.set_title(f"Approximate distribution off difference B-A (sampled from {samples:,d} draws", loc="right")
    ax.set_xticklabels(["{:,.2%}".format(tick) for tick in ax.get_xticks()])

    return dplot
