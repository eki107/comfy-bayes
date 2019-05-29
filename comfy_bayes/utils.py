from math import lgamma, log
from comfy_bayes.bayes_conversion import ConversionTest


def lbeta(alpha, beta):
    """lbeta distribution approx. through lgamma"""
    return lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)


def from_pandas(df, group_col="group", success_col="success"):
    """
    group_col should have values: A or B
    success_col should have true or false boolean values
    """

    aggregated = (
        df
            .loc[:, [success_col, group_col]]
            .groupby([success_col, group_col])
            .size()
    )

    ab_test = ConversionTest(successes_a=aggregated.loc[True, 'A'],
                             total_a=aggregated.loc[False, 'A'] + aggregated.loc[True, 'A'],
                             successes_b=aggregated.loc[True, 'B'],
                             total_b=aggregated.loc[False, 'B'] + aggregated.loc[True, 'B'])

    return ab_test
