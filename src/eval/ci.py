"""
@Created Date: Wednesday April 12th 2023
@Author: Alafate Abulimiti at alafate.abulimiti@gmail.com
@Company: INRIA
@Lab: CoML/Articulab
@School: PSL/ENS
@Description: Confidence interval calculation.
"""

import numpy as np
from scipy.stats import norm
import math

def normal_approximation_binomial_confidence_interval(s, n, confidence_level=0.95):
    """Computes the binomial confidence interval of the probability of a success s,
    based on the sample of n observations. The normal approximation is used,
    appropriate when n is equal to or greater than 30 observations.
    The confidence level is between 0 and 1, with default 0.95.
    Returns [p_estimate, interval_range, lower_bound, upper_bound].
    For reference, see Section 5.2 of Tom Mitchel's "Machine Learning" book."""

    p_estimate = (1.0 * s) / n

    interval_range = norm.interval(confidence_level)[1] * np.sqrt(
        (p_estimate * (1 - p_estimate)) / n
    )

    return (
        p_estimate,
        interval_range,
        p_estimate - interval_range,
        p_estimate + interval_range,
    )

def normal_ci_scoring(score, n, confidence_level=0.95):
    z_score_dict = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    z_score = z_score_dict[confidence_level]
    std_error = math.sqrt((score * (1 - score)) / n)
    return z_score * std_error


def f1_score_confidence_interval(r, p, dr, dp):
    """Computes the confidence interval for the F1-score measure of classification performance
    based on the values of recall (r), precision (p), and their respective confidence
    interval ranges, or absolute uncertainty, about the recall (dr) and the precision (dp).
    Disclaimer: I derived the formula myself based on f(r,p) = 2rp / (r+p).
    Nobody has revised my computation. Feedback appreciated!"""

    f1_score = (2.0 * r * p) / (r + p)

    left_side = np.abs((2.0 * r * p) / (r + p))

    right_side = np.sqrt(
        np.power(dr / r, 2.0)
        + np.power(dp / p, 2.0)
        + ((np.power(dr, 2.0) + np.power(dp, 2.0)) / np.power(r + p, 2.0))
    )

    interval_range = left_side * right_side

    return (
        f1_score,
        interval_range,
        f1_score - interval_range,
        f1_score + interval_range,
    )
