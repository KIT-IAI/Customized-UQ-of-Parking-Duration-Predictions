import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.integrate import quad
import warnings


def integral_error_parametric(parametric_predictions, observations):
    """
    Calculates the mean integral error for a parametric model
    :param parametric_predictions: The predictions from a parametric model
    :param observations: The true observations
    :return: The mean integral error
    """
    emp_scale = np.sqrt((1 / len(parametric_predictions.scale)) * np.sum(np.square(parametric_predictions.scale)))
    emp_location = np.mean(parametric_predictions.location)
    rns = norm.rvs(loc=emp_location, scale=emp_scale, size=1000)
    rns = np.where(rns < 0, 0, rns)
    emp_cdf = ECDF(x=rns)
    real_cdf = ECDF(x=np.concatenate(observations.values, axis=0))
    warnings.filterwarnings("ignore")
    int_func = lambda x: np.abs(np.subtract(emp_cdf(x), real_cdf(x)))
    return quad(int_func, 0, 25)[0]


def integral_error_non_parametric(non_parametric_predictions, observations):
    """
    Calculates the mean integral error for a non-parametric model
    :param non_parametric_predictions: The predictions from a non parametric model
    :param observations: The true observations
    :return: The mean integral error
    """
    emp_cdf = ECDF(x=np.mean(non_parametric_predictions, axis=0))
    real_cdf = ECDF(x=np.concatenate(observations.values, axis=0))
    warnings.filterwarnings("ignore")
    int_func = lambda x: np.abs(np.subtract(emp_cdf(x), real_cdf(x)))
    return quad(int_func, 0, 25)[0]
