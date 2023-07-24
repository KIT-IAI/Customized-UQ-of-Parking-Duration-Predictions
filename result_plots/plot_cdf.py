import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def plot_cdf_parametric(parametric_predictions, observations, x_axis):
    """
    Creates a plot of the forecast and real CDF for a parametric forecasting method
    :param parametric_predictions: The results from the parametric forecasting method, using the scale and location
    :param observations: The true observations
    :param x_axis: The label used for the x_axis
    :return: A figure comparing the forecast and real CDF
    """
    ## Calculate predicted CDF
    empirical_scale = np.sqrt((1 / len(parametric_predictions.scale)) * np.sum(np.square(parametric_predictions.scale)))
    empirical_location = np.mean(parametric_predictions.location)
    sample = norm.rvs(loc=empirical_location, scale=empirical_scale, size=1000)
    corrected_sample = np.where(sample < 0, 0, sample)
    empirical_cdf = ECDF(x=corrected_sample)

    ## Calculate Real CDF
    real_cdf = ECDF(x=np.concatenate(observations.values, axis=0))

    ## Plot both
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(empirical_cdf.x, empirical_cdf.y, label='Predicted CDF')
    ax.plot(real_cdf.x, real_cdf.y, label='Real CDF')
    ax.set_xlabel(str(x_axis))
    ax.set_ylabel('Chance of Departure')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=2)
    fig.suptitle("Parametric Forecast Method")
    return (fig)


def plot_cdf_non_parametric(non_parametric_predictions, observations, x_axis):
    """
    Creates a plot of the forecast and real CDF for a non-parametric forecasting method
    :param non_parametric_predictions: The results from the non-parametric forecasting method as a set of quantiles
    :param observations: The true observations
    :param x_axis: The label used for the x_axis
    :return: A figure comparing the forecast and real CDF
    """
    ## Calculate mean for generating CDF
    mean_prediction = np.mean(non_parametric_predictions, axis=0)

    ## Calculate Real CDF
    real_cdf = ECDF(x=np.concatenate(observations.values, axis=0))

    ## Create plots
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(np.sort(mean_prediction), np.cumsum(mean_prediction) / np.sum(mean_prediction), label='Predicted CDF')
    ax.plot(real_cdf.x, real_cdf.y, label='Real CDF')
    ax.set_xlabel(str(x_axis))
    ax.set_ylabel('Chance of Departure')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=2)
    fig.suptitle("Non-Parametric Forecast Method")
    return (fig)
