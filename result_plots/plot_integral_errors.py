import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def plot_integral_error(predictions, observations, x_axis, is_parametric=True):
    """
    Creates an integral error plot as shown in our paper
    :param predictions: Predictions from a parametric or non parametric model
    :param observations: True observations
    :param x_axis: The x-axis label for the plot
    :param is_parametric: Boolean indicating whether the model is parametric or non parametric
    :return: A plot showing the integral error
    """
    if is_parametric:
        parametric_predictions = predictions
        emp_scale = np.sqrt((1 / len(parametric_predictions.scale)) * np.sum(np.square(parametric_predictions.scale)))
        emp_location = np.mean(parametric_predictions.location)
        rns = norm.rvs(loc=emp_location, scale=emp_scale, size=1000)
        rns = np.where(rns < 0, 0, rns)
        emp_cdf = ECDF(x=rns)
        real_cdf = ECDF(x=np.concatenate(observations.values, axis=0))
        mma = np.max(np.concatenate(observations.values, axis=0))
        mmi = np.min(np.concatenate(observations.values, axis=0))
        x_run = np.arange(mmi, mma, 0.1)
        y1 = emp_cdf(x_run)
        y2 = real_cdf(x_run)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(x_run, y1, label="Prediction")
        ax.plot(x_run, y2, label="Real")
        ax.fill_between(x_run, y1, y2, color='lightsalmon', alpha=0.4, label="Error")
        ax.set_xlabel(str(x_axis))
        ax.set_ylabel('Chance of Departure')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center', ncol=3)
        fig.suptitle("Integral Error for Parametric Model")
    elif not is_parametric:
        non_parametric_predictions = predictions
        emp_cdf = ECDF(x=np.mean(non_parametric_predictions, axis=0))
        real_cdf = ECDF(x=np.concatenate(observations.values, axis=0))
        mma = np.max(np.concatenate(observations.values, axis=0))
        mmi = np.min(np.concatenate(observations.values, axis=0))
        x_run = np.arange(mmi, mma, 0.1)
        y1 = emp_cdf(x_run)
        y2 = real_cdf(x_run)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(x_run, y1, label="Prediction")
        ax.plot(x_run, y2, label="Real")
        ax.fill_between(x_run, y1, y2, color='lightsalmon', alpha=0.4, label="Error")
        ax.set_xlabel(str(x_axis))
        ax.set_ylabel('Chance of Departure')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center', ncol=3)
        fig.suptitle("Integral Error for Non-Parametric Model")
    return fig
