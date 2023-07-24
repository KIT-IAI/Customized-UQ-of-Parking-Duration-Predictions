import pandas as pd
import numpy as np
from scipy.stats import norm


def error_outside_width_parametric(parametric_predictions, observations, prediction_interval):
    """
    Calculate the mean error outside the prediction interval and the mean width of this interval for a parametric
    forecasting model
    :param parametric_predictions: Predictions from the parametric model
    :param observations: The true observations
    :param prediction_interval: The considered prediction interval
    :return: The mean error outside the prediction interval and the mean width of this prediction interval
    """
    pi = np.round(prediction_interval / 100, 2)
    quantile_lower = np.round((1 - pi) / 2, 2)
    quantile_upper = np.round(1 - quantile_lower, 2)

    pred_upper = norm.ppf(quantile_upper, loc=parametric_predictions.location, scale=parametric_predictions.scale)
    pred_lower = norm.ppf(quantile_lower, loc=parametric_predictions.location, scale=parametric_predictions.scale)

    true_observations = np.concatenate(observations.values, axis=0)
    mean_width = np.mean(np.subtract(pred_upper, pred_lower))
    mean_error = np.mean(np.select([(pred_lower > true_observations), (true_observations > pred_upper)],
                                   [np.subtract(pred_lower, true_observations),
                                    np.subtract(true_observations, pred_upper)], default=0))
    return mean_error, mean_width


def error_outside_width_non_parametric(non_parametric_predictions, observations, prediction_interval):
    """
    Calculate the mean error outside the prediction interval and the mean width of this interval for a non parametric
    forecasting model
    :param non_parametric_predictions: Predictions from the non parametric model
    :param observations: The true observations
    :param prediction_interval: The considered prediction interval
    :return: The mean error outside the prediction interval and the mean width of this prediction interval
    """
    pi = np.round(prediction_interval / 100, 2)
    quantile_lower = np.round((1 - pi) / 2, 2)
    quantile_upper = np.round(1 - quantile_lower, 2)

    pred_upper = non_parametric_predictions[str(quantile_upper)]
    pred_lower = non_parametric_predictions[str(quantile_lower)]

    true_observations = np.concatenate(observations.values, axis=0)
    mean_width = np.mean(np.subtract(pred_upper, pred_lower))
    mean_error = np.mean(np.select([(pred_lower > true_observations), (true_observations > pred_upper)],
                                   [np.subtract(pred_lower, true_observations),
                                    np.subtract(true_observations, pred_upper)], default=0))
    return mean_error, mean_width


def calculate_error_outside_and_width(predictions, observations, considered_prediction_intervals, is_parametric=True):
    """
    Calculates the mean error outside the prediction interval and mean width of this predictio interval for multiple
    considered prediction intervals
    :param predictions: The predictions from the parametric or non parametric model
    :param observations: The true observations
    :param considered_prediction_intervals: A list of prediction intervals to be considered
    :param is_parametric: A boolean indicating whether a parametric or non parametric model is being used
    :return: A data frame of mean errors outside the prediction interval and the mean width of this prediction interval
    """
    final_df = pd.DataFrame(columns=["Mean Error Outside Prediction Interval", "Mean Prediction Interval Width"],
                            index=considered_prediction_intervals)
    for pi in considered_prediction_intervals:
        if is_parametric:
            final_df.loc[pi]["Mean Error Outside Prediction Interval"], final_df.loc[pi][
                "Mean Prediction Interval Width"] = error_outside_width_parametric(parametric_predictions=predictions,
                                                                                   observations=observations,
                                                                                   prediction_interval=pi)
        elif not is_parametric:
            final_df.loc[pi]["Mean Error Outside Prediction Interval"], final_df.loc[pi][
                "Mean Prediction Interval Width"] = error_outside_width_non_parametric(
                non_parametric_predictions=predictions,
                observations=observations,
                prediction_interval=pi)
    return final_df
