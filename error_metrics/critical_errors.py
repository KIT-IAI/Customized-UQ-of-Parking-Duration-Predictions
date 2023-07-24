import pandas as pd

import numpy as np
from scipy.stats import norm


def calculate_critical_non_critical_error_parametric(security_level, parametric_predictions, observation):
    """
    Calculates the critical and non-critical error for a given security level for parametric models
    :param security_level: The security level to be calculated
    :param parametric_predictions: The predictions from the parametric model
    :param observation: The true observations
    :return: Returns the mean of the underestimated and overestimated values
    """
    quantile = 1 - security_level
    quantile_prediction = norm.ppf(quantile, loc=parametric_predictions.location, scale=parametric_predictions.scale)
    quantile_prediction = np.where(quantile_prediction < 0, 0, quantile_prediction)
    true_values = np.concatenate(observation.values, axis=0)

    # Calculate difference
    diff = true_values - quantile_prediction

    # Calculate critical and non critical errors
    non_critical_preds = np.where(diff >= 0, diff, 0)
    critical_preds = np.where(diff < 0, -diff, 0)

    mean_non_critical_error = np.mean(non_critical_preds)
    mean_critical_error = np.mean(critical_preds)

    return mean_non_critical_error, mean_critical_error


def calculate_critical_non_critical_error_non_parametric(security_level, non_parametric_predictions, observation):
    """
    Calculates the critical and non-critical error for a given security level for non-parametric models
    :param security_level: The security level to be calculated
    :param non_parametric_predictions: The predictions from the non-parametric model
    :param observation: The true observations
    :return: Returns the mean of the underestimated and overestimated values
    """
    quantile = np.round(1 - security_level, 2)
    quantile_prediction = non_parametric_predictions[str(quantile)]
    true_values = np.concatenate(observation.values, axis=0)

    # Calculate difference
    diff = true_values - quantile_prediction

    # Calculate critical and non critical errors
    non_critical_preds = np.where(diff >= 0, diff, 0)
    critical_preds = np.where(diff < 0, -diff, 0)

    mean_non_critical_error = np.mean(non_critical_preds)
    mean_critical_error = np.mean(critical_preds)

    return mean_non_critical_error, mean_critical_error


# %%
def calculate_security_levels(predictions, observations, security_level_list, is_parametric=True):
    """
    Calculates the critical and non critical error for certain security levels
    :param predictions: The predictions, either parametric or non_parametric
    :param observations: The observations
    :param security_level_list: A list of security levels to consider
    :param is_parametric: A boolean indicating whether the prediction is parametric or not
    :return: A data frame containing the critical and non critical errors as reported in our paper
    """
    final_df = pd.DataFrame(columns=["Critical Error", "Non-Critical Error"], index=security_level_list)
    if is_parametric:
        for sl in security_level_list:
            final_df.loc[sl]["Non-Critical Error"], final_df.loc[sl][
                "Critical Error"] = calculate_critical_non_critical_error_parametric(security_level=sl,
                                                                                     parametric_predictions=predictions,
                                                                                     observation=observations)
    elif not is_parametric:
        for sl in security_level_list:
            final_df.loc[sl]["Non-Critical Error"], final_df.loc[sl][
                "Critical Error"] = calculate_critical_non_critical_error_non_parametric(security_level=sl,
                                                                                         non_parametric_predictions=predictions,
                                                                                         observation=observations)
    return final_df
