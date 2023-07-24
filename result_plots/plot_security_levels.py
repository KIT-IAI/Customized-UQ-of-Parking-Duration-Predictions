from error_metrics.critical_errors import calculate_security_levels
import numpy as np
import matplotlib.pyplot as plt


def plot_security_levels(predictions, observations, security_level_list, is_parametric):
    """
    Creates a plot of the critical and non critical error across security levels
    :param predictions: The predictions, either parametric or non_parametric
    :param observations: The observations
    :param security_level_list: A list of security levels to consider
    :param is_parametric: A boolean indicating whether the prediction is parametric or not
    :return: A plot of the security levels as in our paper
    """
    df_critical_values = calculate_security_levels(predictions=predictions,
                                                   observations=observations,
                                                   security_level_list=security_level_list,
                                                   is_parametric=is_parametric)

    plot_dict = dict()
    plot_dict["Critical Trips (Leave Early)"] = list(df_critical_values["Critical Error"].values)
    plot_dict["Non Critical Trips (Leave Late)"] = list(df_critical_values["Non-Critical Error"].values)
    labels = [np.round(i * 100, 0) for i in security_level_list]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.stackplot(labels, plot_dict.values(), labels=plot_dict.keys())
    ax.set_ylabel('Error in Hours')
    ax.set_xlabel('Security Level')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', ncol=2)
    if is_parametric:
        fig.suptitle("Security Levels and Errors for the Parametric Model")
    elif not is_parametric:
        fig.suptitle("Security Levels and Errors for the Non-Parametric Model")

    return (fig)
