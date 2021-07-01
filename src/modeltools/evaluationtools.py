import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def error_decomposition(y_true, y_pred_train, y_pred_test, scoring_function, unavoidable_error=0.0):
    train_score = scoring_function(y_true, y_pred_train)
    test_score = scoring_function(y_true, y_pred_test)
    errors = [unavoidable_error, 1-train_score-unavoidable_error, 1-train_score, (1-test_score) - (1-train_score), 1-test_score]
    errors_types = ["unavoidable error", "underfitting", "train error", "overfitting ", "test error"]
    error_df = pd.DataFrame(errors, index=errors_types, columns=['value'])
    error_df['bottom'] = [0.0, errors[0], 0.0, errors[2], 0.0]
    return error_df
  
def plot_error_decomposition(y_true, y_pred_train, y_pred_test, scoring_function=None, unavoidable_error=0.0, title="Error Decomposition"):
    if not scoring_function:
        scoring_function = lambda x, y: np.mean( (np.array(x)-np.array(y))**2 )
    error_df = error_decomposition(y_true, y_pred_train, y_pred_test, scoring_function, unavoidable_error=0.0)
    fig, ax = plt.subplots()
    x_values = np.arange(error_df.shape[0])
    ax.set_xticks(x)
    ax.set_xticklabels(error_df.index, rotation=45)
    for i in range(error_df.shape[0]):
        value = error_df.iloc[i]['value']
        bottom = error_df.iloc[i]['bottom']
        label = f"{value:.2f}"
        # rect = ax.bar(x=x_values[i], height=value, bottom=bottom, width=0.5, label=label)
        ax.bar(x=x_values[i], height=value, bottom=bottom, width=0.5, label=label)
    ax.grid()
    ax.legend()
    ax.set_title(title)
    return fig