import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression

from .trainingtools import loo_cv, values_from_dataframe


def error_decomposition(y_true, y_pred_train, y_pred_test, scoring_function, unavoidable_error=0.0):
    train_score = scoring_function(y_true, y_pred_train)
    test_score = scoring_function(y_true, y_pred_test)
    errors = [unavoidable_error, 1 - train_score - unavoidable_error, 1 - train_score,
              (1 - test_score) - (1 - train_score), 1 - test_score]
    errors_types = ["unavoidable error", "underfitting", "train error", "overfitting ", "test error"]
    error_df = pd.DataFrame(errors, index=errors_types, columns=['value'])
    error_df['bottom'] = [0.0, errors[0], 0.0, errors[2], 0.0]
    return error_df


def plot_error_decomposition(y_true, y_pred_train, y_pred_test, scoring_function=None, unavoidable_error=0.0,
                             title="Error Decomposition"):
    if not scoring_function:
        scoring_function = lambda x, y: np.mean((np.array(x) - np.array(y)) ** 2)
    error_df = error_decomposition(y_true, y_pred_train, y_pred_test, scoring_function, unavoidable_error=0.0)
    fig, ax = plt.subplots()
    x_values = np.arange(error_df.shape[0])
    ax.set_xticks(x_values)
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


def plot_roc_curve(y_true, y_pred, features=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred.ravel())
    roc_auc = roc_auc_score(y_true, y_pred)
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='AUC = %0.9f' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    if features:
        ax.set_title(f'ROC Curve\nFeature Set: {str(features)}')
    else:
        ax.set_title('ROC Curve')
    return fig


def plot_prt_curve(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(recall[::-1], precision[::-1], color='navy', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.axhline(y=avg_precision)
    ax.set_title('Precision - Recall Curve\nAvg Precision: {}'.format(avg_precision))
    ax.grid()
    return fig


def plot_pr_curve(y_true, y_pred, features=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(thresholds, precision[1:], color='darkorange', lw=lw, label='Precision')
    ax.plot(thresholds, recall[1:], color='navy', lw=lw, label='Recall')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision - Recall')
    if features:
        ax.set_title(
            f'Precision - Recall \nAvg Precision = {avg_precision}\nPositive Class Frequency = {np.mean(y_true)}\nFeature Set: {str(features)}')
    else:
        ax.set_title(
            f'Precision - Recall \nAvg Precision = {avg_precision}\nPositive Class Frequency = {np.mean(y_true)}')
    return fig


def evaluate_classifier(features: list, label: str, data: pd.DataFrame, model=LogisticRegression(C=1000000)):
    X, y = values_from_dataframe(features, label, data)
    y_true, y_pred = loo_cv(X, y, model=model)
    plot_roc_curve(y_true, y_pred, features)
    plot_pr_curve(y_true, y_pred, features)
