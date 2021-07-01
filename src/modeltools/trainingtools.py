import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

class StatsmodelsWrapper(BaseEstimator, RegressorMixin):
    """
    A simple wrapper class to allow Statsmodels linear model to use various functions from sklearn.
    
    """
    def __init__(self, fit_intercept=True, threshold=None, is_logit=False):
        self.fit_intercept = fit_intercept
        self.threshold = threshold
        self.is_logit = is_logit

    def fit(self, X, y, column_names=() ):
        """
        Parameters
        ------------
        column_names: list
                It is an optional value, such that this class knows 
                what is the name of the feature to associate to 
                each column of X. This is useful if you use the method
                summary(), so that it can show the feature name for each
                coefficient
        """ 
        if self.fit_intercept:
            X = sm.add_constant(X)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if len(column_names) != 0:
            cols = column_names.copy()
            cols = list(cols)
            X = pd.DataFrame(X)
            cols = column_names.copy()
            cols.insert(0,'intercept')
            print('X ', X)
            X.columns = cols

        if self.is_logit:
            self.model_ = sm.Logit(y, X)
        else:
            self.model_ = sm.OLS(y, X)
        self.results_ = self.model_.fit()

        return self
      
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'model_')

        if self.fit_intercept:
            X = sm.add_constant(X)

        # Input validation
        X = check_array(X)

        if self.threshold:
            return self.results_.predict(X) > self.threshold

        return self.results_.predict(X)

    def get_params(self, deep = False):
        return {
            'fit_intercept':self.fit_intercept,
            'threshold':self.threshold,
            'fit_intercept':self.is_logit,
            }

    def summary(self):
        print(self.results_.summary())


def looEval(features: list, label: str, data: pd.DataFrame, model=LogisticRegression(C=1000000)):
    X = data[features].values
    y = data[label] 
    cv = LeaveOneOut()
    y_true, y_pred = list(), list()
    for train_ix, test_ix in cv.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # fit model
        model.fit(X_train, y_train)
        # evaluate model
        yhat = model.predict_proba(X_test)
        # store
        y_true.append(y_test.iloc[0])
        y_pred.append(yhat[0][1])
    # # calculate accuracy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred - y_true
    return y_true, y_pred

def plotROC(y_true, y_pred, features=None):
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

def plotPR(y_true, y_pred, features=None):
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
        ax.set_title(f'Precision - Recall \nAvg Precision = {avg_precision}\nPositive Class Frequency = {np.mean(y_true)}\nFeature Set: {str(features)}')
    else:
        ax.set_title(f'Precision - Recall \nAvg Precision = {avg_precision}\nPositive Class Frequency = {np.mean(y_true)}')
    return fig

def evaluateModel(features: list, label: str, data: pd.DataFrame, model=LogisticRegression(C=1000000)):
    y_true, y_pred = looEval(features, label, data, model)
    plotROC(y_true, y_pred, features)
    plotPR(y_true, y_pred, features)