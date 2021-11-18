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
        """
        Constructor for Statsmodels Wrapper

        @param fit_intercept: To fit the intercept in the linear model or not
        @type fit_intercept: bool
        @param threshold: threshold for prediciting the positive class in classification
        @type threshold: float
        @param is_logit: True for classification, False for regression
        @type is_logit: bool
        """
        self.fit_intercept = fit_intercept
        self.threshold = threshold
        self.is_logit = is_logit
        self._model = None
        self._fit_model = None

    def fit(self, X, y, column_names=[]):
        """
        Wrapper for the fit function
        @param X: Model features, aka exogenous variables
        @type X: array-like
        @param y: Model labels, aka endogenous variable
        @type y: array-like
        @param column_names: option value to associate feature names with columns X for summary
        @type column_names: list[str]
        @return: Self
        @rtype: Self
        """

        if self.fit_intercept:
            X = sm.add_constant(X)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        if len(column_names) != 0:
            X = pd.DataFrame(X)
            cols = column_names.copy()
            cols.insert(0, 'intercept')
            X.columns = cols

        if self.is_logit:
            self._model = sm.Logit(y, X)
        else:
            self._model = sm.OLS(y, X)

        self._fit_model = self._model.fit()

    def predict(self, X):
        """
        Wrapper for predict function
        @param X: Features
        @type X: array-like
        @return: Model predictions
        @rtype: array-like
        """

        # Check is fit had been called
        check_is_fitted(self, '_model')

        if self.fit_intercept:
            X = sm.add_constant(X)

        # Input validation
        X = check_array(X)

        if self.threshold:
            return self._fit_model.predict(X) > self.threshold

        return self._fit_model.predict(X)

    def get_wrapper_params(self):
        """
        Return wrapper parameters
        @return: wrapper parameters
        @rtype: dict
        """
        return {
            'fit_intercept': self.fit_intercept,
            'threshold': self.threshold,
            'is_logit': self.is_logit,
        }

    def get_fit_model(self):
        """
        Return fit model to remove wrapper class
        @return: fit model
        @rtype: Statsmodels fit model
        """
        return self._fit_model

    def summary(self):
        """
        Wrap Statsmodels summary function
        @return: fit model summary
        @rtype: str
        """
        print(self._fit_model.summary())


def loo_cv(X, y, model=LogisticRegression(C=1000000), is_classifier=True):
    """
    Convenience wrapper for Leave One Out Classification with Sklearn

    @param X: Features
    @type X: array-like
    @param y: Label
    @type y: array-like
    @param model: sklearn model class (not yet instantiated)
    @type model: sklearn model (varies)
    @param is_classifier: needed for calling correct predicition method
    @type is_classifier: bool
    @return: two arrays; true label values, predicted label values
    @rtype: numpy arrays
    """
    cv = LeaveOneOut()
    y_true, y_pred = list(), list()
    for train_ix, test_ix in cv.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # fit model
        model.fit(X_train, y_train)
        # evaluate model
        if is_classifier:
            y_hat = model.predict_proba(X_test)
            y_pred.append(y_hat[0][1])
        else:
            y_hat = model.predict(X_test)
            y_pred.append(y_hat)
        y_true.append(y_test[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return y_true, y_pred
