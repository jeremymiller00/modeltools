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

def values_from_dataframe(features: list, label: str, data: pd.DataFrame):
    X = data[features].values
    y = data[label]
    return X, y

def loo_cv(X, y, model=LogisticRegression(C=1000000), is_classifier=True):
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
            yhat = model.predict_proba(X_test)
            y_pred.append(yhat[0][1])
        else:
            yhat = model.predict(X_test)
            y_pred.append(yhat)
        y_true.append(y_test)
        
    # # calculate accuracy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return y_true, y_pred

