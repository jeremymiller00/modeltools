import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np

from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator


class StatsmodelsWrapper(BaseEstimator, RegressorMixin):
    """
    A simple wrapper class to allow Statsmodels linear model to use various functions from sklearn.
    
    """
    def __init__(self, fit_intercept=True, threshold=None, is_logit=False):

        self.fit_intercept = fit_intercept
        self.threshold = threshold
        self.is_logit = is_logit


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
    def fit(self, X, y, column_names=() ):

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
        print(self.results_.summary() )