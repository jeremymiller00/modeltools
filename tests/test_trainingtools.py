import unittest
import numpy as np
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from modeltools import trainingtools as tt


class TestTrainingTools(unittest.TestCase):

    def setUp(self):
        self.linreg = tt.StatsmodelsWrapper()
        self.logreg = tt.StatsmodelsWrapper(is_logit=True)
        self.logreg_thresh = tt.StatsmodelsWrapper(threshold=0.5, is_logit=True)
        self.X_dummy = np.random.rand(100)
        self.X_dummy_2d = np.random.rand(100, 20)
        self.X_dummy_predict = np.random.rand(100)
        self.y_dummy_continuous = np.random.rand(100)
        self.y_dummy_categorical = np.random.randint(low=0, high=2, size=100)

    def test_init(self):
        self.assertEqual(self.linreg.is_logit, False)
        self.assertEqual(self.logreg.is_logit, True)
        self.assertEqual(self.logreg_thresh.threshold, 0.5)
        self.assertEqual(self.logreg_thresh.is_logit, True)

    def test_fit(self):
        self.linreg.fit(self.X_dummy, self.y_dummy_continuous)
        self.assertIsInstance(self.linreg._fit_model,
                              statsmodels.regression.linear_model.RegressionResultsWrapper)
        self.logreg.fit(self.X_dummy, self.y_dummy_categorical)
        self.assertIsInstance(self.logreg._fit_model,
                              statsmodels.discrete.discrete_model.BinaryResultsWrapper)

    def test_predict(self):
        self.linreg.fit(self.X_dummy, self.y_dummy_continuous)
        linreg_predictions = self.linreg.predict(self.X_dummy_predict)
        self.assertEqual(len(linreg_predictions), 100)

        self.logreg.fit(self.X_dummy, self.y_dummy_categorical)
        logreg_predictions = self.logreg.predict(self.X_dummy_predict)
        self.assertEqual(len(logreg_predictions), 100)

    def test_threshold(self):
        self.logreg_thresh.fit(self.X_dummy, self.y_dummy_categorical)
        logreg_predictions = self.logreg_thresh.predict(self.X_dummy_predict)
        for item in logreg_predictions:
            self.assertIn(item, [0, 1], "Threshold Error")

    def test_loo_cv(self):
        y_true, y_pred = tt.loo_cv(self.X_dummy_2d, self.y_dummy_categorical)
        self.assertEqual(len(y_true), len(y_pred))
        model = LogisticRegression(C=10000000)
        # predictions from model without CV
        model.fit(self.X_dummy_2d, self.y_dummy_categorical)
        preds = model.predict_proba(self.X_dummy_2d)
        self.assertNotEqual(y_pred, preds)


if __name__ == '__main__':
    unittest.main()
