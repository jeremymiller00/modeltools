import unittest
import numpy as np
import statsmodels
import statsmodels.api as sm
import src.modeltools.trainingtools as tt

class TestTrainingTools(unittest.TestCase):

    def setUp(self):
        self.linreg = tt.StatsmodelsWrapper()
        self.logreg = tt.StatsmodelsWrapper(is_logit=True)
        self.logreg_thresh = tt.StatsmodelsWrapper(threshold=0.5, is_logit=True)
        self.X_dummy = np.random.rand(100)
        self.X_dummy_predict = np.random.rand(20)
        self.y_dummy_continuous = np.random.rand(100)
        self.y_dummy_categorical = np.random.randint(low=0, high=2, size=100)

    def test_setup(self):
        """to be deleted
        """
        # print(self.X_dummy)
        # print(self.y_dummy_continuous)
        # print(self.y_dummy_categorical)

    def test_init(self):
        self.assertEqual(self.linreg.is_logit, False)
        self.assertEqual(self.logreg.is_logit, True)
        self.assertEqual(self.logreg_thresh.threshold, 0.5)
        self.assertEqual(self.logreg_thresh.is_logit, True)

    def test_fit(self):
        self.linreg.fit(self.X_dummy, self.y_dummy_continuous)
        self.assertIsInstance(self.linreg.results_, 
            statsmodels.regression.linear_model.RegressionResultsWrapper)
        self.logreg.fit(self.X_dummy, self.y_dummy_categorical)
        self.assertIsInstance(self.logreg.results_, 
            statsmodels.discrete.discrete_model.BinaryResultsWrapper)

    def test_predict(self):
        self.linreg.fit(self.X_dummy, self.y_dummy_continuous)
        linreg_predictions = self.linreg.predict(self.X_dummy_predict)
        self.assertEqual(len(linreg_predictions), 20)

        self.logreg.fit(self.X_dummy, self.y_dummy_categorical)
        logreg_predictions = self.logreg.predict(self.X_dummy_predict)
        self.assertEqual(len(logreg_predictions), 20)
        # check that s.split fails when the separator is not a string
        # with self.assertRaises(TypeError):
        #     s.split(2)

    def test_threshold(self):
        self.logreg_thresh.fit(self.X_dummy, self.y_dummy_categorical)
        logreg_predictions = self.logreg_thresh.predict(self.X_dummy_predict)
        pred_set = set(logreg_predictions)
        self.assertEqual(sorted(list(pred_set)), [0,1])


if __name__ == '__main__':
    unittest.main()