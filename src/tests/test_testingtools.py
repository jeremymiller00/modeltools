from unittest import TestCase
from modeltools.testingtools import BayesianABTesting
import matplotlib

class TestBayesianABTesting(TestCase):
    def setUp(self) -> None:
        self.data = {
            "a_trials": 100,
            "a_successes": 10,
            "b_trials": 1000,
            "b_successes": 120
        }
        self.likelihood_function = "binomial"
        self.tester = BayesianABTesting(self.likelihood_function, self.data)

    def test__test_binom(self):
        winner, diff, prob = self.tester._test_binom(metric="metric", verbose=1, labels=["A", "B"], plot=False)
        self.assertIsInstance(winner, str, "Invalid type, should be str")
        self.assertIsInstance(diff, float, "Invalid numeric type")
        self.assertIsInstance(prob, float, "Invalid numeric type")
        self.assertIn(winner, ["A", "B"], "Invalid value for winner")
        self.assertAlmostEqual(diff, 0.0129, delta=0.0001)
        self.assertAlmostEqual(prob, 0.53, delta=0.01)

    def test__plot_posteriors(self):
        fig = self.tester._plot_posteriors(
            self.data.get("a_successes"),
            self.data.get("a_trials") - self.data.get("a_successes"),
            self.data.get("b_successes"),
            self.data.get("b_trials") - self.data.get("b_successes"),
            ["A", "B"]
        )
        self.assertIsInstance(fig, matplotlib.figure.Figure,
                              "Invalid object returned, matplotlib.figure.Figure required")