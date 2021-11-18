# ModelTools
A collection of utility model training and statistical tools. These tools intend to solve problems such as:
* Integrating components of different ML libraries so they can be used together
* Addressing specific shortcomings of specific ML library functions
* Simplify common tasks, such as plots for classifier evaluation

# Bayesian AB Testing

A convenience wrapper for common Bayesian A/B Testing tasks.  
Bayesian AB Testing is base on the use of conjugate prior distributions. A description of common conjugate prior relationships can be found [here](https://en.wikipedia.org/wiki/Conjugate_prior).

The most common use case is when the metric of interest is a measure of K successes in N trials, such as a click-through-rate. Another common use case is a Poisson proccess (k events per interval) such as signups or api traffic per day.

Sample usage:

```python
from modeltools.testingtools import BayesianABTesting  
data = {  
  "a_trials": 1000,
  "a_successes": 100,
  "b_trials": 2000,
  "b_successes": 201,
}
ab_test_ctr = BayesianABTesting(likelihood_function="binomial", data=data)
result = ab_test_ctr.execute_test(metric="CTR")
```

You can call some static methods to get info:
```python
BayesianABTesting.get_likelihood_options()
BayesianABTesting.get_required_data_fields()
```