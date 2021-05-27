import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_binom_metric(trials: int, successes: int, metric_name: str, cred_int: int = 95) -> plt.figure:
    """    
    Wrapper function for calculating and plotting relevant metrics.
    Uses Bayesian updating to produce a Beta distribution of the relevant metric.
    Appropriate for any metric that can be modeled by a binomial distribution:
        k successes in N trials

    Params:
    trials: number of relevant labels
    successes: number of 'successful' labels, definition varies by metric, see scoping doc for definitions
    metric_name: Accuracy, Recall, Precision, Readability
    cred_int: desired size of credible interval
    
    Returns:
    Labeled fig with relevant metrics
    
    """
    failures = trials - successes
    posterior = stats.beta(1 + successes, 1 + failures)
    sample = posterior.rvs(size=10000)    
    bootstrap_ci = np.percentile(sample, [100-cred_int, cred_int])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.linspace(0, 1, 10000)
    y = posterior.pdf(x) / trials
    idx = y.argmax()
    expected_value = x[idx]
    
    ax.plot(x, y)
    ax.vlines(x=expected_value, ymin=0, ymax=posterior.pdf(expected_value) / trials, color='orange')
    ax.set_xlabel(metric_name)
    ax.set_ylabel("PDF")
    ax.grid()
    ax.set_xlim([0.5, 1.0])
    title_1 = f"Probability Distribution of {metric_name} for RF Labels"
    title_2 = f"Most Likely {metric_name} Value: {perc(expected_value)}%"
    title_3 = f"N = {trials}"
    title_4 = f"{cred_int}% {metric_name} Credible Interval: {perc(bootstrap_ci[0])}% -> {perc(bootstrap_ci[1])}%"
    ax.set_title(f"{title_1}\n{title_2}\n{title_3}\n{title_4}", fontsize=20)
    
    return fig

def perc(x: float) -> int:
    """
    Format a 0-1 decimal value to display as a percentage
    """
    return int(np.round(x * 100))