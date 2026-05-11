"""
This module contains classes for calculating online statistics and a mixture of variances.

Classes:
    MixtureOFVariances: Maintains a list of OnlineStats objects and computes a quantile of their variances.
    OnlineStats: Computes online statistics such as mean, variance, and standard deviation with optional windowing.

Usage:
    mixture = MixtureOFVariances(size=10, quantile=0.95, window_size=100, inclusion_probability=0.9)
    mixture.update(new_value)
    result = mixture()

    stats = OnlineStats(window_size=100, inclusion_probability=0.9)
    stats.update(new_value)
    variance = stats.var()
    mean = stats.mean()
    std_dev = stats.std()
    std_error = stats.std_error()
"""

import numpy as np
from collections import deque
import scipy.stats as st


class MixtureOfVariances:

    def __init__(
        self,
        mixture_size=10,
        quantile=0.95,
        window_size=None,
        inclusion_probability=1.0,
    ):
        self.variances = [
            OnlineStats(
                window_size=window_size, inclusion_probability=inclusion_probability
            )
            for _ in range(mixture_size)
        ]
        self.quantile = quantile

    def update(self, x):
        for var in self.variances:
            var.update(x)

    def reset(self):
        for var in self.variances:
            var.reset()

    def mean(self):
        return np.mean([var.mean() for var in self.variances])

    def var(self, *args, **kwds):
        variances = [var.var() for var in self.variances]
        return np.quantile(variances, self.quantile)

    def std(self):
        return np.sqrt(self.var())

    def __len__(self):
        return len(self.variances)


class OnlineStats:
    def __init__(
        self, window_size=None, inclusion_probability=1.0, model_x=0.0, seed=None
    ):
        self.prob = inclusion_probability
        self.window_size = window_size
        if window_size is not None:
            self.data = deque(maxlen=window_size)
        self.model_x = model_x
        self.seed = seed
        self.reset()

    def reset(self):
        _type = type(self.model_x)
        if _type is float:
            self.n = 0
            self.mu = 0.0
            self.ssv = 0.0
        elif _type is np.ndarray:
            self.n = 0
            self.mu = np.zeros(self.model_x.size, dtype=np.float64)
            self.ssv = np.zeros(self.model_x.size, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported type: {_type}")

        self.rng = np.random.default_rng(self.seed)

        if self.window_size is not None:
            self.data.clear()

    def update(self, x):
        if not isinstance(x, type(self.model_x)):
            raise ValueError(f"Type mismatch: {type(x)} != {type(self.model_x)}")

        if self.rng.random() > self.prob:
            return

        if self.window_size is not None:
            if len(self.data) >= self.window_size:  # Window is full
                old_x = self.data.popleft()
                # At this point, self.n, self.mu, self.ssv are for self.window_size elements.
                # We need to remove old_x's contribution.

                if (
                    self.n == 1
                ):  # Special case: removing the only element (window_size was 1)
                    # Reset stats as the window becomes momentarily empty
                    self.mu = (
                        0.0
                        if isinstance(self.model_x, float)
                        else np.zeros_like(self.model_x)
                    )
                    self.ssv = (
                        0.0
                        if isinstance(self.model_x, float)
                        else np.zeros_like(self.model_x)
                    )
                    self.n = 0  # Count becomes 0 before adding the new element
                elif self.n > 1:
                    # Store the mean and count *before* removing old_x
                    mu_old_val = self.mu
                    n_old_val = self.n

                    # Update n first to reflect the removal
                    self.n -= 1  # Now self.n is (window_size - 1)

                    # Update mu using the stored old values and the new count
                    # mu_new = (n_old * mu_old - old_x) / n_new
                    self.mu = (n_old_val * mu_old_val - old_x) / self.n

                    # Update ssv using the stored old mean and the new mean
                    # ssv_new = ssv_old - (old_x - mu_old) * (old_x - mu_new)
                    self.ssv -= (old_x - mu_old_val) * (old_x - self.mu)

        # Add new element x (Welford's algorithm for adding a point)
        # Current self.n, self.mu, self.ssv are for elements after potential removal.
        mu_before_add = self.mu
        self.n += 1
        error = x - mu_before_add
        self.mu = mu_before_add + (error) / float(self.n)
        self.ssv += error * (x - self.mu)

        if self.window_size is not None:
            self.data.append(x)

    def var(self):
        if self.n > 1:
            return self.ssv / (self.n - 1.0)
        if type(self.model_x) is float:
            return 0.0
        return np.zeros(self.model_x.size, dtype=np.float64)

    def std(self):
        return np.sqrt(self.var())

    def mean(self):
        return self.mu

    def std_error(self):
        if self.n > 1:
            return self.std() / np.sqrt(self.n)
        if type(self.model_x) is float:
            return 0.0
        return np.zeros(self.model_x.size, dtype=np.float64)


class MonteCarloIntegration:
    def __init__(
        self,
        delta_abs=1.0,
        p=0.95,
        min_samples=None,
        max_samples=None,
        method="gaussian",
        inflation=1.0,
        model_x=0.0,
        **kwargs,
    ):
        self.max_samples = max_samples
        self.sample = OnlineStats(
            window_size=max_samples, model_x=model_x, seed=kwargs.get("seed")
        )
        self.delta_abs = delta_abs
        self.p = p
        self.inflation = inflation
        self.mixture_of_variances = None
        if min_samples is None:
            self.min_samples = 2
        else:
            self.min_samples = min_samples

        if method == "gaussian":
            self.__stopping_rule = self.__gaussian_stopping_rule
        elif method == "students":
            self.__stopping_rule = self.__students_stopping_rule
        elif method == "confidence_interval":
            self.__stopping_rule = self.__confidence_interval_stopping_rule
        elif method == "mixture":
            self.__stopping_rule = self.__mixture_stopping_rule
            self.mixture_of_variances = MixtureOfVariances(**kwargs)
        elif method == "chebyshev":
            self.__stopping_rule = self.__chebyshev_stopping_rule
        else:
            raise ValueError(f"Invalid stopping rule method: {method}")

        self.reset()

    def reset(self):
        self.sample.reset()
        if self.mixture_of_variances is not None:
            self.mixture_of_variances.reset()

    def update(self, x):
        if isinstance(x, list):
            x = np.array(x)
        self.sample.update(x)
        if self.mixture_of_variances is not None:
            self.mixture_of_variances.update(x)

    def n(self):
        return self.sample.n

    def mean(self):
        return self.sample.mean()

    def var(self):
        return self.sample.var()

    def std(self):
        return self.sample.std()

    def stop(self):
        if self.sample.n < self.min_samples:
            return False
        return self.__stopping_rule()

    def __gaussian_stopping_rule(self):
        """
        Implements a stopping rule based on the Gaussian distribution.

        Returns:
            True if the stopping criterion is met, False otherwise.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (
                -np.sqrt(self.sample.n)
                * self.delta_abs
                / (self.sample.std() * self.inflation)
            )
        p_value = (1 - self.p) - 2 * st.norm.cdf(z)
        return np.all(p_value > 0)

    def __students_stopping_rule(self):
        """
        Implements a stopping rule based on the Student's t-distribution.  This method is
        more conservative than the Gaussian method due to the heavier tails of the t-distribution.

        In practice, it seems this method has a similar stopping behavior to the Gaussian method.

        Returns:
            True if the stopping criterion is met, False otherwise.
        """
        z = (
            -np.sqrt(self.sample.n)
            * self.delta_abs
            / (self.sample.std() * self.inflation)
        )
        p_value = (1 - self.p) - 2 * st.t.cdf(z, df=self.sample.n - 1)
        return np.all(p_value > 0)

    def __confidence_interval_stopping_rule(self, inflation=1.0):
        """
        Implements a stopping rule based on confidence intervals.

        Returns:
            True if the stopping criterion is met, False otherwise.
        """
        sample_standard_error = (self.sample.std() * self.inflation) / np.sqrt(
            self.sample.n
        )
        margin = (
            st.t.ppf((1 + self.p) / 2.0, df=self.sample.n - 1) * sample_standard_error
        )
        return np.all(margin <= self.delta_abs)

    def __mixture_stopping_rule(self):
        """
        Implements a robust stopping rule that maintains mixture of variances to produce
        an inflated estimate of the variance (vs. the other methods).  The stopping criterion
        is based on the confidence interval.

        Returns:
            True if the stopping criterion is met, False otherwise.
        """
        sample_standard_error = (self.mixture_of_variances.std()) / np.sqrt(
            self.sample.n
        )
        margin = (
            st.t.ppf((1 + self.p) / 2.0, df=self.sample.n - 1) * sample_standard_error
        )
        return np.all(margin <= self.delta_abs)

    def __chebyshev_stopping_rule(self):
        """
        Implements a stopping rule based on the Chebyshev inequality.  This method is the
        most conservative and is not recommended for practical use.

        Ignore the inflation parameter as Chebyshev is already conservative.

        Returns:
            True if the stopping criterion is met, False otherwise.
        """
        m = self.sample.std() ** 2 / ((1 - self.p) * self.delta_abs**2)
        return np.all(self.sample.n >= m)


# Based on the work of Kim and Nelson:
#   Kim, Seong-Hee, and Barry L. Nelson.
#   "A fully sequential procedure for indifference-zone selection in simulation."
#   ACM Transactions on Modeling and Computer Simulation (TOMACS) 11.3 (2001): 251-273.
class IndifferenceZoneSelector:
    def __init__(self, delta, alpha, n0=2, c=1, model_x=0.0):
        """
        delta: indifference zone (minimum practically significant difference)
        alpha: desired error level (e.g. 0.05)
        n0: initial number of samples per alternative
        c: constant used to compute eta (typically 1)
        """
        self.delta = delta
        self.alpha = alpha
        self.n0 = n0
        self.c = c
        self.alternatives = {}
        self.active = set()
        self.h2 = None
        self.eta = None
        self.r = n0
        self.I = None

    def add_alternative(self, name):
        self.alternatives[name] = {"samples": [], "stats": OnlineStats(), "max_N": None}
        self.active.add(name)

    def update(self, name, value):
        alt = self.alternatives[name]
        alt["stats"].update(value)
        alt["samples"].append(value)

    def compute_eta(self, k):
        return 0.5 * (((2 * self.alpha / (k - 1)) ** (-2 / (self.n0 - 1))) - 1)

    def initialize(self):
        k = len(self.active)
        self.eta = self.compute_eta(k)
        self.h2 = 2 * self.c * self.eta * (self.n0 - 1)
        for i in self.active:
            max_Ni = 0
            for j in self.active:
                if i == j:
                    continue
                diff = np.array(self.alternatives[i]["samples"]) - np.array(
                    self.alternatives[j]["samples"]
                )
                S2_ij = np.var(diff, ddof=1)
                Ni_j = int(np.floor(self.h2 * S2_ij / self.delta**2))
                max_Ni = max(max_Ni, Ni_j)
            self.alternatives[i]["max_N"] = max_Ni

    def screen(self):
        survivors = set()
        for i in self.active:
            keep = True
            for j in self.active:
                if i == j:
                    continue
                xi = self.alternatives[i]["stats"].mean
                xj = self.alternatives[j]["stats"].mean
                diff_samples = np.array(self.alternatives[i]["samples"]) - np.array(
                    self.alternatives[j]["samples"]
                )
                S2_ij = np.var(diff_samples, ddof=1)
                Wi = max(
                    0.0,
                    (self.delta / (2 * self.c * self.r))
                    * (self.h2 * S2_ij / self.delta**2 - self.r),
                )
                if xi > xj + Wi:
                    keep = False
                    break
            if keep:
                survivors.add(i)
        self.active = survivors

    def step(self, updates):
        for name, value in updates.items():
            self.update(name, value)
        self.r += 1
        self.screen()

    def done(self):
        return len(self.active) == 1 or all(
            self.r > self.alternatives[i]["max_N"] for i in self.active
        )

    def best(self):
        if len(self.active) == 1:
            return next(iter(self.active))
        return min(self.active, key=lambda i: self.alternatives[i]["stats"].mean)
