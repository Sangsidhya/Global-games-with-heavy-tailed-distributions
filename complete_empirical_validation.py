"""
Empirical Validation for Global Games with GPD Noise
====================================================
Implements:
1. Uniqueness condition verification (gamma_GPD < 2*pi)
2. Gaussian information projection quality assessment
3. Threshold estimation via subsample bootstrap
4. Monte Carlo coverage simulations and robustness checks

Author: Sangsidhya Kar



from typing import Tuple, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# NumPy 2.0 removed np.trapz in favour of np.trapezoid.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================================
# PART 1: GPD DISTRIBUTION UTILITIES
# ============================================================================

class SymmetricGPD:
    """
    Symmetric Generalized Pareto Distribution centred at zero.

        f(x) = (1 / (2 * sigma_scale)) * (1 + xi * |x| / sigma_scale)^(-1/xi - 1)

    At xi = 0 this is the Laplace density with scale sigma_scale, obtained as
    the limit of the expression above.
    """

    def __init__(self, xi: float, sigma_scale: float):
        if xi >= 0.5:
            raise ValueError("xi must be < 0.5 for finite variance")
        if xi < 0:
            raise ValueError("xi must be >= 0; the equilibrium analysis covers [0, 0.5)")
        if sigma_scale <= 0:
            raise ValueError("sigma_scale must be positive")

        self.xi = float(xi)
        self.sigma_scale = float(sigma_scale)

        # Second moment of the symmetric GPD, matching the analytical result
        # E[X^2] = 2 * sigma_scale^2 / ((1 - xi) * (1 - 2 xi)).
        self.variance = (2 * self.sigma_scale ** 2) / ((1 - self.xi) * (1 - 2 * self.xi))

    def log_pdf(self, x) -> np.ndarray:
        """Log density. Returns an array matching the shape of x."""
        x = np.asarray(x, dtype=float)
        z = np.abs(x) / self.sigma_scale

        if self.xi == 0.0:
            # Laplace limit: log f(x) = -|x| / sigma_scale - log(2 sigma_scale)
            return -z - np.log(2 * self.sigma_scale)

        term = 1.0 + self.xi * z
        with np.errstate(divide="ignore", invalid="ignore"):
            log_val = (-1.0 / self.xi - 1.0) * np.log(term) - np.log(2 * self.sigma_scale)
        return np.where(term > 0, log_val, -np.inf)

    def pdf(self, x) -> np.ndarray:
        """Density. Returns an array matching the shape of x."""
        return np.exp(self.log_pdf(x))

    def rvs(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw samples by inverting the one-sided CDF and randomising the sign."""
        rng = np.random.default_rng() if rng is None else rng
        u = rng.uniform(0.0, 1.0, size)

        if self.xi == 0.0:
            y = -self.sigma_scale * np.log(1 - u)
        else:
            y = (self.sigma_scale / self.xi) * ((1 - u) ** (-self.xi) - 1)

        signs = rng.choice(np.array([-1.0, 1.0]), size=size)
        return signs * y

    def fisher_information(self) -> float:
        """
        Fisher information for the location parameter:
            I(theta) = (1 + xi)^2 / (sigma_scale^2 * (1 + 2 xi)).
        At xi = 0 this reduces to 1 / sigma_scale^2, the Laplace value.
        """
        return (1 + self.xi) ** 2 / (self.sigma_scale ** 2 * (1 + 2 * self.xi))


# ============================================================================
# PART 2: POSTERIOR COMPUTATION AND GAUSSIAN PROJECTION
# ============================================================================

class PosteriorGPD:
    """
    Beliefs about theta given a private signal x (GPD noise) and a public
    signal y ~ N(theta, tau^2), together with the Gaussian information
    projection used in the theory.
    """

    def __init__(self, gpd: SymmetricGPD, tau: float):
        if tau <= 0:
            raise ValueError("tau must be positive")

        self.gpd = gpd
        self.tau = float(tau)

        self.I_theta = gpd.fisher_information()
        self.H = self.I_theta + 1.0 / self.tau ** 2
        self.sigma_eff_sq = 1.0 / self.H
        self.sigma_noise_sq = gpd.variance
        self.Delta = np.sqrt(self.sigma_eff_sq + self.sigma_noise_sq)

        self.alpha_I = self.I_theta / self.H
        self.alpha_tau = (1.0 / self.tau ** 2) / self.H

    # -- exact posterior ---------------------------------------------------

    def log_posterior(self, theta, x: float, y: float) -> np.ndarray:
        """Unnormalised log posterior: GPD likelihood times Gaussian prior."""
        log_lik = self.gpd.log_pdf(np.asarray(theta, dtype=float) - x)  # symmetric in sign
        log_prior = stats.norm.logpdf(theta, loc=y, scale=self.tau)
        return log_lik + log_prior

    def _grid_halfwidth(self) -> float:
        """
        Integration half-width. The posterior spread is governed by the
        smaller of the prior scale and the likelihood scale, but the tails are
        set by the larger, so scale the grid by the larger of the two.
        """
        return 8.0 * max(self.tau, self.gpd.sigma_scale)

    def find_mode(self, x: float, y: float) -> float:
        """Posterior mode, by bounded minimisation of the negative log posterior."""
        w = self._grid_halfwidth()
        result = minimize_scalar(
            lambda theta: -float(self.log_posterior(theta, x, y)),
            bounds=(min(x, y) - w, max(x, y) + w),
            method="bounded",
        )
        return float(result.x)

    def true_posterior(self, x: float, y: float, n_grid: int = 20000
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact posterior on a grid, normalised by numerical integration."""
        mode = self.find_mode(x, y)
        w = self._grid_halfwidth()
        theta_grid = np.linspace(mode - w, mode + w, n_grid)

        log_post = self.log_posterior(theta_grid, x, y)
        finite = np.isfinite(log_post)
        post = np.zeros_like(theta_grid)
        post[finite] = np.exp(log_post[finite] - np.max(log_post[finite]))
        post /= _trapz(post, theta_grid)
        return theta_grid, post

    # -- projected (Gaussian) posterior ------------------------------------

    def projected_mean(self, x: float, y: float) -> float:
        """mu_post = alpha_I * x + alpha_tau * y, the linear update of the theory."""
        return self.alpha_I * x + self.alpha_tau * y

    def projected_posterior(self, theta_grid: np.ndarray, x: float, y: float) -> np.ndarray:
        """The Gaussian N(mu_post, sigma_eff^2) that the theory assumes players hold."""
        return stats.norm.pdf(theta_grid,
                              loc=self.projected_mean(x, y),
                              scale=np.sqrt(self.sigma_eff_sq))


# ============================================================================
# PART 3: UNIQUENESS CONDITION VERIFICATION
# ============================================================================

@dataclass
class UniquenessResult:
    xi: float
    sigma_scale: float
    tau: float
    gamma_gpd: float
    predicts_unique: bool
    slope_at_fixed_point: float
    n_equilibria_mean: float
    n_equilibria_std: float
    unique_rate: float
    multiple_rate: float
    theory_validated: bool
    equilibria_counts: List[int] = field(default_factory=list)


def compute_gamma_gpd(xi: float, sigma_scale: float, tau: float) -> float:
    """
    gamma_GPD = sigma_eff^4 / [ (tau^2 - sigma_eff^2)^2 * (sigma_eff^2 + sigma_noise^2) ].
    Uniqueness holds when this is below 2*pi.
    """
    posterior = PosteriorGPD(SymmetricGPD(xi, sigma_scale), tau)
    se2 = posterior.sigma_eff_sq
    sn2 = posterior.sigma_noise_sq
    return se2 ** 2 / (((tau ** 2 - se2) ** 2) * (se2 + sn2))


def best_response_slope(posterior: PosteriorGPD) -> float:
    """
    Slope of the best response at the symmetric fixed point,
        (1/alpha_I) * [phi(0)/Delta] / [1 + phi(0)/Delta].
    The contraction condition is that this is below 1; it is algebraically
    equivalent to gamma_GPD < 2*pi.
    """
    r = 1.0 / (np.sqrt(2 * np.pi) * posterior.Delta)
    return (1.0 / posterior.alpha_I) * (r / (1.0 + r))


def find_switching_equilibria(posterior: PosteriorGPD, y: float,
                              n_grid: int = 2000,
                              search_halfwidth: Optional[float] = None) -> List[float]:
    """
    Locate all switching equilibria: thresholds k solving
        mu_post(k, y) = Phi( (k - mu_post(k, y)) / Delta ).

    The search interval is chosen wide enough to bracket the fixed point. The
    earlier version searched only k in [0, 1]; for plausible parameters the
    fixed point lies outside that interval (for example xi = 0.3,
    sigma_scale = 1.5, tau = 1, y = 0.3), so equilibria were being missed and
    counted as non-existent.
    """
    aI, aT, D = posterior.alpha_I, posterior.alpha_tau, posterior.Delta

    def indiff(k: float) -> float:
        mu = aI * k + aT * y
        return mu - stats.norm.cdf((k - mu) / D)

    if search_halfwidth is None:
        # mu ranges over [0, 1] once alpha_I * k spans that range; pad generously.
        search_halfwidth = 5.0 * (1.0 + 1.0 / aI + D)

    lo, hi = 0.5 - search_halfwidth, 0.5 + search_halfwidth
    k_grid = np.linspace(lo, hi, n_grid)
    mu_grid = aI * k_grid + aT * y
    values = mu_grid - stats.norm.cdf((k_grid - mu_grid) / D)

    equilibria: List[float] = []
    tol = (hi - lo) / n_grid * 2
    for idx in np.where(np.diff(np.sign(values)) != 0)[0]:
        try:
            k_eq = brentq(indiff, k_grid[idx], k_grid[idx + 1])
        except Exception:
            continue
        if not equilibria or min(abs(k_eq - e) for e in equilibria) > tol:
            equilibria.append(k_eq)
    return equilibria


def test_uniqueness_condition(xi: float, sigma_scale: float, tau: float,
                              n_sims: int = 100, seed: int = 0) -> UniquenessResult:
    """Check that gamma_GPD < 2*pi coincides with a single switching equilibrium."""
    rng = np.random.default_rng(seed)
    posterior = PosteriorGPD(SymmetricGPD(xi, sigma_scale), tau)
    gamma = compute_gamma_gpd(xi, sigma_scale, tau)
    slope = best_response_slope(posterior)

    counts = [len(find_switching_equilibria(posterior, rng.uniform(0.3, 0.7)))
              for _ in range(n_sims)]

    unique_rate = float(np.mean([c == 1 for c in counts]))
    multiple_rate = float(np.mean([c > 1 for c in counts]))
    predicts_unique = gamma < 2 * np.pi

    # Two-sided check. The earlier version returned True automatically whenever
    # the theory predicted multiplicity, so the gamma >= 2*pi branch was never
    # tested at all; combined with configurations that all sat far below 2*pi,
    # nothing in the suite could ever have falsified the condition.
    theory_validated = (unique_rate > 0.90) if predicts_unique else (multiple_rate > 0.90)

    return UniquenessResult(
        xi=xi, sigma_scale=sigma_scale, tau=tau,
        gamma_gpd=gamma, predicts_unique=predicts_unique,
        slope_at_fixed_point=slope,
        n_equilibria_mean=float(np.mean(counts)),
        n_equilibria_std=float(np.std(counts)),
        unique_rate=unique_rate,
        multiple_rate=multiple_rate,
        theory_validated=theory_validated,
        equilibria_counts=counts,
    )


# ============================================================================
# PART 4: GAUSSIAN PROJECTION QUALITY
# ============================================================================

@dataclass
class ProjectionQuality:
    xi: float
    sigma_ratio: float           # sigma_scale / tau
    kl_projection_mean: float    # KL(true || N(mu_post, sigma_eff^2)), the theory's belief
    kl_projection_max: float
    kl_moment_matched_mean: float  # KL(true || N(E[theta], Var[theta])), the attainable floor
    l1_distance_mean: float
    variance_ratio_mean: float   # sigma_eff^2 / Var[theta | x, y]
    variance_ratio_std: float


def test_projection_quality(xi: float, sigma_ratio: float,
                            n_tests: int = 100, seed: int = 0) -> ProjectionQuality:
    """
    Compare the exact posterior with the Gaussian the theory assumes.

    Two Gaussians are scored:
      * the projection actually used in the model, N(mu_post, sigma_eff^2);
      * the moment-matched Gaussian, which minimises KL(true || Gaussian) and
        therefore bounds below what any Gaussian belief can achieve.
    """
    rng = np.random.default_rng(seed)
    tau = 1.0
    sigma_scale = sigma_ratio * tau
    posterior = PosteriorGPD(SymmetricGPD(xi, sigma_scale), tau)

    kl_proj, kl_mm, l1s, var_ratios = [], [], [], []

    for _ in range(n_tests):
        x = rng.uniform(0.2, 0.8)
        y = rng.uniform(0.2, 0.8)

        grid, p_true = posterior.true_posterior(x, y)

        mean_true = _trapz(grid * p_true, grid)
        var_true = _trapz((grid - mean_true) ** 2 * p_true, grid)

        p_proj = posterior.projected_posterior(grid, x, y)
        p_mm = stats.norm.pdf(grid, loc=mean_true, scale=np.sqrt(var_true))

        def kl(q):
            mask = (p_true > 1e-14) & (q > 1e-14)
            return max(0.0, float(_trapz(p_true[mask] * np.log(p_true[mask] / q[mask]),
                                         grid[mask])))

        kl_proj.append(kl(p_proj))
        kl_mm.append(kl(p_mm))
        l1s.append(float(_trapz(np.abs(p_true - p_proj), grid)))
        var_ratios.append(posterior.sigma_eff_sq / var_true)

    return ProjectionQuality(
        xi=xi,
        sigma_ratio=sigma_ratio,
        kl_projection_mean=float(np.mean(kl_proj)),
        kl_projection_max=float(np.max(kl_proj)),
        kl_moment_matched_mean=float(np.mean(kl_mm)),
        l1_distance_mean=float(np.mean(l1s)),
        variance_ratio_mean=float(np.mean(var_ratios)),
        variance_ratio_std=float(np.std(var_ratios)),
    )


# ============================================================================
# PART 5: THRESHOLD ESTIMATION
# ============================================================================

def subsample_bootstrap_ci(data: np.ndarray, m: int, B: int = 999,
                           alpha: float = 0.05,
                           rng: Optional[np.random.Generator] = None
                           ) -> Tuple[float, float, float]:
    """
    Politis-Romano subsampling interval for the median.

    Draws B subsamples of size m without replacement, forms
    sqrt(m) * (median_m - median_n), and inverts the quantiles at rate
    sqrt(n). Vectorized over the B replications.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(data)
    if not 0 < m < n:
        raise ValueError("subsample size m must satisfy 0 < m < n")

    theta_hat_n = float(np.median(data))

    # B independent subsamples without replacement, drawn in one shot:
    # argsort of uniform noise gives an independent permutation in each row.
    idx = np.argsort(rng.random((B, n)), axis=1)[:, :m]
    theta_hat_m = np.median(data[idx], axis=1)
    boot = np.sqrt(m) * (theta_hat_m - theta_hat_n)

    q_lower, q_upper = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return theta_hat_n, theta_hat_n - q_upper / np.sqrt(n), theta_hat_n - q_lower / np.sqrt(n)


def naive_t_ci(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Textbook t interval for the mean."""
    theta_hat = float(np.mean(data))
    se = float(stats.sem(data))
    t_crit = stats.t.ppf(1 - alpha / 2, df=len(data) - 1)
    return theta_hat, theta_hat - t_crit * se, theta_hat + t_crit * se


def median_normal_ci(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Normal-approximation interval for the median using the asymptotic standard
    error 1 / (2 f(theta) sqrt(n)).

    The density at the median is estimated by quantile spacing rather than by
    a Gaussian kernel: kernel bandwidths chosen by Scott's or Silverman's rule
    scale with the sample standard deviation, which is enormous under heavy
    tails, so the kernel oversmooths, understates f, and inflates the interval
    to the point of covering essentially always.

    Included so that the comparison separates the choice of estimator (median
    against mean) from the choice of inference method (subsampling against a
    normal approximation). Without it, any subsampling gain confounds the two.
    """
    n = len(data)
    theta_hat = float(np.median(data))
    h = max(0.05, n ** (-1 / 3))          # spacing half-width in probability
    lo_q, hi_q = np.quantile(data, [0.5 - h, 0.5 + h])
    f_hat = (2 * h) / max(hi_q - lo_q, 1e-12)
    se = 1.0 / (2 * f_hat * np.sqrt(n))
    z = stats.norm.ppf(1 - alpha / 2)
    return theta_hat, theta_hat - z * se, theta_hat + z * se


# ============================================================================
# PART 6: MONTE CARLO COVERAGE
# ============================================================================

@dataclass
class MonteCarloResults:
    xi: float
    n: int
    m: int
    subsample_coverage: float
    naive_coverage: float
    median_normal_coverage: float
    subsample_width_median: float
    naive_width_median: float
    median_normal_width_median: float
    efficiency_gain: float       # naive width divided by subsample width (>1 favours subsampling)
    subsample_ci_list: List[Tuple[float, float]] = field(default_factory=list)
    naive_ci_list: List[Tuple[float, float]] = field(default_factory=list)


def run_coverage_simulation(xi: float, sigma_scale: float, n: int = 500,
                            N_sim: int = 10000, true_theta: float = 0.5,
                            m_exp: float = 0.7, B: int = 999,
                            seed: int = 0, store_cis: bool = False
                            ) -> MonteCarloResults:
    """Coverage of the three intervals against the known threshold."""
    rng = np.random.default_rng(seed)
    gpd = SymmetricGPD(xi, sigma_scale)
    m = int(np.floor(n ** m_exp))

    ss_cov, nv_cov, mn_cov = [], [], []
    ss_w, nv_w, mn_w = [], [], []
    ss_cis, nv_cis = [], []

    for _ in range(N_sim):
        signals = true_theta + gpd.rvs(n, rng=rng)

        _, lo, hi = subsample_bootstrap_ci(signals, m, B=B, rng=rng)
        ss_cov.append(lo <= true_theta <= hi)
        ss_w.append(hi - lo)

        _, nlo, nhi = naive_t_ci(signals)
        nv_cov.append(nlo <= true_theta <= nhi)
        nv_w.append(nhi - nlo)

        _, mlo, mhi = median_normal_ci(signals)
        mn_cov.append(mlo <= true_theta <= mhi)
        mn_w.append(mhi - mlo)

        if store_cis:
            ss_cis.append((lo, hi))
            nv_cis.append((nlo, nhi))

    return MonteCarloResults(
        xi=xi, n=n, m=m,
        subsample_coverage=float(np.mean(ss_cov)),
        naive_coverage=float(np.mean(nv_cov)),
        median_normal_coverage=float(np.mean(mn_cov)),
        subsample_width_median=float(np.median(ss_w)),
        naive_width_median=float(np.median(nv_w)),
        median_normal_width_median=float(np.median(mn_w)),
        efficiency_gain=float(np.median(nv_w) / np.median(ss_w)),
        subsample_ci_list=ss_cis,
        naive_ci_list=nv_cis,
    )


# ============================================================================
# PART 7: ROBUSTNESS CHECKS
# ============================================================================

def robustness_m_choice(xi: float = 1 / 3, sigma_scale: float = 1.0,
                        n: int = 500, N_sim: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Coverage across subsample sizes m = n^0.6 through n^0.8."""
    rows = []
    for m_exp in [0.6, 0.65, 0.7, 0.75, 0.8]:
        res = run_coverage_simulation(xi, sigma_scale, n, N_sim, m_exp=m_exp, seed=seed)
        rows.append({"m_exponent": m_exp, "m": res.m,
                     "coverage": res.subsample_coverage,
                     "width_median": res.subsample_width_median})
        print(f"    m = n^{m_exp:.2f} = {res.m}: coverage {res.subsample_coverage:.4f}")
    return pd.DataFrame(rows)


def robustness_tail_index(n: int = 500, N_sim: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Coverage across tail parameters, including the exponential case xi = 0."""
    configs = [(0.0, "xi=0.00 (exponential tails)"),
               (0.1, "xi=0.10 (light)"),
               (0.2, "xi=0.20 (moderate)"),
               (1 / 3, "xi=0.33 (heavy)"),
               (0.4, "xi=0.40 (very heavy)")]
    rows = []
    for xi, label in configs:
        res = run_coverage_simulation(xi, 1.0, n, N_sim, seed=seed)
        rows.append({"xi": xi, "label": label,
                     "subsample_coverage": res.subsample_coverage,
                     "naive_coverage": res.naive_coverage,
                     "median_normal_coverage": res.median_normal_coverage,
                     "coverage_difference": res.subsample_coverage - res.naive_coverage})
        print(f"    {label}: subsample {res.subsample_coverage:.4f}, "
              f"naive {res.naive_coverage:.4f}")
    return pd.DataFrame(rows)


def robustness_sample_size(xi: float = 1 / 3, sigma_scale: float = 1.0,
                           N_sim: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Coverage and interval width across sample sizes."""
    rows = []
    for n in [100, 250, 500, 1000]:
        res = run_coverage_simulation(xi, sigma_scale, n, N_sim, seed=seed)
        rows.append({"n": n,
                     "subsample_coverage": res.subsample_coverage,
                     "naive_coverage": res.naive_coverage,
                     "median_normal_coverage": res.median_normal_coverage,
                     "subsample_width_median": res.subsample_width_median,
                     "naive_width_median": res.naive_width_median,
                     "efficiency_gain": res.efficiency_gain})
        print(f"    n = {n}: subsample {res.subsample_coverage:.4f}, "
              f"naive {res.naive_coverage:.4f}")
    return pd.DataFrame(rows)


def robustness_bootstrap_replications(xi: float = 1 / 3, sigma_scale: float = 1.0,
                                      n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Stability of the interval across the number of subsampling replications."""
    rng = np.random.default_rng(seed)
    gpd = SymmetricGPD(xi, sigma_scale)
    m = int(np.floor(n ** 0.7))
    signals = 0.5 + gpd.rvs(n, rng=rng)   # a single fixed dataset

    rows = []
    for B in [199, 499, 999, 1999, 4999]:
        cis = [subsample_bootstrap_ci(signals, m, B=B, rng=rng)[1:] for _ in range(100)]
        lower = [c[0] for c in cis]
        upper = [c[1] for c in cis]
        widths = [c[1] - c[0] for c in cis]
        rows.append({"B": B,
                     "mean_lower": np.mean(lower), "sd_lower": np.std(lower),
                     "mean_upper": np.mean(upper), "sd_upper": np.std(upper),
                     "mean_width": np.mean(widths), "sd_width": np.std(widths)})
        print(f"    B = {B}: width sd {np.std(widths):.6f}")
    return pd.DataFrame(rows)


def gamma_sensitivity_table(tau: float = 1.0) -> pd.DataFrame:
    """gamma_GPD across the parameter grid, with the uniqueness verdict."""
    rows = []
    for xi in [0.0, 0.1, 0.2, 1 / 3, 0.4]:
        for sigma in [0.5, 0.75, 1.0, 1.5, 2.0]:
            gamma = compute_gamma_gpd(xi, sigma, tau)
            rows.append({"xi": round(xi, 3), "sigma_scale": sigma,
                         "sigma_over_tau": sigma / tau,
                         "gamma_GPD": gamma,
                         "gamma_over_2pi": gamma / (2 * np.pi),
                         "unique": "Yes" if gamma < 2 * np.pi else "No"})
    return pd.DataFrame(rows)


# ============================================================================
# PART 8: FIGURES
# ============================================================================

def plot_coverage_panels(results: dict, path: str = "figure1_coverage_rates.pdf") -> None:
    main = results["main_coverage"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    methods = ["Subsample\nbootstrap", "Median\nnormal", "Naive\nt-test"]
    covs = [main.subsample_coverage * 100, main.median_normal_coverage * 100,
            main.naive_coverage * 100]
    bars = ax.bar(methods, covs, color=["#2ecc71", "#3498db", "#e74c3c"],
                  alpha=0.75, edgecolor="black", linewidth=1.5)
    ax.axhline(95, color="black", ls="--", lw=2, label="Nominal 95%")
    ax.set_ylabel("Coverage (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"A. Coverage (n={main.n}, xi={main.xi:.2f})",
                 fontsize=12, fontweight="bold")
    ax.set_ylim([80, 100])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, c in zip(bars, covs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax = axes[0, 1]
    mr = results["m_robustness"]
    ax.plot(mr["m_exponent"], mr["coverage"] * 100, marker="o", ms=8, lw=2, color="#3498db")
    ax.axhline(95, color="red", ls="--", lw=2, label="Nominal 95%")
    ax.fill_between(mr["m_exponent"], 94, 96, alpha=0.2, color="green", label="Band of one point")
    ax.set_xlabel("Subsample size exponent (m = n^exp)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coverage (%)", fontsize=12, fontweight="bold")
    ax.set_title("B. Robustness to subsample size", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    tr = results["tail_robustness"]
    x = np.arange(len(tr))
    ax.bar(x - 0.2, tr["subsample_coverage"] * 100, 0.4, label="Subsample bootstrap",
           color="#2ecc71", alpha=0.75, edgecolor="black")
    ax.bar(x + 0.2, tr["naive_coverage"] * 100, 0.4, label="Naive t-test",
           color="#e74c3c", alpha=0.75, edgecolor="black")
    ax.axhline(95, color="black", ls="--", lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in tr["xi"]])
    ax.set_xlabel("Tail parameter xi", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coverage (%)", fontsize=12, fontweight="bold")
    ax.set_title("C. Robustness to tail index", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([80, 100])

    ax = axes[1, 1]
    sr = results["size_robustness"]
    ax.plot(sr["n"], sr["subsample_coverage"] * 100, marker="o", ms=8, lw=2,
            label="Subsample bootstrap", color="#2ecc71")
    ax.plot(sr["n"], sr["naive_coverage"] * 100, marker="s", ms=8, lw=2,
            label="Naive t-test", color="#e74c3c")
    ax.axhline(95, color="black", ls="--", lw=2)
    ax.set_xlabel("Sample size n", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coverage (%)", fontsize=12, fontweight="bold")
    ax.set_title("D. Robustness to sample size", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_projection_quality(results: dict, path: str = "figure2_projection_quality.pdf") -> None:
    df = pd.DataFrame([vars(r) for r in results["projection"]])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    kl_pivot = df.pivot(index="xi", columns="sigma_ratio", values="kl_projection_mean")
    sns.heatmap(kl_pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                cbar_kws={"label": "KL divergence"}, ax=axes[0])
    axes[0].set_xlabel("sigma_scale / tau", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("xi (tail parameter)", fontsize=12, fontweight="bold")
    axes[0].set_title("A. KL(true || projected), lower is better",
                      fontsize=12, fontweight="bold")

    vr_pivot = df.pivot(index="xi", columns="sigma_ratio", values="variance_ratio_mean")
    sns.heatmap(vr_pivot, annot=True, fmt=".3f", cmap="coolwarm", center=1.0,
                cbar_kws={"label": "sigma_eff^2 / Var[theta | x, y]"}, ax=axes[1])
    axes[1].set_xlabel("sigma_scale / tau", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("xi (tail parameter)", fontsize=12, fontweight="bold")
    axes[1].set_title("B. Variance ratio, 1.0 is exact", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gamma_heatmap(tau: float = 1.0, path: str = "figure3_gamma_heatmap.pdf") -> None:
    xi_range = np.linspace(0.0, 0.45, 50)
    sigma_range = np.linspace(0.2, 2.0, 50)
    Xi, Sigma = np.meshgrid(xi_range, sigma_range)
    Gamma = np.array([[compute_gamma_gpd(Xi[j, i], Sigma[j, i], tau)
                       for i in range(len(xi_range))] for j in range(len(sigma_range))])

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.contourf(Xi, Sigma, Gamma, levels=40, cmap="viridis")
    plt.colorbar(im, ax=ax).set_label("gamma_GPD", fontsize=12, fontweight="bold")

    if Gamma.min() < 2 * np.pi < Gamma.max():
        cs = ax.contour(Xi, Sigma, Gamma, levels=[2 * np.pi], colors="red", linewidths=3)
        ax.clabel(cs, inline=True, fontsize=10, fmt="gamma=2pi")
        note = "Red contour: gamma = 2pi boundary"
    else:
        note = (f"gamma stays in [{Gamma.min():.3f}, {Gamma.max():.3f}], "
                f"far below 2pi = {2*np.pi:.3f}:\nuniqueness holds throughout this grid")

    ax.set_xlabel("xi (tail parameter)", fontsize=12, fontweight="bold")
    ax.set_ylabel("sigma_scale", fontsize=12, fontweight="bold")
    ax.set_title(f"Uniqueness parameter landscape (tau = {tau})\n{note}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_posterior_examples(xi: float = 1 / 3, sigma_ratio: float = 0.5,
                            path: str = "figure4_posterior_examples.pdf",
                            seed: int = 42) -> None:
    """Exact posterior against the projected Gaussian, for four random signal pairs."""
    tau = 1.0
    posterior = PosteriorGPD(SymmetricGPD(xi, sigma_ratio * tau), tau)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, ax in enumerate(axes.flatten()):
        x, y = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        grid, p_true = posterior.true_posterior(x, y)
        p_proj = posterior.projected_posterior(grid, x, y)

        mask = (p_true > 1e-14) & (p_proj > 1e-14)
        kl = max(0.0, float(_trapz(p_true[mask] * np.log(p_true[mask] / p_proj[mask]),
                                   grid[mask])))

        span = 4 * np.sqrt(posterior.sigma_eff_sq) + 2 * posterior.gpd.sigma_scale
        centre = posterior.projected_mean(x, y)
        ax.plot(grid, p_true, "b-", lw=2, label="Exact posterior", alpha=0.85)
        ax.plot(grid, p_proj, "r--", lw=2, label="Projected Gaussian")
        ax.set_xlim(centre - span, centre + span)
        ax.set_title(f"x={x:.3f}, y={y:.3f}, KL={kl:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("theta", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Projection quality (xi = {xi:.2f}, sigma_scale/tau = {sigma_ratio})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PART 9: MAIN SUITE
# ============================================================================

def run_full_validation_suite(quick: bool = False) -> dict:
    """Run every test. Set quick=True for a fast smoke run."""
    N_main = 500 if quick else 10000
    N_rob = 200 if quick else 2000
    N_size = 200 if quick else 1500
    n_proj = 20 if quick else 100
    n_uniq = 20 if quick else 100

    print("=" * 78)
    print("EMPIRICAL VALIDATION SUITE" + ("  [quick mode]" if quick else ""))
    print("=" * 78)

    print("\n[1/6] Uniqueness condition (gamma_GPD < 2*pi)...")
    uniqueness = []
    for cfg in [# gamma below 2*pi: uniqueness predicted
                {"xi": 0.0, "sigma_scale": 0.5, "tau": 1.0},
                {"xi": 0.1, "sigma_scale": 0.5, "tau": 1.0},
                {"xi": 0.2, "sigma_scale": 1.0, "tau": 1.0},
                {"xi": 0.3, "sigma_scale": 1.5, "tau": 1.0},
                # gamma above 2*pi: multiplicity predicted, so the condition
                # is tested in both directions rather than only where it holds
                {"xi": 0.0, "sigma_scale": 5.0, "tau": 0.3},
                {"xi": 0.2, "sigma_scale": 10.0, "tau": 0.2},
                {"xi": 0.3, "sigma_scale": 20.0, "tau": 0.15}]:
        r = test_uniqueness_condition(**cfg, n_sims=n_uniq)
        uniqueness.append(r)
        print(f"    xi={r.xi:.2f}, sigma_scale={r.sigma_scale}: gamma={r.gamma_gpd:.4f}, "
              f"slope={r.slope_at_fixed_point:.3f}, unique rate={r.unique_rate:.3f}, "
              f"multiple rate={r.multiple_rate:.3f}")

    print("\n[2/6] Gaussian projection quality...")
    projection = []
    for xi in [0.0, 0.1, 0.2, 1 / 3, 0.4]:
        for ratio in [0.25, 0.5, 1.0, 2.0]:
            q = test_projection_quality(xi, ratio, n_tests=n_proj)
            projection.append(q)
            print(f"    xi={xi:.2f}, sigma/tau={ratio}: KL={q.kl_projection_mean:.4f} "
                  f"(floor {q.kl_moment_matched_mean:.4f}), "
                  f"var ratio={q.variance_ratio_mean:.3f}")

    print(f"\n[3/6] Main coverage simulation (n=500, N={N_main})...")
    main = run_coverage_simulation(xi=1 / 3, sigma_scale=1.0, n=500,
                                   N_sim=N_main, store_cis=True)
    print(f"    Subsample bootstrap: {main.subsample_coverage:.4f}")
    print(f"    Median, normal approximation: {main.median_normal_coverage:.4f}")
    print(f"    Naive t-test: {main.naive_coverage:.4f}")

    print("\n[4/6] Robustness to subsample size...")
    m_rob = robustness_m_choice(N_sim=N_rob)

    print("\n[5/6] Robustness to tail index and sample size...")
    tail_rob = robustness_tail_index(N_sim=N_rob)
    size_rob = robustness_sample_size(N_sim=N_size)

    print("\n[6/6] Subsampling replication convergence...")
    convergence = robustness_bootstrap_replications()

    return {"uniqueness": uniqueness, "projection": projection,
            "main_coverage": main, "m_robustness": m_rob,
            "tail_robustness": tail_rob, "size_robustness": size_rob,
            "convergence": convergence,
            "gamma_sensitivity": gamma_sensitivity_table()}


def print_summary(results: dict) -> None:
    main = results["main_coverage"]
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print("\n1. COVERAGE (n=%d, xi=%.2f, m=%d)" % (main.n, main.xi, main.m))
    print(f"   Subsample bootstrap (median):   {main.subsample_coverage * 100:6.2f}%")
    print(f"   Normal approximation (median):  {main.median_normal_coverage * 100:6.2f}%")
    print(f"   Naive t-test (mean):            {main.naive_coverage * 100:6.2f}%")
    print(f"   Naive interval width divided by subsample width: {main.efficiency_gain:.2f}")
    print("   Note: the t interval does NOT undercover here. With xi < 0.5 the")
    print("   variance is finite, the central limit theorem applies, and at these")
    print("   sample sizes the t interval sits at or slightly above nominal. The")
    print("   gain from the median plus subsampling is width, not coverage: the")
    print("   interval is several times narrower, and the margin widens with xi.")

    print("\n2. UNIQUENESS")
    for r in results["uniqueness"]:
        verdict = "passed" if r.theory_validated else "FAILED"
        print(f"   xi={r.xi:.2f}, sigma_scale={r.sigma_scale}: gamma={r.gamma_gpd:.4f} "
              f"({'<' if r.predicts_unique else '>='} 2pi), unique rate "
              f"{r.unique_rate:.3f}, multiple rate {r.multiple_rate:.3f} [{verdict}]")

    print("\n3. PROJECTION QUALITY (the main negative finding)")
    df = pd.DataFrame([vars(r) for r in results["projection"]])
    by_ratio = df.groupby("sigma_ratio").agg(
        mean_KL=("kl_projection_mean", "mean"),
        KL_floor=("kl_moment_matched_mean", "mean"),
        var_ratio=("variance_ratio_mean", "mean")).reset_index()
    print(by_ratio.to_string(index=False))
    print("   The projection is WORST at small sigma_scale/tau and improves as the")
    print("   ratio grows, the opposite of the ordering the paper asserts. When the")
    print("   private signal is precise the likelihood cusp dominates the posterior,")
    print("   so the Gaussian fits badly; when it is diffuse the Gaussian prior")
    print("   dominates and the fit is close. sigma_eff^2 also understates the true")
    print("   posterior variance substantially at small ratios.")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    import sys
    matplotlib.use("Agg")
    np.random.seed(2025)

    quick = "--quick" in sys.argv
    results = run_full_validation_suite(quick=quick)

    print("\nGenerating figures...")
    plot_coverage_panels(results)
    plot_projection_quality(results)
    plot_gamma_heatmap()
    plot_posterior_examples()

    results["gamma_sensitivity"].to_csv("gamma_sensitivity_table.csv", index=False)
    print("  Saved: gamma_sensitivity_table.csv")

    print_summary(results)
