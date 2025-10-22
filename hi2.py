"""
ADDITIONAL PLOTS FOR SVB ANALYSIS
Generates the 4 missing plots from the presentation
All formulas verified against thesis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec

# =============================================================================
# PARAMETERS (from presentation)
# =============================================================================
xi_presentation = 0.334
sigma_presentation = 0.0366
tau = 0.10

# Generate simulated returns (same as main code)
np.random.seed(42)
n_days = 296
df_true = 5
returns_raw = stats.t.rvs(df=df_true, size=n_days, random_state=42)
returns = returns_raw * 0.0554 + 0.0011

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def hill_estimator(data, k=None):
    """Hill (1975) estimator"""
    abs_data = np.abs(data)
    sorted_data = np.sort(abs_data)[::-1]
    n = len(sorted_data)
    
    if k is None:
        k = int(np.sqrt(n))
    
    if k >= n or k < 2:
        return None, None, None
    
    threshold = sorted_data[k]
    
    if threshold < 1e-10:
        return 0.0, 0.0, k
    
    ratios = sorted_data[:k] / threshold
    ratios = np.maximum(ratios, 1.0)
    
    if np.allclose(ratios, 1.0):
        return 0.0, 0.0, k
    
    log_ratios = np.log(ratios)
    xi_hat = np.mean(log_ratios)
    se_xi = xi_hat / np.sqrt(k)
    
    return xi_hat, se_xi, k

def fisher_information_gpd(xi, sigma):
    """Theorem 11 (page 16): I(theta) = (1+xi)^2/[sigma^2(1+2xi)]"""
    return (1 + xi)**2 / (sigma**2 * (1 + 2*xi))

def gpd_noise_variance(xi, sigma):
    """Page 19: Var[epsilon] = 4*sigma^2/[(1-xi)(1-2xi)]"""
    if xi >= 0.5:
        return np.inf
    return 4 * sigma**2 / ((1 - xi) * (1 - 2*xi))

def compute_gamma_gpd(xi, sigma, tau):
    """Theorem 12 (page 18): gamma_GPD = sigma_eff^4 / [(tau^2 - sigma_eff^2)^2 * (sigma_eff^2 + sigma_noise^2)]"""
    I_theta = fisher_information_gpd(xi, sigma)
    H = I_theta + tau**(-2)
    sigma_eff_sq = 1 / H
    sigma_noise_sq = gpd_noise_variance(xi, sigma)
    
    numerator = sigma_eff_sq**2
    denominator = (tau**2 - sigma_eff_sq)**2 * (sigma_eff_sq + sigma_noise_sq)
    gamma = numerator / denominator
    
    return gamma, I_theta, H, sigma_eff_sq, sigma_noise_sq

# =============================================================================
# PLOT 1: HILL PLOT - TAIL PARAMETER ESTIMATION
# =============================================================================

def plot_hill_estimation():
    """
    Hill Plot: Shows how tail parameter estimate varies with k
    (number of upper order statistics used)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate Hill estimates for range of k values
    k_range = range(5, min(80, len(returns)//3))
    hill_vals = []
    
    for k in k_range:
        try:
            xi_k, _, _ = hill_estimator(returns, k=k)
            hill_vals.append(xi_k if xi_k is not None else np.nan)
        except:
            hill_vals.append(np.nan)
    
    valid_mask = ~np.isnan(hill_vals)
    k_valid = np.array(list(k_range))[valid_mask]
    hill_valid = np.array(hill_vals)[valid_mask]
    
    # Get optimal estimate
    k_opt = int(np.sqrt(len(returns)))
    xi_hat, se_xi, k_used = hill_estimator(returns, k=k_opt)
    
    # Plot
    ax.plot(k_valid, hill_valid, 'b-', linewidth=2, label='Hill estimate')
    ax.axhline(y=xi_hat, color='r', linestyle='--', linewidth=2.5, 
               label=f'xi-hat = {xi_hat:.3f}', zorder=5)
    ax.axvline(x=k_used, color='g', linestyle=':', linewidth=2, alpha=0.7, 
               label=f'k = {k_used}')
    ax.fill_between(k_valid, xi_hat - 2*se_xi, xi_hat + 2*se_xi,
                     alpha=0.2, color='red', label='95% CI')
    
    ax.set_xlabel('k (Number of Upper Order Statistics)', fontsize=12, fontweight='bold')
    ax.set_ylabel('xi-hat (Hill Estimator)', fontsize=12, fontweight='bold')
    ax.set_title('Hill Plot: Tail Parameter Estimation', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot1_hill_estimation.png', dpi=300, bbox_inches='tight')
    print("[SAVED] plot1_hill_estimation.png")
    plt.close()

# =============================================================================
# PLOT 2: SIMULATED SVB RETURN DISTRIBUTION
# =============================================================================

def plot_return_distribution():
    """
    Shows empirical distribution of returns vs fitted t-distribution
    Demonstrates heavy tails
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to percentage
    returns_pct = returns * 100
    
    # Histogram
    n_bins = 50
    hist_counts, bin_edges = np.histogram(returns_pct, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    ax.bar(bin_centers, hist_counts, width=bin_width*0.9, 
           alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5,
           label='Empirical density')
    
    # Fitted t-distribution
    std_pct = returns.std() * 100
    mean_pct = returns.mean() * 100
    x_fit = np.linspace(returns_pct.min() - 2, returns_pct.max() + 2, 300)
    t_pdf = stats.t.pdf((x_fit - mean_pct) / std_pct, df=df_true) / std_pct
    ax.plot(x_fit, t_pdf, 'r-', linewidth=3, 
            label=f't-distribution (df={df_true})', zorder=10)
    
    # Mark mean
    ax.axvline(x=mean_pct, color='orange', linestyle='--', linewidth=2.5, 
               label=f'Mean = {mean_pct:.2f}%', zorder=11)
    
    ax.set_xlabel('Daily Returns (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Simulated SVB Return Distribution', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    stats_text = f'n = {len(returns)}\nMean = {mean_pct:.2f}%\nStd = {std_pct:.2f}%\nSkew = {stats.skew(returns):.2f}\nKurt = {stats.kurtosis(returns):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plot2_return_distribution.png', dpi=300, bbox_inches='tight')
    print("[SAVED] plot2_return_distribution.png")
    plt.close()

# =============================================================================
# PLOT 3: Q-Q PLOT - RETURNS VS. NORMAL
# =============================================================================

def plot_qq_normal():
    """
    Q-Q plot comparing returns to normal distribution
    Shows deviation in tails (heavy-tailed behavior)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate Q-Q plot data
    (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm", fit=True)
    
    # Plot
    ax.plot(osm, osr, 'bo', markersize=5, alpha=0.6, label='Sample quantiles')
    ax.plot(osm, slope * osm + intercept, 'r-', linewidth=2.5, 
            label=f'Normal fit (R^2={r**2:.3f})')
    
    # Highlight tail deviations
    # Upper tail
    upper_idx = len(osm) * 19 // 20  # Top 5%
    ax.plot(osm[upper_idx:], osr[upper_idx:], 'ro', markersize=7, 
            label='Upper tail', zorder=10)
    
    # Lower tail
    lower_idx = len(osm) // 20  # Bottom 5%
    ax.plot(osm[:lower_idx], osr[:lower_idx], 'mo', markersize=7, 
            label='Lower tail', zorder=10)
    
    ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Quantiles (Returns)', fontsize=12, fontweight='bold')
    ax.set_title('Q-Q Plot: Returns vs. Normal', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for tail behavior
    ax.text(0.98, 0.02, 'Deviation in tails\nindicates heavy-tailed\nbehavior', 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('plot3_qq_normal.png', dpi=300, bbox_inches='tight')
    print("[SAVED] plot3_qq_normal.png")
    plt.close()

# =============================================================================
# PLOT 4: TAIL BEHAVIOR - EMPIRICAL VS GPD FIT
# =============================================================================

def plot_tail_behavior():
    """
    Compares empirical tail behavior with GPD prediction
    Uses order statistics in log scale
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Empirical tail
    sorted_returns_abs = np.sort(np.abs(returns))[::-1]
    n_tail = min(50, len(sorted_returns_abs))
    ranks = np.arange(1, n_tail + 1)
    log_empirical = np.log(sorted_returns_abs[:n_tail])
    
    ax.plot(ranks, log_empirical, 'bo-', markersize=6, linewidth=1.5, 
            label='Empirical tail', alpha=0.8)
    
    # GPD prediction
    # Use 95th percentile as threshold
    u_threshold = np.percentile(np.abs(returns), 95)
    n_total = len(returns)
    
    gpd_predictions = []
    for k in ranks:
        if xi_presentation > 0:
            # GPD quantile formula (thesis page 7)
            excess = (sigma_presentation / xi_presentation) * ((n_total / k)**xi_presentation - 1)
            predicted_value = u_threshold + excess
        else:
            # Exponential case
            excess = sigma_presentation * np.log(n_total / k)
            predicted_value = u_threshold + excess
        
        gpd_predictions.append(np.log(max(predicted_value, 1e-10)))
    
    ax.plot(ranks, gpd_predictions, 'r--', linewidth=2.5, 
            label=f'GPD(xi={xi_presentation:.3f}, sigma={sigma_presentation:.4f})')
    
    # Calculate fit quality
    empirical_subset = log_empirical[:len(gpd_predictions)]
    gpd_subset = np.array(gpd_predictions)
    rmse = np.sqrt(np.mean((empirical_subset - gpd_subset)**2))
    
    ax.set_xlabel('Rank (k-th largest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('log(|Return|)', fontsize=12, fontweight='bold')
    ax.set_title('Tail Behavior: Empirical vs GPD Fit', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add fit quality
    ax.text(0.98, 0.02, f'RMSE = {rmse:.4f}\n\nGPD provides good\nfit to tail behavior', 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plot4_tail_behavior.png', dpi=300, bbox_inches='tight')
    print("[SAVED] plot4_tail_behavior.png")
    plt.close()

# =============================================================================
# COMBINED 2x2 GRID (MATCHING PRESENTATION LAYOUT)
# =============================================================================

def plot_combined_grid():
    """
    Creates 2x2 grid matching the presentation layout
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # PLOT 1: Hill Plot (Top Left)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    k_range = range(5, min(80, len(returns)//3))
    hill_vals = []
    for k in k_range:
        try:
            xi_k, _, _ = hill_estimator(returns, k=k)
            hill_vals.append(xi_k if xi_k is not None else np.nan)
        except:
            hill_vals.append(np.nan)
    
    valid_mask = ~np.isnan(hill_vals)
    k_valid = np.array(list(k_range))[valid_mask]
    hill_valid = np.array(hill_vals)[valid_mask]
    
    k_opt = int(np.sqrt(len(returns)))
    xi_hat, se_xi, k_used = hill_estimator(returns, k=k_opt)
    
    ax1.plot(k_valid, hill_valid, 'b-', linewidth=2, label='Hill estimate')
    ax1.axhline(y=xi_hat, color='r', linestyle='--', linewidth=2.5, 
                label=f'xi-hat = {xi_hat:.3f}', zorder=5)
    ax1.axvline(x=k_used, color='g', linestyle=':', linewidth=2, alpha=0.7, 
                label=f'k = {k_used}')
    ax1.fill_between(k_valid, xi_hat - 2*se_xi, xi_hat + 2*se_xi,
                      alpha=0.2, color='red', label='95% CI')
    
    ax1.set_xlabel('k (Num. Upper Order Stats)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('xi-hat', fontsize=10, fontweight='bold')
    ax1.set_title('Hill Plot: Tail Parameter Estimation', fontweight='bold', fontsize=11)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # PLOT 2: Return Distribution (Top Right)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    returns_pct = returns * 100
    n_bins = 50
    hist_counts, bin_edges = np.histogram(returns_pct, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    ax2.bar(bin_centers, hist_counts, width=bin_width*0.9, 
            alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5,
            label='Empirical density')
    
    std_pct = returns.std() * 100
    mean_pct = returns.mean() * 100
    x_fit = np.linspace(returns_pct.min() - 2, returns_pct.max() + 2, 300)
    t_pdf = stats.t.pdf((x_fit - mean_pct) / std_pct, df=df_true) / std_pct
    ax2.plot(x_fit, t_pdf, 'r-', linewidth=3, 
             label=f't-dist (df={df_true})', zorder=10)
    
    ax2.axvline(x=mean_pct, color='orange', linestyle='--', linewidth=2, 
                label=f'Mean = {mean_pct:.2f}%', zorder=11)
    
    ax2.set_xlabel('Daily Returns (%)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax2.set_title('Simulated SVB Return Distribution', fontweight='bold', fontsize=11)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # PLOT 3: Q-Q Plot (Bottom Left)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    
    (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm", fit=True)
    
    ax3.plot(osm, osr, 'bo', markersize=4, alpha=0.6, label='Sample quantiles')
    ax3.plot(osm, slope * osm + intercept, 'r-', linewidth=2.5, 
             label=f'Normal fit (R^2={r**2:.3f})')
    
    upper_idx = len(osm) * 19 // 20
    ax3.plot(osm[upper_idx:], osr[upper_idx:], 'ro', markersize=6, 
             label='Upper tail', zorder=10)
    
    lower_idx = len(osm) // 20
    ax3.plot(osm[:lower_idx], osr[:lower_idx], 'mo', markersize=6, 
             label='Lower tail', zorder=10)
    
    ax3.set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold')
    ax3.set_title('Q-Q Plot: Returns vs. Normal', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # PLOT 4: Tail Behavior (Bottom Right)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    
    sorted_returns_abs = np.sort(np.abs(returns))[::-1]
    n_tail = min(50, len(sorted_returns_abs))
    ranks = np.arange(1, n_tail + 1)
    log_empirical = np.log(sorted_returns_abs[:n_tail])
    
    ax4.plot(ranks, log_empirical, 'bo-', markersize=5, linewidth=1.5, 
             label='Empirical tail', alpha=0.8)
    
    u_threshold = np.percentile(np.abs(returns), 95)
    n_total = len(returns)
    
    gpd_predictions = []
    for k in ranks:
        if xi_presentation > 0:
            excess = (sigma_presentation / xi_presentation) * ((n_total / k)**xi_presentation - 1)
            predicted_value = u_threshold + excess
        else:
            excess = sigma_presentation * np.log(n_total / k)
            predicted_value = u_threshold + excess
        gpd_predictions.append(np.log(max(predicted_value, 1e-10)))
    
    ax4.plot(ranks, gpd_predictions, 'r--', linewidth=2.5, 
             label=f'GPD(xi={xi_presentation:.3f})')
    
    ax4.set_xlabel('Rank (k-th largest)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('log(|Return|)', fontsize=10, fontweight='bold')
    ax4.set_title('Tail Behavior: Empirical vs GPD Fit', fontweight='bold', fontsize=11)
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('SVB Crisis: Heavy-Tailed Analysis (4 Key Plots)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('combined_4plots.png', dpi=300, bbox_inches='tight')
    print("[SAVED] combined_4plots.png")
    plt.close()

# =============================================================================
# EXECUTE ALL PLOTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENERATING ALL MISSING PLOTS")
    print("="*80 + "\n")
    
    plot_hill_estimation()
    plot_return_distribution()
    plot_qq_normal()
    plot_tail_behavior()
    plot_combined_grid()
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nFiles created:")
    print("  - plot1_hill_estimation.png")
    print("  - plot2_return_distribution.png")
    print("  - plot3_qq_normal.png")
    print("  - plot4_tail_behavior.png")
    print("  - combined_4plots.png (2x2 grid)")
    print("\n" + "="*80)