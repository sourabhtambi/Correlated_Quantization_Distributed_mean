# -*- coding: utf-8 -*-
"""
Reproduction of Figure 2(a) from:
"Correlated quantization for distributed mean estimation and optimization"
(Suresh et al., 2022)

Key details from Section 8 (Experiments), page 13:
- n = 100 clients, d = 1024 dimensions, k = 2 (1-bit quantization)
- Each coordinate j sampled as: mu(j) + U, where
    mu(j) ~ Uniform[0,1] (fixed across clients)
    U ~ Uniform[-4*sigma_md, 4*sigma_md] (independent per client per coord)
- Vary sigma_md in {0.01, 0.02, 0.04, 0.08, 0.16}
- Averaged over 10 runs for statistical consistency

Algorithm 1 (OneDimOneBitCQ):
  - Input: x1..xn, l, r  (the range covering all data)
  - y_i = (x_i - l) / (r - l)
  - U_i = pi_i/n + gamma_i (correlated via shared permutation)
  - Q_i = l + (r-l) * 1{U_i < y_i}

The quantizer range [l, r] per coordinate is set to [min, max] of that
coordinate's data. As sigma_md grows, this range widens, increasing
independent quantization error (which scales as (r-l)^2). Correlated
quantization error scales as sigma_md*(r-l), growing more slowly.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n, d, sigma_md, rng):
    """
    Generate data per Section 8: mu(j) + U_ij.
    NO clipping — the quantizer adapts its range to the data.
    """
    mu = rng.uniform(0, 1, size=d)
    noise = rng.uniform(-4 * sigma_md, 4 * sigma_md, size=(n, d))
    data = mu[None, :] + noise
    return data


def independent_quantization(data, rng, l, r):
    """
    Standard independent stochastic quantization (Eq. 3).
    Per-coordinate range [l_j, r_j].
    """
    n, d = data.shape
    ranges = r - l  # shape (d,)
    ys = (data - l[None, :]) / ranges[None, :]
    ys = np.clip(ys, 0, 1)

    U = rng.uniform(size=(n, d))
    Q = (U < ys).astype(float)

    return l[None, :] + ranges[None, :] * Q


def correlated_quantization(data, rng, l, r):
    """
    OneDimOneBitCQ (Algorithm 1).
    Shared permutation per dimension correlates the uniforms.
    Per-coordinate range [l_j, r_j].
    """
    n, d = data.shape
    ranges = r - l
    ys = (data - l[None, :]) / ranges[None, :]
    ys = np.clip(ys, 0, 1)

    Q = np.zeros_like(data)
    for j in range(d):
        pi = rng.permutation(n)
        gamma = rng.uniform(0, 1.0 / n, size=n)
        U = pi / n + gamma
        Q[:, j] = (U < ys[:, j]).astype(float)

    return l[None, :] + ranges[None, :] * Q


def run_experiment(n, d, sigma_md, n_runs, n_trials_per_run):
    """
    For each run:
      1. Generate a fresh dataset
      2. Compute per-coordinate range [l_j, r_j] = [min_j, max_j]
      3. Run both quantizers n_trials_per_run times
      4. Compute RMSE = sqrt(mean MSE)
    Average RMSE over n_runs.
    """
    ind_rmses = []
    cor_rmses = []

    for run in range(n_runs):
        rng_data = np.random.default_rng(seed=run * 37)
        data = generate_data(n, d, sigma_md, rng_data)
        true_mean = np.mean(data, axis=0)

        # Known theoretical range: mu(j) in [0,1], noise in [-4s, 4s]
        # So data lies in [-4*sigma_md, 1 + 4*sigma_md]
        global_l = -4 * sigma_md
        global_r = 1 + 4 * sigma_md
        l = np.full(d, global_l)
        r = np.full(d, global_r)

        # Independent quantization
        sq_errors_ind = []
        rng_q = np.random.default_rng(seed=run * 37 + 1000)
        for _ in range(n_trials_per_run):
            Q = independent_quantization(data, rng_q, l, r)
            est = np.mean(Q, axis=0)
            sq_errors_ind.append(np.sum((est - true_mean) ** 2))
        ind_rmses.append(np.sqrt(np.mean(sq_errors_ind)))

        # Correlated quantization
        sq_errors_cor = []
        rng_q = np.random.default_rng(seed=run * 37 + 2000)
        for _ in range(n_trials_per_run):
            Q = correlated_quantization(data, rng_q, l, r)
            est = np.mean(Q, axis=0)
            sq_errors_cor.append(np.sum((est - true_mean) ** 2))
        cor_rmses.append(np.sqrt(np.mean(sq_errors_cor)))

    return np.mean(ind_rmses), np.mean(cor_rmses)


def main():
    n = 100
    d = 1024
    sigma_md_values = [0.01, 0.02, 0.04, 0.08, 0.16]
    n_runs = 10
    n_trials_per_run = 50

    rmse_independent = []
    rmse_correlated = []

    for sigma_md in sigma_md_values:
        print(f"Running sigma_md = {sigma_md}...")
        rmse_ind, rmse_cor = run_experiment(
            n, d, sigma_md, n_runs, n_trials_per_run
        )
        rmse_independent.append(rmse_ind)
        rmse_correlated.append(rmse_cor)
        print(f"  Independent RMSE = {rmse_ind:.4f}")
        print(f"  Correlated  RMSE = {rmse_cor:.4f}")

    # ----- Plot Figure 2(a) -----
    fig, ax = plt.subplots(figsize=(5.5, 4))

    x_pos = np.arange(len(sigma_md_values))
    ax.plot(x_pos, rmse_independent, '-+', color='tab:blue', label='Independent',
            markersize=10, linewidth=1.5, markeredgewidth=1.5)
    ax.plot(x_pos, rmse_correlated, '-+', color='tab:orange', label='Correlated',
            markersize=10, linewidth=1.5, markeredgewidth=1.5)

    ax.set_xlabel(r'$\sigma_{md}$', fontsize=13)
    ax.set_ylabel('RMSE', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in sigma_md_values])
    ax.set_ylim(0, max(rmse_independent) * 1.1)
    ax.set_yticks(np.arange(0, max(rmse_independent) * 1.1, 0.5))
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title(r'(a) RMSE as a function of $\sigma_{md}$', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure2a_rmse_vs_sigma.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to figure2a_rmse_vs_sigma.png")
    plt.close()


if __name__ == "__main__":
    main()
