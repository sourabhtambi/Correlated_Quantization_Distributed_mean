import numpy as np
import matplotlib.pyplot as plt


def generate_data(n, d, sigma_md, rng):
    """
    Generate data : mu(j) + U_ij.
    NO clipping — the quantizer adapts its range to the data.
    """
    mu = rng.uniform(0, 1, size=d)
    noise = rng.uniform(-4 * sigma_md, 4 * sigma_md, size=(n, d))
    data = mu[None, :] + noise
    return data


def independent_quantization(data, rng, l, r):
    """
    Standard independent stochastic quantization
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


import numpy as np
import matplotlib.pyplot as plt

def generate_data(n, d, sigma_md, rng):
    mu = rng.uniform(0, 1, size=d)
    noise = rng.uniform(-4 * sigma_md, 4 * sigma_md, size=(n, d))
    data = mu[None, :] + noise
    return data


# ================================================================
# Independent stochastic quantization 
# ================================================================

def independent_quantization(data, rng, l, r):
    """1-bit independent stochastic quantization."""
    n, d = data.shape
    ranges = r - l
    ys = (data - l[None, :]) / ranges[None, :]
    ys = np.clip(ys, 0, 1)

    U = rng.uniform(size=(n, d))
    Q = (U < ys).astype(float)
    return l[None, :] + ranges[None, :] * Q


def independent_quantization_klevel(data, k, rng, l, r):
    """
    k-level independent stochastic quantization.
    Divides [0,1] into (k-1) equal intervals; each value is
    stochastically rounded to the nearest endpoint.
    """
    if k == 2:
        return independent_quantization(data, rng, l, r)

    n, d = data.shape
    ranges = r - l
    ys = (data - l[None, :]) / ranges[None, :]
    ys = np.clip(ys, 0, 1)

    scaled = ys * (k - 1)
    lower_idx = np.floor(scaled).astype(int)
    lower_idx = np.clip(lower_idx, 0, k - 2)

    lower_val = lower_idx / (k - 1)
    upper_val = (lower_idx + 1) / (k - 1)

    frac = scaled - lower_idx
    frac = np.clip(frac, 0, 1)

    U = rng.uniform(size=(n, d))
    Q_norm = np.where(U < frac, upper_val, lower_val)
    return l[None, :] + ranges[None, :] * Q_norm

# ================================================================
# Correlated quantization — Algorithm 1 (1-bit) & Algorithm 2 (k-level)
# ================================================================

def correlated_quantization(data, rng, l, r):
    """OneDimOneBitCQ (Algorithm 1)."""
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


def correlated_quantization_klevel(data, k, rng, l, r):
    """
    OneDimKLevelsCQ (Algorithm 2).
    For k >= 3: randomized interval levels, correlated 1-bit
    quantization within each interval.
    For k == 2: falls back to Algorithm 1.
    """
    if k == 2:
        return correlated_quantization(data, rng, l, r)

    n, d = data.shape
    ranges = r - l
    ys = (data - l[None, :]) / ranges[None, :]
    ys = np.clip(ys, 0, 1)

    beta = (k + 1) / (k * (k - 1))

    Q_norm = np.zeros_like(ys)
    for j in range(d):
        # Randomized levels (fresh per dimension)
        c1 = rng.uniform(-1.0 / k, 0)
        levels = np.array([c1 + m * beta for m in range(k)])

        # c'_i = largest level <= y_i
        level_idx = np.searchsorted(levels, ys[:, j], side='right') - 1
        level_idx = np.clip(level_idx, 0, k - 1)
        c_prime = levels[level_idx]

        # Normalize within interval
        z = (ys[:, j] - c_prime) / beta
        z = np.clip(z, 0, 1)

        # Correlated 1-bit quantization (shared permutation)
        pi = rng.permutation(n)
        gamma = rng.uniform(0, 1.0 / n, size=n)
        U = pi / n + gamma
        Q_bit = (U < z).astype(float)

        Q_norm[:, j] = c_prime + beta * Q_bit

    return l[None, :] + ranges[None, :] * Q_norm

# ===============================================================
# Effect of random Rotation — Algorithm 3 (WalshHadamardCQ)
# ===============================================================

#   "We let mu = (1.0, -1.0, 0.0, ..., 0.0), k=2, sigma_md=0.01."

from scipy.linalg import hadamard

def generate_data_2d(n, d, sigma_md, rng):
    """
    Figure 2(d) uses FIXED sparse mu, not random.
    mu = (1.0, -1.0, 0.0, ..., 0.0).
    """
    mu = np.zeros(d)
    mu[0] = 1.0
    mu[1] = -1.0
    noise = rng.uniform(-4 * sigma_md, 4 * sigma_md, size=(n, d))
    return mu[None, :] + noise

def walsh_hadamard_quantize(data, rng, R, quantize_fn):
    """
    Algorithm 3: WalshHadamardCQ wrapper.
    Applies Walsh-Hadamard rotation, then per-coordinate quantization
    with range [-1, 1], then inverse rotation.

    Works with either independent_quantization or correlated_quantization
    as quantize_fn.
    """

    n, d = data.shape

    # W = (1/sqrt(d)) * H * D
    H = hadamard(d)                          # d×d Hadamard
    D_diag = rng.choice([-1, 1], size=d)     # Rademacher entries
    
    # Efficient: W @ x = (1/sqrt(d)) * H @ (D * x)
    # Instead of building full W, use broadcast: H * D_diag[None, :]
   
    W = (1.0 / np.sqrt(d)) * H * D_diag[None, :]  # (d, d)

    # Scale: sqrt(d) / (R * sqrt(8 * log(d*n)))
    scale = np.sqrt(d) / (R * np.sqrt(8 * np.log(d * n)))

    # Rotate and scale each client: y_i = scale * W @ x_i
    # data is (n, d), W is (d, d) → rotated = data @ W^T * scale
    
    rotated = (data @ W.T) * scale   # (n, d)

    # Clip to [-1, 1]
    rotated = np.clip(rotated, -1, 1)

    # Quantize per-coordinate with range [-1, 1]
    l_rot = np.full(d, -1.0)
    r_rot = np.full(d, 1.0)
    Q_rot = quantize_fn(rotated, rng, l_rot, r_rot)  # (n, d)

    # Server aggregates
    z = np.mean(Q_rot, axis=0)  # (d,)

    # Inverse rotation and rescale
    # W is orthogonal → W^{-1} = W^T
    inv_scale = R * np.sqrt(8 * np.log(d * n)) / np.sqrt(d)
    est_mean = inv_scale * (W.T @ z)

    return est_mean
# ================================================================
# Experiment
# ================================================================

def run_experiment_2a(n, d, sigma_md, n_runs, n_trials_per_run):
    """Figure 2(a): vary sigma_md, k=2 (1-bit)."""
    ind_rmses = []
    cor_rmses = []

    for run in range(n_runs):
        rng_data = np.random.default_rng(seed=run * 37)
        data = generate_data(n, d, sigma_md, rng_data)
        true_mean = np.mean(data, axis=0)

        # Theoretical range covering all data
        global_l = -4 * sigma_md
        global_r = 1 + 4 * sigma_md
        l = np.full(d, global_l)
        r = np.full(d, global_r)

        sq_errors_ind = []
        rng_q = np.random.default_rng(seed=run * 37 + 1000)
        for _ in range(n_trials_per_run):
            Q = independent_quantization(data, rng_q, l, r)
            est = np.mean(Q, axis=0)
            sq_errors_ind.append(np.sum((est - true_mean) ** 2))
        ind_rmses.append(np.sqrt(np.mean(sq_errors_ind)))

        sq_errors_cor = []
        rng_q = np.random.default_rng(seed=run * 37 + 2000)
        for _ in range(n_trials_per_run):
            Q = correlated_quantization(data, rng_q, l, r)
            est = np.mean(Q, axis=0)
            sq_errors_cor.append(np.sum((est - true_mean) ** 2))
        cor_rmses.append(np.sqrt(np.mean(sq_errors_cor)))

    return np.mean(ind_rmses), np.mean(cor_rmses)


def run_experiment_2b(n, d, sigma_md, k, n_runs, n_trials_per_run):
    """Figure 2(b): vary k, fixed sigma_md."""
    ind_rmses = []
    cor_rmses = []

    for run in range(n_runs):
        rng_data = np.random.default_rng(seed=run * 37)
        data = generate_data(n, d, sigma_md, rng_data)
        true_mean = np.mean(data, axis=0)

        global_l = -4 * sigma_md
        global_r = 1 + 4 * sigma_md
        l = np.full(d, global_l)
        r = np.full(d, global_r)

        sq_errors_ind = []
        rng_q = np.random.default_rng(seed=run * 37 + 1000)
        for _ in range(n_trials_per_run):
            Q = independent_quantization_klevel(data, k, rng_q, l, r)
            est = np.mean(Q, axis=0)
            sq_errors_ind.append(np.sum((est - true_mean) ** 2))
        ind_rmses.append(np.sqrt(np.mean(sq_errors_ind)))

        sq_errors_cor = []
        rng_q = np.random.default_rng(seed=run * 37 + 2000)
        for _ in range(n_trials_per_run):
            Q = correlated_quantization_klevel(data, k, rng_q, l, r)
            est = np.mean(Q, axis=0)
            sq_errors_cor.append(np.sum((est - true_mean) ** 2))
        cor_rmses.append(np.sqrt(np.mean(sq_errors_cor)))

    return np.mean(ind_rmses), np.mean(cor_rmses)


def run_experiment_2c(n, d, sigma_md, n_runs, n_trials_per_run):
    """Figure 2(c): vary n, fixed sigma_md=0.01 and k=2."""
    ind_rmses = []
    cor_rmses = []

    for run in range(n_runs):
        rng_data = np.random.default_rng(seed=run * 37)
        data = generate_data(n, d, sigma_md, rng_data)
        true_mean = np.mean(data, axis=0)

        global_l = -4 * sigma_md
        global_r = 1 + 4 * sigma_md
        l = np.full(d, global_l)
        r = np.full(d, global_r)

        sq_errors_ind = []
        rng_q = np.random.default_rng(seed=run * 37 + 1000)
        for _ in range(n_trials_per_run):
            Q = independent_quantization(data, rng_q, l, r)
            est = np.mean(Q, axis=0)
            sq_errors_ind.append(np.sum((est - true_mean) ** 2))
        ind_rmses.append(np.sqrt(np.mean(sq_errors_ind)))

        sq_errors_cor = []
        rng_q = np.random.default_rng(seed=run * 37 + 2000)
        for _ in range(n_trials_per_run):
            Q = correlated_quantization(data, rng_q, l, r)
            est = np.mean(Q, axis=0)
            sq_errors_cor.append(np.sum((est - true_mean) ** 2))
        cor_rmses.append(np.sqrt(np.mean(sq_errors_cor)))

    return np.mean(ind_rmses), np.mean(cor_rmses)


def run_experiment_2d(n, d, sigma_md, quantize_fn, use_rotation,
                      n_runs, n_trials_per_run):
    """
    Figure 2(d): Effect of random rotation.
    """
    rmses = []

    for run in range(n_runs):
        rng_data = np.random.default_rng(seed=run * 37)
        data = generate_data_2d(n, d, sigma_md, rng_data)
        true_mean = np.mean(data, axis=0)

        # R = max L2 norm (known bound for B^d(R))
        R = np.max(np.linalg.norm(data, axis=1))

        sq_errors = []
        rng_q = np.random.default_rng(seed=run * 37 + 3000)

        for _ in range(n_trials_per_run):
            if use_rotation:
                # Algorithm 3: Walsh-Hadamard rotation
                est = walsh_hadamard_quantize(data, rng_q, R, quantize_fn)
            else:
                # No rotation: per-coordinate quantization
                # Range = [-R, R] (L2 ball, standard without rotation)
                l = np.full(d, -R)
                r = np.full(d, R)
                Q = quantize_fn(data, rng_q, l, r)
                est = np.mean(Q, axis=0)
            sq_errors.append(np.sum((est - true_mean) ** 2))
        rmses.append(np.sqrt(np.mean(sq_errors)))

    return np.mean(rmses)


# n = 100, d = 1024
# "averaged over ten runs"

def main():
    n = 100       # number of clients
    d = 1024      # dimension
    n_runs = 10   # "averaged over ten runs for statistical consistency"
    n_trials_per_run = 50  # quantization trials per dataset for stable RMSE

    # ========================
    # Figure 2(a): 
    # ========================
    sigma_md_values = [0.01, 0.02, 0.04, 0.08, 0.16]

    rmse_ind_a = []
    rmse_cor_a = []

    print("=" * 55)
    print("Figure 2(a): RMSE vs sigma_md  (k=2, n=100, d=1024)")
    print("=" * 55)
    for sigma_md in sigma_md_values:
        print(f"  sigma_md = {sigma_md} ...", end="", flush=True)
        ri, rc = run_experiment_2a(n, d, sigma_md, n_runs, n_trials_per_run)
        rmse_ind_a.append(ri)
        rmse_cor_a.append(rc)
        print(f"  Ind={ri:.4f}  Cor={rc:.4f}")

    # Plot 2(a)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x_pos = np.arange(len(sigma_md_values))
    ax.plot(x_pos, rmse_ind_a, '-+', color='tab:blue', label='Independent',
            markersize=10, linewidth=1.5, markeredgewidth=1.5)
    ax.plot(x_pos, rmse_cor_a, '-+', color='tab:orange', label='Correlated',
            markersize=10, linewidth=1.5, markeredgewidth=1.5)
    ax.set_xlabel(r'$\sigma_{md}$', fontsize=13)
    ax.set_ylabel('RMSE', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in sigma_md_values])
    ax.set_ylim(0, max(rmse_ind_a) * 1.1)
    ax.set_yticks(np.arange(0, max(rmse_ind_a) * 1.1, 0.5))
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title(r'(a) RMSE as a function of $\sigma_{md}$', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure2a_rmse_vs_sigma.png', dpi=150, bbox_inches='tight')
    print("  -> Saved figure2a_rmse_vs_sigma.png\n")
    plt.close()

    # ========================
    # Figure 2(b): 
    # ========================
    sigma_md_b = 0.01
    k_values = [2, 3, 4, 6, 8, 16]

    rmse_ind_b = []
    rmse_cor_b = []

    print("=" * 55)
    print(f"Figure 2(b): log2(RMSE) vs k  (sigma_md={sigma_md_b}, n=100)")
    print("=" * 55)
    for k in k_values:
        print(f"  k = {k} ...", end="", flush=True)
        ri, rc = run_experiment_2b(n, d, sigma_md_b, k, n_runs, n_trials_per_run)
        rmse_ind_b.append(ri)
        rmse_cor_b.append(rc)
        print(f"  Ind={ri:.4f} (log2={np.log2(ri):.2f})  "
              f"Cor={rc:.4f} (log2={np.log2(rc):.2f})")

    # Plot 2(b)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    log2_ind = np.log2(rmse_ind_b)
    log2_cor = np.log2(rmse_cor_b)
    ax.plot(k_values, log2_ind, '-+', color='tab:blue', label='Independent',
            markersize=10, linewidth=1.5, markeredgewidth=1.5)
    ax.plot(k_values, log2_cor, '-+', color='tab:orange', label='Correlated',
            markersize=10, linewidth=1.5, markeredgewidth=1.5)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('k', fontsize=13)
    ax.set_ylabel(r'$\log_2$(RMSE)', fontsize=13)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(v) for v in k_values])
    ax.set_ylim(-5, 1)
    ax.set_yticks([-4, -3, -2, -1, 0])
    ax.legend(fontsize=11, loc='upper right')
    ax.set_title(r'(b) RMSE as a function of $k$', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure2b_rmse_vs_k.png', dpi=150, bbox_inches='tight')
    print("  -> Saved figure2b_rmse_vs_k.png\n")
    plt.close()

    # ========================
    # Figure 2(c):
    # ========================

    sigma_md_c = 0.01
    n_values = [10, 20, 40, 80, 160]

    rmse_ind_c = []
    rmse_cor_c = []

    print("=" * 55)
    print(f"Figure 2(c): log2(RMSE) vs n  (sigma_md={sigma_md_c}, k=2)")
    print("=" * 55)
    for n_val in n_values:
        print(f"  n = {n_val} ...", end="", flush=True)
        ri, rc = run_experiment_2c(n_val, d, sigma_md_c, n_runs, n_trials_per_run)
        rmse_ind_c.append(ri)
        rmse_cor_c.append(rc)
        print(f"  Ind={ri:.4f} (log2={np.log2(ri):.2f})  "
              f"Cor={rc:.4f} (log2={np.log2(rc):.2f})")

    # Plot 2(c)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    log2_ind_c = np.log2(rmse_ind_c)
    log2_cor_c = np.log2(rmse_cor_c)
    ax.plot(n_values, log2_ind_c, '-+', color='tab:blue', label='Independent',
            markersize=10, linewidth=1.5, markeredgewidth=1.5)
    ax.plot(n_values, log2_cor_c, '-+', color='tab:orange', label='Correlated',
            markersize=10, linewidth=1.5, markeredgewidth=1.5)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('n', fontsize=13)
    ax.set_ylabel(r'$\log_2$(RMSE)', fontsize=13)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(v) for v in n_values])
    ax.set_ylim(-3, 3)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.legend(fontsize=11, loc='upper right')
    ax.set_title(r'(c) RMSE as a function of $n$', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure2c_rmse_vs_n.png', dpi=150, bbox_inches='tight')
    print("  -> Saved figure2c_rmse_vs_n.png")
    plt.close()

    # ========================
    # Figure 2(d): random rotation, vary n
    # ========================

    sigma_md_d = 0.01
    n_values_d = [10, 20, 40, 80, 160]
    n_trials_2d = 20  # fewer trials (rotation is expensive)

    configs_2d = {
        'Independent':          (independent_quantization, False),
        'Independent+Rotation': (independent_quantization, True),
        'Correlated':           (correlated_quantization,  False),
        'Correlated+Rotation':  (correlated_quantization,  True),
    }
    results_2d = {name: [] for name in configs_2d}

    print("\n" + "=" * 55)
    print(f"Figure 2(d): RMSE with random rotation")
    print(f"  d={d}, sigma_md={sigma_md_d}, k=2")
    print(f"  mu = (1.0, -1.0, 0.0, ..., 0.0)")
    print("=" * 55)

    for n_val in n_values_d:
        print(f"\n  n = {n_val}:")
        for name, (qfn, rot) in configs_2d.items():
            rmse = run_experiment_2d(
                n_val, d, sigma_md_d, qfn, rot, n_runs, n_trials_2d
            )
            results_2d[name].append(rmse)
            print(f"    {name:25s}  RMSE={rmse:.4f}  "
                  f"log2={np.log2(rmse):.2f}")

    # Plot 2(d)
    fig, ax = plt.subplots(figsize=(5.5, 4))

    colors_2d = {
        'Independent':          'tab:blue',
        'Independent+Rotation': 'tab:blue',
        'Correlated':           'tab:orange',
        'Correlated+Rotation':  'tab:orange',
    }
    styles_2d = {
        'Independent':          '-+',
        'Independent+Rotation': '--o',
        'Correlated':           '-+',
        'Correlated+Rotation':  '--o',
    }

    for name in configs_2d:
        log2_rmse = np.log2(results_2d[name])
        ax.plot(n_values_d, log2_rmse, styles_2d[name],
                color=colors_2d[name], label=name,
                markersize=8, linewidth=1.5, markeredgewidth=1.2)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('n', fontsize=13)
    ax.set_ylabel(r'$\log_2$(RMSE)', fontsize=13)
    ax.set_xticks(n_values_d)
    ax.set_xticklabels([str(v) for v in n_values_d])
    ax.set_ylim(-3, 5)
    ax.set_yticks([-2, 0, 2, 4])
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(r'(d) RMSE with random rotation', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure2d_random_rotation.png', dpi=150, bbox_inches='tight')
    print("\n  -> Saved figure2d_random_rotation.png")
    plt.close()


if __name__ == "__main__":
    main()
