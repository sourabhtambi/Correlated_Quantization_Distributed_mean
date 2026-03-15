import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt

# ==================================================
# ALGORITHM 1 : OneDimOneBitCQ | 
# ==================================================
def one_dim_one_bit_cq(x, l, r):
    
    n = len(x)

    # Step 1 : normalise to [0, 1)
    y = (x - l) / (r - l)

    # Step 2 : correlated U_i = pi_i/n + gamma_i
    pi    = np.random.permutation(n)           # random permutation of {0,...,n-1}
    gamma = np.random.uniform(0, 1/n, size=n)  # gamma_i ~ U[0, 1/n)
    U     = pi / n + gamma

    # Step 3 : Q_i(x_i) = (r-l) * 1[U_i < y_i]
    Q = (r - l) * (U < y).astype(float)

    # Output : shift back by l becasuse of normalising step
    return l + Q.mean()


# ==============================================
# ALGORITHM 2 : OneDimKLevelsCQ
# ==============================================

def one_dim_k_levels_cq(x, l, r, k):
    """
    Algorithm 2: One-dimensional k-level correlated quantization.

    Input : x  – array of n client values, all in [l, r)
            l  – lower bound
            r  – upper bound
            k  – number of quantization levels (k >= 3)
    Output: unbiased scalar estimate of mean(x)
    """
    assert k >= 3, "k must be >= 3"
    n   = len(x)
    rng = r - l

    # randomised levels in [0,1] space
    beta   = (k + 1) / (k * (k - 1))
    c1     = np.random.uniform(-1/k, 0)
    levels = c1 + np.arange(k) * beta         # c_1, ..., c_k

    # Step 1 : normalise x to [0,1) — NOT divided by beta
    y_norm = (x - l) / rng

    # Step 2 : c'_i = largest level strictly below y_norm_i
    c_prime = np.array([
        levels[levels < yi].max() if np.any(levels < yi) else levels[0]
        for yi in y_norm
    ])

    # sub-interval position: z_i = (y_norm_i - c'_i) / beta  in [0,1)
    y_sub = (y_norm - c_prime) / beta

    # Step 3 : correlated U_i
    pi    = np.random.permutation(n)
    gamma = np.random.uniform(0, 1/n, size=n)
    U     = pi / n + gamma

    # Step 4 : Q_i = c'_i + beta * 1[U_i < y_sub_i]  (in [0,1] space)
    Q_norm = c_prime + beta * (U < y_sub).astype(float)

    # Output : map back to [l, r]
    return l + rng * Q_norm.mean()


# ===============================================
# ALGORITHM 3 : WalshHadamardCQ
# ===============================================
def walsh_hadamard_cq(X, R, k=2):
    
    n, d = X.shape

    # pad d to the next power of 2
    d_pad = 1
    while d_pad < d:
        d_pad *= 2

    # Building W = (1/sqrt(d_pad)) * H * D
    H    = hadamard(d_pad).astype(float)
    diag = np.random.choice([-1.0, 1.0], size=d_pad)

    def apply_W(v):
        return (H @ (diag * v)) / np.sqrt(d_pad)

    def apply_W_inv(v):
        return diag * (H @ v) / np.sqrt(d_pad)

    # scaling
    scale = np.sqrt(d_pad) / (R * np.sqrt(8.0 * np.log(float(d) * n)))

    # Step 1 : rotate + scale each client vector
    X_pad        = np.zeros((n, d_pad))
    X_pad[:, :d] = X
    Y = np.array([apply_W(X_pad[i]) * scale for i in range(n)])  # (n, d_pad)

    # Step 2 : clip to [-1, 1]
    Y_clip = np.clip(Y, -1.0, 1.0)

    # Step 3 : quantise each coordinate with Algo 1 or 2
    z = np.zeros(d_pad)
    for j in range(d_pad):
        col = Y_clip[:, j]
        if k == 2:
            z[j] = one_dim_one_bit_cq(col, l=-1.0, r=1.0)
        else:
            z[j] = one_dim_k_levels_cq(col, l=-1.0, r=1.0, k=k)

    # Output : invert scale and rotation
    x_hat_pad = apply_W_inv(z) / scale
    return x_hat_pad[:d]


# ======================================================================
#  INDEPENDENT (STANDARD) BASELINES  – matching the paper's baselines
# ======================================================================

def independent_one_bit(x, l, r):
   
    n = len(x)
    y = (x - l) / (r - l)                     # normalise to [0,1)
    U = np.random.uniform(0, 1, size=n)        # independent, NOT permutation-coupled
    Q = (r - l) * (U < y).astype(float)
    return l + Q.mean()


def independent_k_levels(x, l, r, k):
   
    assert k >= 3
    n   = len(x)
    rng = r - l
    beta = 1.0 / (k - 1)                      # fixed interval width

    y_norm  = (x - l) / rng                   # in [0,1)
    y_floor = np.floor(y_norm / beta) * beta  # lower level
    y_sub   = (y_norm - y_floor) / beta       # fractional position in sub-interval

    U     = np.random.uniform(0, 1, size=n)   # independent
    Q_norm = y_floor + beta * (U < y_sub).astype(float)
    return l + rng * Q_norm.mean()


def independent_walsh_hadamard(X, R, k=2):
   
    n, d = X.shape
    d_pad = 1
    while d_pad < d:
        d_pad *= 2

    H    = hadamard(d_pad).astype(float)
    diag = np.random.choice([-1.0, 1.0], size=d_pad)

    def apply_W(v):
        return (H @ (diag * v)) / np.sqrt(d_pad)

    def apply_W_inv(v):
        return diag * (H @ v) / np.sqrt(d_pad)

    scale = np.sqrt(d_pad) / (R * np.sqrt(8.0 * np.log(float(d) * n)))

    X_pad        = np.zeros((n, d_pad))
    X_pad[:, :d] = X
    Y = np.array([apply_W(X_pad[i]) * scale for i in range(n)])
    Y_clip = np.clip(Y, -1.0, 1.0)

    z = np.zeros(d_pad)
    for j in range(d_pad):
        col = Y_clip[:, j]
        if k == 2:
            z[j] = independent_one_bit(col, l=-1.0, r=1.0)
        else:
            z[j] = independent_k_levels(col, l=-1.0, r=1.0, k=k)

    x_hat_pad = apply_W_inv(z) / scale
    return x_hat_pad[:d]


#======================================================================
#  COMPARISON EXPERIMENT HELPERS
#======================================================================

def run_scalar_comparison(n_trials=200, n=100, sigma_md=0.1,
                           l=0.0, r=1.0, k=4, seed=42):
    """
    Compare 1-D scalar mean estimation:
      - Independent 1-bit
      - Correlated  1-bit  (Algo 1)
      - Independent k-level
      - Correlated  k-level (Algo 2)

    Returns dict of absolute-error lists keyed by algorithm name.
    """
    np.random.seed(seed)
    true_mean = (l + r) / 2.0
    errors = {
        'Independent 1-bit':   [],
        'Correlated 1-bit':    [],
        f'Independent {k}-level': [],
        f'Correlated {k}-level':  [],
    }
    for _ in range(n_trials):
        x = np.clip(
            np.random.uniform(true_mean - 2*sigma_md, true_mean + 2*sigma_md, n),
            l, r - 1e-9
        )
        errors['Independent 1-bit'].append(
            abs(independent_one_bit(x, l, r) - true_mean))
        errors['Correlated 1-bit'].append(
            abs(one_dim_one_bit_cq(x, l, r) - true_mean))
        errors[f'Independent {k}-level'].append(
            abs(independent_k_levels(x, l, r, k) - true_mean))
        errors[f'Correlated {k}-level'].append(
            abs(one_dim_k_levels_cq(x, l, r, k) - true_mean))
    return errors, true_mean


def run_vary_sigma_md(sigma_md_range, n_trials=50, n=100, d=64,
                      k=2, seed=7):
    """
    Reproduce Figure 2(a): RMSE vs sigma_md.
    Compares Independent vs Correlated (1-bit) on vector mean estimation.
    """
    np.random.seed(seed)
    results = {'Independent': [], 'Correlated': []}
    for sigma_md in sigma_md_range:
        ind_rmse, cor_rmse = [], []
        true_mean = np.random.uniform(0, 1, d)  # fixed mean across trials
        R = float(np.linalg.norm(true_mean)) + 0.5
        for _ in range(n_trials):
            X = true_mean + np.random.uniform(-2*sigma_md, 2*sigma_md, (n, d))
            R_trial = float(np.linalg.norm(X, axis=1).max()) + 0.1

            est_ind = independent_walsh_hadamard(X, R_trial, k=2)
            est_cor = walsh_hadamard_cq(X, R_trial, k=2)

            ind_rmse.append(np.sqrt(np.mean((est_ind - true_mean)**2)))
            cor_rmse.append(np.sqrt(np.mean((est_cor - true_mean)**2)))

        results['Independent'].append((np.mean(ind_rmse), np.std(ind_rmse)))
        results['Correlated'].append((np.mean(cor_rmse), np.std(cor_rmse)))
    return results


def run_vary_n(n_range, n_trials=50, d=64, sigma_md=0.01, k=2, seed=8):
    """
    Reproduce Figure 2(c): log2(RMSE) vs number of clients n.
    """
    results = {'Independent': [], 'Correlated': []}
    true_mean = np.random.RandomState(seed).uniform(0, 1, d)
    for n in n_range:
        ind_rmse, cor_rmse = [], []
        rng = np.random.RandomState(seed + n)
        for _ in range(n_trials):
            X = true_mean + rng.uniform(-2*sigma_md, 2*sigma_md, (n, d))
            R_trial = float(np.linalg.norm(X, axis=1).max()) + 0.1

            est_ind = independent_walsh_hadamard(X, R_trial, k=k)
            est_cor = walsh_hadamard_cq(X, R_trial, k=k)

            ind_rmse.append(np.sqrt(np.mean((est_ind - true_mean)**2)))
            cor_rmse.append(np.sqrt(np.mean((est_cor - true_mean)**2)))

        results['Independent'].append((np.mean(ind_rmse), np.std(ind_rmse)))
        results['Correlated'].append((np.mean(cor_rmse), np.std(cor_rmse)))
    return results


def run_vary_k(k_range, n_trials=50, n=100, d=64,
               sigma_md=0.01, seed=9):
    """
    Reproduce Figure 2(b): log2(RMSE) vs quantization levels k.
    """
    true_mean = np.random.RandomState(seed).uniform(0, 1, d)
    results = {'Independent': [], 'Correlated': []}
    for k in k_range:
        ind_rmse, cor_rmse = [], []
        rng = np.random.RandomState(seed + k)
        for _ in range(n_trials):
            X = true_mean + rng.uniform(-2*sigma_md, 2*sigma_md, (n, d))
            R_trial = float(np.linalg.norm(X, axis=1).max()) + 0.1

            if k == 2:
                est_ind = independent_walsh_hadamard(X, R_trial, k=2)
                est_cor = walsh_hadamard_cq(X, R_trial, k=2)
            else:
                est_ind = independent_walsh_hadamard(X, R_trial, k=k)
                est_cor = walsh_hadamard_cq(X, R_trial, k=k)

            ind_rmse.append(np.sqrt(np.mean((est_ind - true_mean)**2)))
            cor_rmse.append(np.sqrt(np.mean((est_cor - true_mean)**2)))

        results['Independent'].append((np.mean(ind_rmse), np.std(ind_rmse)))
        results['Correlated'].append((np.mean(cor_rmse), np.std(cor_rmse)))
    return results


def run_rotation_comparison(n_range, n_trials=50, d=64,
                             sigma_md=0.01, seed=10):
    """
    Reproduce Figure 2(d): effect of rotation.
    Four curves: Independent, Independent+Rotation,
                 Correlated,  Correlated+Rotation
    """
    true_mean = np.zeros(d)
    true_mean[0] =  1.0   # sparse mean like paper (µ = (1,-1,0,...,0))
    true_mean[1] = -1.0
    results = {
        'Independent':            [],
        'Independent+Rotation':   [],
        'Correlated':             [],
        'Correlated+Rotation':    [],
    }
    for n in n_range:
        rmses = {k: [] for k in results}
        rng = np.random.RandomState(seed + n)
        for _ in range(n_trials):
            X = true_mean + rng.uniform(-2*sigma_md, 2*sigma_md, (n, d))
            R_trial = float(np.linalg.norm(X, axis=1).max()) + 0.1

            # --- without rotation: independent (scalar per coord) ---
            est_ind = np.array([
                independent_one_bit(X[:, j], l=-R_trial, r=R_trial)
                for j in range(d)
            ])
            # --- without rotation: correlated (scalar per coord) ---
            est_cor = np.array([
                one_dim_one_bit_cq(X[:, j], l=-R_trial, r=R_trial)
                for j in range(d)
            ])
            # --- with rotation ---
            est_ind_rot = independent_walsh_hadamard(X, R_trial, k=2)
            est_cor_rot = walsh_hadamard_cq(X, R_trial, k=2)

            rmses['Independent'].append(
                np.sqrt(np.mean((est_ind - true_mean)**2)))
            rmses['Correlated'].append(
                np.sqrt(np.mean((est_cor - true_mean)**2)))
            rmses['Independent+Rotation'].append(
                np.sqrt(np.mean((est_ind_rot - true_mean)**2)))
            rmses['Correlated+Rotation'].append(
                np.sqrt(np.mean((est_cor_rot - true_mean)**2)))

        for name in results:
            results[name].append((np.mean(rmses[name]), np.std(rmses[name])))
    return results


# ==============================================
# TEST + MONTE CARLO + PLOTTING  (original)
# ==============================================
if __name__ == "__main__":
    np.random.seed(42)

    n_trials = 100
    n        = 200
    d        = 16

    err1_list = []
    err2_list = []
    mse3_list = []

    true_mean = 0.4
    true_mean_vec = np.full(d, 0.3)

    for t in range(n_trials):

        # -------- Algo 1 & 2 data --------
        x = np.random.uniform(true_mean - 0.05,
                              true_mean + 0.05,
                              size=n)

        est1 = one_dim_one_bit_cq(x, l=0.0, r=1.0)
        est2 = one_dim_k_levels_cq(x, l=0.0, r=1.0, k=4)

        err1_list.append(abs(est1 - true_mean))
        err2_list.append(abs(est2 - true_mean))

        # -------- Algo 3 data --------
        X = true_mean_vec + np.random.normal(0, 0.02, size=(n, d))
        R = float(np.linalg.norm(X, axis=1).max()) + 0.1

        est3 = walsh_hadamard_cq(X, R=R, k=2)
        mse  = float(np.mean((est3 - true_mean_vec) ** 2))

        mse3_list.append(mse)


    #=========================================
    #  COMPARISON EXPERIMENTS  
    # ========================================

    print("\nRunning scalar comparison (1-D)…")
    scalar_errors, _ = run_scalar_comparison(
        n_trials=300, n=100, sigma_md=0.05, l=0.0, r=1.0, k=4, seed=42)

    print("Running vary-sigma_md experiment…")
    sigma_md_range = [0.01, 0.02, 0.04, 0.08, 0.16]
    vary_sigma  = run_vary_sigma_md(sigma_md_range, n_trials=30, n=100, d=64, seed=7)

    print("Running vary-n experiment…")
    n_range = [10, 20, 40, 80, 160]
    vary_n  = run_vary_n(n_range, n_trials=30, d=64, sigma_md=0.01, seed=8)

    print("Running vary-k experiment…")
    k_range = [2, 4, 8, 16]
    vary_k  = run_vary_k(k_range, n_trials=30, n=100, d=64, sigma_md=0.01, seed=9)

    print("Running rotation comparison…")
    n_range_rot = [10, 20, 40, 80, 160]
    vary_rot = run_rotation_comparison(n_range_rot, n_trials=30, d=64, seed=10)

    print("All experiments done. Plotting…")

    # ======================================
    #  FIGURE 1 – Original plots (unchanged)
    # ======================================
    plt.figure(figsize=(12, 5))
    plt.suptitle("Original Algorithms – Monte-Carlo Trials", fontsize=13, fontweight='bold')

    plt.subplot(1, 2, 1)
    plt.plot(err1_list, label="Algo1 OneBitCQ")
    plt.plot(err2_list, label="Algo2 KLevelCQ")
    plt.xlabel("Trial")
    plt.ylabel("Absolute Error")
    plt.title("Scalar Mean Estimation Error")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(mse3_list, label="Algo3 WalshHadamardCQ")
    plt.xlabel("Trial")
    plt.ylabel("MSE")
    plt.title("Vector Mean Estimation MSE")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("original_trials.png", dpi=130, bbox_inches='tight')
    plt.show()

    # ======================================
    #  FIGURE 2 – 1-D Scalar comparison
    # ======================================
    # Color / style palette matching paper's blue (+) / orange (+)
    COLORS = {
        'Independent 1-bit':   ('#2196F3', '--'),
        'Correlated 1-bit':    ('#FF5722', '-'),
        'Independent 4-level': ('#4CAF50', '--'),
        'Correlated 4-level':  ('#9C27B0', '-'),
    }

    plt.figure(figsize=(13, 5))
    plt.suptitle("1-D Scalar Mean Estimation: Correlated vs Independent",
                 fontsize=13, fontweight='bold')

    # Left: absolute error per trial
    plt.subplot(1, 2, 1)
    for name, errs in scalar_errors.items():
        c, ls = COLORS.get(name, ('grey', '-'))
        plt.plot(errs, label=name, color=c, linestyle=ls, alpha=0.8)
    plt.xlabel("Trial")
    plt.ylabel("Absolute Error")
    plt.title("Error Trace (300 trials)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.4)

    # Right: box-plot
    plt.subplot(1, 2, 2)
    data_box  = list(scalar_errors.values())
    labels_box = list(scalar_errors.keys())
    bp = plt.boxplot(data_box, patch_artist=True, notch=False)
    colors_box = [COLORS.get(l, ('grey', '-'))[0] for l in labels_box]
    for patch, col in zip(bp['boxes'], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    plt.xticks(range(1, len(labels_box)+1), labels_box, rotation=15, ha='right', fontsize=8)
    plt.ylabel("Absolute Error")
    plt.title("Error Distribution")
    plt.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig("scalar_comparison.png", dpi=130, bbox_inches='tight')
    plt.show()

    # =======================================
    #  FIGURE 3 – Figure 2
    # =======================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparison of Compression Algorithms on Mean Estimation\n"
                 "(reproducing Fig. 2 of Suresh et al. 2022)",
                 fontsize=13, fontweight='bold')

    CURVE_STYLE = {
        'Independent': dict(color='#2196F3', marker='+', linestyle='-',  label='Independent'),
        'Correlated':  dict(color='#FF5722', marker='+', linestyle='-',  label='Correlated'),
    }
    ROT_STYLE = {
        'Independent':           dict(color='#2196F3', marker='+', ls='--', label='Independent'),
        'Independent+Rotation':  dict(color='#4CAF50', marker='+', ls='-',  label='Independent+Rotation'),
        'Correlated':            dict(color='#FF5722', marker='+', ls='--', label='Correlated'),
        'Correlated+Rotation':   dict(color='#9C27B0', marker='+', ls='-',  label='Correlated+Rotation'),
    }

    # --- (a) RMSE vs sigma_md ---
    ax = axes[0, 0]
    for name, vals in vary_sigma.items():
        means = [v[0] for v in vals]
        stds  = [v[1] for v in vals]
        s = CURVE_STYLE[name]
        ax.errorbar(sigma_md_range, means, stds,
                    color=s['color'], marker=s['marker'], label=s['label'])
    ax.set_xscale('log')
    ax.set_xticks(sigma_md_range)
    ax.set_xticklabels([str(s) for s in sigma_md_range])
    ax.set_xlabel('σ_md')
    ax.set_ylabel('RMSE')
    ax.set_title('(a) RMSE as a function of σ_md\n(n=100, k=2, d=64)')
    ax.legend()
    ax.grid(alpha=0.4)

    # --- (b) log2(RMSE) vs k ---
    ax = axes[0, 1]
    for name, vals in vary_k.items():
        log2_means = [np.log2(v[0]) for v in vals]
        log2_stds  = [v[1] / (v[0] * np.log(2) + 1e-12) for v in vals]
        s = CURVE_STYLE[name]
        ax.errorbar(k_range, log2_means, log2_stds,
                    color=s['color'], marker=s['marker'], label=s['label'])
    ax.set_xscale('log')
    ax.set_xticks(k_range)
    ax.set_xticklabels([str(k) for k in k_range])
    ax.set_xlabel('k (quantization levels)')
    ax.set_ylabel('log₂(RMSE)')
    ax.set_title('(b) RMSE as a function of k\n(n=100, σ_md=0.01, d=64)')
    ax.legend()
    ax.grid(alpha=0.4)

    # --- (c) log2(RMSE) vs n ---
    ax = axes[1, 0]
    for name, vals in vary_n.items():
        log2_means = [np.log2(v[0]) for v in vals]
        log2_stds  = [v[1] / (v[0] * np.log(2) + 1e-12) for v in vals]
        s = CURVE_STYLE[name]
        ax.errorbar(n_range, log2_means, log2_stds,
                    color=s['color'], marker=s['marker'], label=s['label'])
    ax.set_xscale('log')
    ax.set_xticks(n_range)
    ax.set_xticklabels([str(n) for n in n_range])
    ax.set_xlabel('n (number of clients)')
    ax.set_ylabel('log₂(RMSE)')
    ax.set_title('(c) RMSE as a function of n\n(σ_md=0.01, k=2, d=64)')
    ax.legend()
    ax.grid(alpha=0.4)

    # --- (d) RMSE with rotation ---
    ax = axes[1, 1]
    for name, vals in vary_rot.items():
        log2_means = [np.log2(v[0]) for v in vals]
        log2_stds  = [v[1] / (v[0] * np.log(2) + 1e-12) for v in vals]
        s = ROT_STYLE[name]
        ax.errorbar(n_range_rot, log2_means, log2_stds,
                    color=s['color'], marker=s['marker'], linestyle=s['ls'],
                    label=s['label'])
    ax.set_xscale('log')
    ax.set_xticks(n_range_rot)
    ax.set_xticklabels([str(n) for n in n_range_rot])
    ax.set_xlabel('n (number of clients)')
    ax.set_ylabel('log₂(RMSE)')
    ax.set_title('(d) RMSE with random rotation\n(σ_md=0.01, k=2, d=64)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig("paper_figure2_reproduction.png", dpi=130, bbox_inches='tight')
    plt.show()

    # ========================================
    #  FIGURE 4 – Summary bar chart of
    #  average errors across all methods
    # ========================================
    plt.figure(figsize=(10, 5))
    plt.suptitle("Average Error Summary: Independent vs Correlated\n"
                 "(scalar 1-D mean estimation, 300 trials)",
                 fontsize=12, fontweight='bold')

    algo_names = list(scalar_errors.keys())
    avg_errors  = [np.mean(v) for v in scalar_errors.values()]
    std_errors  = [np.std(v)  for v in scalar_errors.values()]
    bar_colors  = [COLORS.get(n, ('grey', '-'))[0] for n in algo_names]

    bars = plt.bar(algo_names, avg_errors, yerr=std_errors,
                   color=bar_colors, alpha=0.75, capsize=6, edgecolor='black')
    for bar, val in zip(bars, avg_errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.ylabel("Average Absolute Error")
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig("summary_bar.png", dpi=130, bbox_inches='tight')
    plt.show()

    # ========================================
    #  PRINT SUMMARY
    # ========================================
    print("\n" + "="*55)
    print("  SCALAR 1-D MEAN ESTIMATION SUMMARY (300 trials)")
    print("="*55)
    for name, errs in scalar_errors.items():
        print(f"  {name:<30s}  mean err = {np.mean(errs):.5f}  "
              f"  std = {np.std(errs):.5f}")

    print("\n" + "="*55)
    print("  ORIGINAL ALGORITHM AVERAGES")
    print("="*55)
    print("  Average Error Algo1 (CorrOneBit) :", np.mean(err1_list))
    print("  Average Error Algo2 (CorrKLevel) :", np.mean(err2_list))
    print("  Average MSE   Algo3 (WalshHadCQ) :", np.mean(mse3_list))
