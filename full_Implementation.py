import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# ALGORITHM 1 : OneDimOneBitCQ
# ──────────────────────────────────────────────
def one_dim_one_bit_cq(x, l, r):
    """
    Algorithm 1: One-dimensional one-bit correlated quantization.

    Input : x  – array of n client values, all in [l, r)
            l  – lower bound
            r  – upper bound
    Output: unbiased scalar estimate of mean(x)
    """
    n = len(x)

    # Step 1 : normalise to [0, 1)
    y = (x - l) / (r - l)

    # Step 2 : correlated U_i = pi_i/n + gamma_i
    pi    = np.random.permutation(n)           # random permutation of {0,...,n-1}
    gamma = np.random.uniform(0, 1/n, size=n)  # gamma_i ~ U[0, 1/n)
    U     = pi / n + gamma

    # Step 3 : Q_i(x_i) = (r-l) * 1[U_i < y_i]
    Q = (r - l) * (U < y).astype(float)

    # Output : shift back by l
    return l + Q.mean()


# ──────────────────────────────────────────────
# ALGORITHM 2 : OneDimKLevelsCQ
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# ALGORITHM 3 : WalshHadamardCQ
# ──────────────────────────────────────────────
def walsh_hadamard_cq(X, R, k=2):
    """
    Algorithm 3: Walsh-Hadamard correlated quantization.

    Input : X  – (n, d) matrix; each row is one client vector with ||x_i||_2 <= R
            R  – radius bound
            k  – quantization levels per coordinate (default 2 = 1 bit)
    Output: x_hat – (d,) unbiased estimate of the mean (1/n)*sum x_i
    """
    n, d = X.shape

    # pad d to the next power of 2
    d_pad = 1
    while d_pad < d:
        d_pad *= 2

    # Build W = (1/sqrt(d_pad)) * H * D
    H    = hadamard(d_pad).astype(float)
    diag = np.random.choice([-1.0, 1.0], size=d_pad)

    def apply_W(v):
        return (H @ (diag * v)) / np.sqrt(d_pad)

    def apply_W_inv(v):
        return diag * (H @ v) / np.sqrt(d_pad)

    # scale factor
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


# ──────────────────────────────────────────────
# TEST + MONTE CARLO + PLOTTING
# ──────────────────────────────────────────────
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


    # ───────────────── PLOTS ─────────────────

    plt.figure(figsize=(12,5))

    # Error plot Algo 1 vs Algo 2
    plt.subplot(1,2,1)
    plt.plot(err1_list, label="Algo1 OneBitCQ")
    plt.plot(err2_list, label="Algo2 KLevelCQ")
    plt.xlabel("Trial")
    plt.ylabel("Absolute Error")
    plt.title("Scalar Mean Estimation Error")
    plt.legend()
    plt.grid()

    # MSE plot Algo 3
    plt.subplot(1,2,2)
    plt.plot(mse3_list, label="Algo3 WalshHadamardCQ")
    plt.xlabel("Trial")
    plt.ylabel("MSE")
    plt.title("Vector Mean Estimation MSE")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


    # Print averages (important for paper reproduction)
    print("\nAverage Error Algo1:", np.mean(err1_list))
    print("Average Error Algo2:", np.mean(err2_list))
    print("Average MSE Algo3 :", np.mean(mse3_list))