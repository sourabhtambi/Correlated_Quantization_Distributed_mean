"""
Experiment: Federated Averaging on MNIST
  - Logistic regression (multinomial, 10 classes)
  - d = 784 * 10 + 10 = 7850 parameters
  - 1000 communication rounds of FedAvg
  - 5 repeated trials
  - Per-scheme total timing
"""

import numpy as np
from scipy.linalg import hadamard
import warnings
import time

warnings.filterwarnings('ignore')


# ================================================================
# Quantizers
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


def correlated_quantization_batch(data, rng, l, r, batch_size=100):
    n, d = data.shape
    ranges = r - l
    ys = (data - l[None, :]) / ranges[None, :]
    ys = np.clip(ys, 0, 1)
    Q = np.zeros_like(data)
    for start in range(0, d, batch_size):
        end = min(start + batch_size, d)
        pi = np.argsort(rng.random((n, end - start)), axis=0)
        gamma = rng.uniform(0, 1.0 / n, size=(n, end - start))
        U = pi / n + gamma
        Q[:, start:end] = (U < ys[:, start:end])
    return l[None, :] + ranges[None, :] * Q


def fast_walsh_hadamard_transform(x):
    """Fast WHT in O(d log d). No matrix construction."""
    d = x.shape[-1]
    result = x.copy().astype(float)
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            a = result[..., i:i+h].copy()
            b = result[..., i+h:i+2*h].copy()
            result[..., i:i+h] = a + b
            result[..., i+h:i+2*h] = a - b
        h *= 2
    result /= np.sqrt(d)
    return result


def walsh_hadamard_quantize_grads(grads, rng, R, quantize_fn):
    """Algorithm 3 — FAST version using O(d log d) WHT."""
    n, d = grads.shape
    d_pad = 1
    while d_pad < d:
        d_pad *= 2
    if d_pad != d:
        grads_padded = np.zeros((n, d_pad))
        grads_padded[:, :d] = grads
    else:
        grads_padded = grads.copy()

    D_diag = rng.choice([-1, 1], size=d_pad)
    grads_padded *= D_diag[None, :]
    rotated = fast_walsh_hadamard_transform(grads_padded)
    scale = np.sqrt(d_pad) / (R * np.sqrt(8 * np.log(d_pad * n + 1)))
    rotated = np.clip(rotated * scale, -1, 1)

    l_rot = np.full(d_pad, -1.0)
    r_rot = np.full(d_pad, 1.0)
    Q_rot = quantize_fn(rotated, rng, l_rot, r_rot)

    inv_scale = R * np.sqrt(8 * np.log(d_pad * n + 1)) / np.sqrt(d_pad)
    Q_original = fast_walsh_hadamard_transform(Q_rot)
    Q_original *= D_diag[None, :]
    Q_original *= inv_scale

    if d_pad != d:
        Q_original = Q_original[:, :d]
    return Q_original


# ================================================================
# Logistic Regression model
# ================================================================

def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict(X, W, b):
    return softmax(X @ W + b[None, :])


def compute_accuracy(X, y, W, b):
    probs = predict(X, W, b)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)


def compute_gradient(X, y, W, b):
    n = X.shape[0]
    probs = predict(X, W, b)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y] = 1.0
    diff = (probs - one_hot) / n
    grad_W = X.T @ diff
    grad_b = diff.sum(axis=0)
    return grad_W, grad_b


# ================================================================
# Quantize client gradients
# ================================================================

def quantize_client_gradients(client_grad_flats, quantizer_name, rng):
    n, d = client_grad_flats.shape

    if quantizer_name == "none":
        return np.mean(client_grad_flats, axis=0)

    R = np.max(np.linalg.norm(client_grad_flats, axis=1))
    if R < 1e-12:
        return np.mean(client_grad_flats, axis=0)

    l = np.full(d, -R)
    r = np.full(d, R)

    if quantizer_name == "independent":
        Q = independent_quantization(client_grad_flats, rng, l, r)
        return np.mean(Q, axis=0)

    elif quantizer_name == "correlated":
        Q = correlated_quantization_batch(client_grad_flats, rng, l, r)
        return np.mean(Q, axis=0)

    elif quantizer_name == "independent+rotation":
        Q = walsh_hadamard_quantize_grads(
            client_grad_flats, rng, R, independent_quantization
        )
        return np.mean(Q, axis=0)

    elif quantizer_name == "correlated+rotation":
        Q = walsh_hadamard_quantize_grads(
            client_grad_flats, rng, R, correlated_quantization_batch
        )
        return np.mean(Q, axis=0)

    else:
        raise ValueError(f"Unknown quantizer: {quantizer_name}")


# ================================================================
# Federated Averaging
# ================================================================

def federated_averaging(X_trains, y_trains, X_test, y_test,
                        quantizer_name, n_rounds, learning_rate,
                        clients_per_round, rng):
    d_features = X_trains[0].shape[1]
    n_classes = 10
    W = np.zeros((d_features, n_classes))
    b = np.zeros(n_classes)
    total_clients = len(X_trains)

    for t in range(n_rounds):
        if clients_per_round < total_clients:
            selected = rng.choice(
                total_clients, size=clients_per_round, replace=False
            )
        else:
            selected = np.arange(total_clients)

        n_selected = len(selected)
        client_grad_flats = []
        for idx in selected:
            gW, gb = compute_gradient(X_trains[idx], y_trains[idx], W, b)
            grad_flat = np.concatenate([gW.ravel(), gb])
            client_grad_flats.append(grad_flat)

        client_grad_flats = np.array(client_grad_flats)

        q_grad_flat = quantize_client_gradients(
            client_grad_flats, quantizer_name, rng
        )

        d_params = d_features * n_classes
        q_grad_W = q_grad_flat[:d_params].reshape(d_features, n_classes)
        q_grad_b = q_grad_flat[d_params:]

        W -= learning_rate * q_grad_W
        b -= learning_rate * q_grad_b

    accuracy = compute_accuracy(X_test, y_test, W, b)
    return accuracy


# ================================================================
# Load MNIST
# ================================================================

def load_mnist():
    from sklearn.datasets import fetch_openml
    print("  Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False,
                         parser='auto')
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(int)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


def split_data_to_clients(X_train, y_train, n_clients, rng):
    n = X_train.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    X_trains = [X_train[s] for s in splits]
    y_trains = [y_train[s] for s in splits]
    return X_trains, y_trains


# ================================================================
# Main — with per-scheme timing
# ================================================================

def main():
    X_train, y_train, X_test, y_test = load_mnist()

    n_clients = 100
    n_rounds = 1000
    learning_rate = 0.5
    n_trials = 5
    clients_per_round = 20

    quantizers = {
        "No quantization":      "none",
        "Independent":          "independent",
        "Independent+Rotation": "independent+rotation",
        "Correlated":           "correlated",
        "Correlated+Rotation":  "correlated+rotation",
    }

    print("=" * 70)
    print("Table 2 (FedAvg): Federated Learning on MNIST")
    print(f"  Logistic regression, {n_clients} clients, "
          f"{clients_per_round} per round, "
          f"{n_rounds} rounds, {n_trials} trials")
    print("=" * 70)

    results = {}
    timing = {}

    for name, q_name in quantizers.items():
        trial_accs = []
        scheme_start = time.time()

        for trial in range(n_trials):
            trial_start = time.time()
            rng = np.random.default_rng(seed=trial * 53)

            X_trains, y_trains = split_data_to_clients(
                X_train, y_train, n_clients, rng
            )

            acc = federated_averaging(
                X_trains, y_trains, X_test, y_test,
                q_name, n_rounds, learning_rate,
                clients_per_round, rng
            )
            trial_accs.append(acc * 100)
            trial_time = time.time() - trial_start

            print(f"  {name:25s}  trial {trial+1}/{n_trials}: "
                  f"{acc*100:.2f}%  ({trial_time:.1f}s)")

        scheme_time = time.time() - scheme_start
        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        results[name] = (mean_acc, std_acc)
        timing[name] = scheme_time

        print(f"  {'-> ' + name:27s}  {mean_acc:.2f} ({std_acc:.2f})  "
              f"[total: {scheme_time:.1f}s, "
              f"avg: {scheme_time/n_trials:.1f}s/trial]\n")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Test Accuracy (%) and Timing")
    print("=" * 70)
    print(f"{'Algorithm':25s} | {'Accuracy %':>12} | "
          f"{'Total (s)':>10} | {'Per trial (s)':>13}")
    print("-" * 70)
    for name in quantizers:
        mean_acc, std_acc = results[name]
        total_t = timing[name]
        per_trial = total_t / n_trials
        print(f"{name:25s} | {mean_acc:5.2f} ({std_acc:.2f}) | "
              f"{total_t:10.1f} | {per_trial:13.1f}")


if __name__ == "__main__":
    main()
