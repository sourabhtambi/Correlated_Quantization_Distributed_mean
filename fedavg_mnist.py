""""
Experiment: Federated Averaging on MNIST
  - Logistic regression (multinomial, 10 classes)
  - d = 784 * 10 + 10 = 7850 parameters
  - 1000 communication rounds of FedAvg
  - 5 repeated trials
"""

import numpy as np
from scipy.linalg import hadamard
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# Quantizers (same as previous wrok)
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





def walsh_hadamard_quantize_grads(grads, rng, R, quantize_fn):
    """
    Algorithm 3 applied to gradient vectors.
    """
    n, d = grads.shape

    # Pad d to next power of 2 for Hadamard
    d_pad = 1
    while d_pad < d:
        d_pad *= 2

    if d_pad != d:
        grads_padded = np.zeros((n, d_pad))
        grads_padded[:, :d] = grads
    else:
        grads_padded = grads

    H = hadamard(d_pad)
    D_diag = rng.choice([-1, 1], size=d_pad)
    W = (1.0 / np.sqrt(d_pad)) * H * D_diag[None, :]

    scale = np.sqrt(d_pad) / (R * np.sqrt(8 * np.log(d_pad * n + 1)))

    rotated = np.clip((grads_padded @ W.T) * scale, -1, 1)

    l_rot = np.full(d_pad, -1.0)
    r_rot = np.full(d_pad, 1.0)
    Q_rot = quantize_fn(rotated, rng, l_rot, r_rot)

    inv_scale = R * np.sqrt(8 * np.log(d_pad * n + 1)) / np.sqrt(d_pad)
    Q_original = (Q_rot @ W) * inv_scale

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
    """Gradient of cross-entropy loss w.r.t. W and b."""
    n = X.shape[0]
    probs = predict(X, W, b)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y] = 1.0
    diff = (probs - one_hot) / n
    grad_W = X.T @ diff
    grad_b = diff.sum(axis=0)
    return grad_W, grad_b


# ================================================================
# Quantize client gradients — the core distributed protocol
# ================================================================

def quantize_client_gradients(client_grad_flats, quantizer_name, rng):
    """
    Each client has already computed their local gradient.
    This function quantizes them and returns the server's estimate
    of the mean gradient.

BASIC INFO: 
    client_grad_flats: (n_clients, d) — each row is one client's gradient
    quantizer_name: which quantizer to use
    rng: random generator

    Returns: (d,) — server's estimate of mean gradient
    """
    n, d = client_grad_flats.shape

    if quantizer_name == "none":
        return np.mean(client_grad_flats, axis=0)

    # Compute range: L2 ball [-R, R] per coordinate
    R = np.max(np.linalg.norm(client_grad_flats, axis=1))
    if R < 1e-12:
        return np.mean(client_grad_flats, axis=0)

    l = np.full(d, -R)
    r = np.full(d, R)

    if quantizer_name == "independent":
        Q = independent_quantization(client_grad_flats, rng, l, r)
        return np.mean(Q, axis=0)

    elif quantizer_name == "correlated":
        Q = correlated_quantization(client_grad_flats, rng, l, r)
        return np.mean(Q, axis=0)

    #elif quantizer_name == "terngrad":
        Q = terngrad_quantization(client_grad_flats, rng, l, r)
        return np.mean(Q, axis=0)

    elif quantizer_name == "independent+rotation":
        Q = walsh_hadamard_quantize_grads(
            client_grad_flats, rng, R, independent_quantization
        )
        return np.mean(Q, axis=0)

    elif quantizer_name == "correlated+rotation":
        Q = walsh_hadamard_quantize_grads(
            client_grad_flats, rng, R, correlated_quantization
        )
        return np.mean(Q, axis=0)

    else:
        raise ValueError(f"Unknown quantizer: {quantizer_name}")


# ================================================================
# Federated Averaging — FIXED: per-client quantization
# ================================================================

def federated_averaging(X_trains, y_trains, X_test, y_test,
                        quantizer_name, n_rounds, learning_rate,
                        clients_per_round, rng):
    """
    Federated Averaging :

    Protocol per round:
      1. Server broadcasts current model (W, b) to selected clients(which are randomly selected)
      2. Each client computes local gradient on their own data
      3. Each client QUANTIZES their gradient (uplink compression) <- here is our role
      4. Server receives n quantized gradients and AVERAGES them(using FEDAvg )
      5. Server updates global model
      6. Serever again broadcasts this updated global model

    Remember Quantization happens
    at the client side, not after server-side aggregation.
    """
    d_features = X_trains[0].shape[1]  # 784 <-MNIST Input features
    n_classes = 10 # <- Classification Category

    # Initialize global model
    W = np.zeros((d_features, n_classes)) # weight matrix
    b = np.zeros(n_classes) # bias vector

    total_clients = len(X_trains)

    for t in range(n_rounds): #<- #comunication rounds
        # Select clients for this round
        if clients_per_round < total_clients:
            selected = rng.choice(
                total_clients, size=clients_per_round, replace=False
            )
        else:
            selected = np.arange(total_clients)

        n_selected = len(selected)

        # Each client computes their LOCAL gradient
        client_grad_flats = []
        for idx in selected: # happening over client side
            gW, gb = compute_gradient(
                X_trains[idx], y_trains[idx], W, b
            )
            # Flatten: concat W gradient and b gradient
            grad_flat = np.concatenate([gW.ravel(), gb]) # ravel() <- flatten matrix into vector 
            client_grad_flats.append(grad_flat)

        # Stack into (n_selected, d) matrix
        client_grad_flats = np.array(client_grad_flats) #shape = (n_selected ,d)

        # SERVER: quantize and aggregate

        # Each client's gradient is quantized individually,
        # then the server averages the quantized versions
        q_grad_flat = quantize_client_gradients(
            client_grad_flats, quantizer_name, rng
        )

        # Unflatten
        d_params = d_features * n_classes
        q_grad_W = q_grad_flat[:d_params].reshape(d_features, n_classes)
        q_grad_b = q_grad_flat[d_params:]

        # Update global model <- gradient descent updates
        W -= learning_rate * q_grad_W
        b -= learning_rate * q_grad_b

    # Final evaluation
    accuracy = compute_accuracy(X_test, y_test, W, b)
    return accuracy


# ================================================================
# Load MNIST
# ================================================================

def load_mnist():
    """Load MNIST using sklearn."""
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
    """Split training data evenly across n_clients."""
    n = X_train.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    X_trains = [X_train[s] for s in splits]
    y_trains = [y_train[s] for s in splits]
    return X_trains, y_trains


# ================================================================
# Main
# ================================================================

def main():
    """
    Reproduce Table 2, FedAvg column.

    Paper: Federated MNIST (341K, 3383 clients), logistic regression,
           1000 rounds, 5 trials, k=2 (except TernGrad k=3).

    We use sklearn MNIST (60K train, 10K test) with 100 clients.
    """
    X_train, y_train, X_test, y_test = load_mnist()

    n_clients = 100
    n_rounds = 1000
    learning_rate = 0.5
    n_trials = 5
    clients_per_round = 20

    quantizers = {
        "No quantization":          "none",
        "Independent":              "independent",
        "Independent+Rotation":     "independent+rotation",
        "TernGrad (log2(3) bits)":  "terngrad",
        "Correlated":               "correlated",
        "Correlated+Rotation":      "correlated+rotation",
    }

    print("=" * 65)
    print("Table 2 (FedAvg): Federated Learning on MNIST")
    print(f"  Logistic regression, {n_clients} clients, "
          f"{clients_per_round} per round, "
          f"{n_rounds} rounds, {n_trials} trials")
    print("=" * 65)

    results = {}

    for name, q_name in quantizers.items():
        trial_accs = []

        for trial in range(n_trials):
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
            print(f"  {name:30s}  trial {trial+1}/{n_trials}: "
                  f"{acc*100:.2f}%")

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        results[name] = (mean_acc, std_acc)
        print(f"  {'-> ' + name:32s}  {mean_acc:.2f} ({std_acc:.2f})\n")

    # Summary table
    print("\n" + "=" * 50)
    print("Summary: Test Accuracy (%)")
    print("=" * 50)
    print(f"{'Algorithm':30s} | {'Accuracy %':>12}")
    print("-" * 50)
    for name, (mean_acc, std_acc) in results.items():
        print(f"{name:30s} | {mean_acc:5.2f} ({std_acc:.2f})")


if __name__ == "__main__":
    main()
