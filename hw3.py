import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Task 1 - Draw samples from 2 Gaussians
# ============================================================

def datagen(n1, n2, m1, m2, sig1, sig2, rng):
    """
    Generate 2D Gaussian samples for binary classification.

    Returns:
        X: array of shape (n1+n2, 2)
        y: array of shape (n1+n2,) with labels +1 or -1
    """
    m1 = np.array(m1)
    m2 = np.array(m2)

    # x ~ N(0,1)^2, then sig * x + mean  =>  N(mean, sig^2 * I2)
    X1 = rng.normal(0, 1, (n1, 2)) * sig1 + m1   # class +1
    X2 = rng.normal(0, 1, (n2, 2)) * sig2 + m2   # class -1

    X = np.vstack([X1, X2])
    y = np.concatenate([np.ones(n1), -np.ones(n2)])
    return X, y


rng = np.random.default_rng(42)
X, y = datagen(50, 50, [1, 1], [-1, -0.5], 2.0, 2.0, rng)

plt.figure(figsize=(7, 5))
plt.scatter(X[y ==  1, 0], X[y ==  1, 1], c='steelblue', label='y = +1', alpha=0.7)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='tomato',    label='y = -1', alpha=0.7)
plt.title('Task 1: Samples from 2 Gaussians')
plt.xlabel('x₀')
plt.ylabel('x₁')
plt.legend()
plt.tight_layout()
plt.savefig('task1_gaussians.png', dpi=120)
plt.show()
print("Task 1 done – plot saved as task1_gaussians.png")


# ============================================================
# Task 2 - Test errors for one particular classifier
# ============================================================

# Fixed linear classifier:  a = w · x + b,  y_hat = sign(a)
w = np.array([2.0, 1.5])
b = -3.0 / 8.0

def classify_fixed(X):
    return np.sign(X @ w + b)

def binary_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def run_test_experiment(n, n_repeats=100, rng=None):
    accs = []
    for _ in range(n_repeats):
        X_test, y_test = datagen(n, n, [1, 1], [-1, -0.5], 2.0, 2.0, rng)
        y_pred = classify_fixed(X_test)
        accs.append(binary_accuracy(y_pred, y_test))
    return np.array(accs)

rng2 = np.random.default_rng(0)

print("\nTask 2: Fixed classifier – mean/std accuracy over 100 draws")
print(f"{'n1=n2':>8} | {'mean acc':>10} | {'std acc':>10}")
print("-" * 35)
for n in [10, 100, 1000]:
    accs = run_test_experiment(n, 100, rng2)
    print(f"n={n:>5} | {accs.mean():>10.4f} | {accs.std():>10.4f}")

# Observation: as n grows, std shrinks (law of large numbers) – mean stays stable.


# ============================================================
# Task 3 – Finish last week's NN task (skipped here)
# ============================================================


# ============================================================
# Task 4 - 1-NN Classifier (Overfitting & Varianz)
# ============================================================

def knn_predict(X_train, y_train, X_test):
    """
    1-Nearest Neighbor prediction.
    For each test point: find closest training point, return its label.
    """
    # diffs: (n_test, n_train, 2)
    diffs = X_test[:, None, :] - X_train[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))   # (n_test, n_train)
    nearest_idx = np.argmin(dists, axis=1)       # (n_test,)
    return y_train[nearest_idx]


rng3 = np.random.default_rng(1)
accs_1nn = []
n_runs = 10

print("\nTask 4: 1-NN Classifier (n1=n2=100 train & test, 10 runs)")
print(f"{'Run':>5} | {'Test Accuracy':>14}")
print("-" * 25)

for i in range(n_runs):
    X_train, y_train = datagen(100, 100, [1, 1], [-1, -0.5], 2.0, 2.0, rng3)
    X_test,  y_test  = datagen(100, 100, [1, 1], [-1, -0.5], 2.0, 2.0, rng3)
    y_pred = knn_predict(X_train, y_train, X_test)
    acc = binary_accuracy(y_pred, y_test)
    accs_1nn.append(acc)
    print(f"{i+1:>5} | {acc:>14.4f}")

print(f"\n1-NN  mean accuracy : {np.mean(accs_1nn):.4f}")
print(f"1-NN  std  accuracy : {np.std(accs_1nn):.4f}")
print("\nExpected: 1-NN accuracy < fixed classifier accuracy (overfitting)")


# ---- Bonus: Voronoi plot (requires scipy) ----------------------------
try:
    from scipy.spatial import Voronoi, voronoi_plot_2d

    rng4 = np.random.default_rng(7)
    X_tr, y_tr = datagen(30, 30, [1, 1], [-1, -0.5], 2.0, 2.0, rng4)

    fig, ax = plt.subplots(figsize=(8, 6))
    vor = Voronoi(X_tr)
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                    line_colors='gray', line_width=0.8, line_alpha=0.6)

    ax.scatter(X_tr[y_tr ==  1, 0], X_tr[y_tr ==  1, 1],
               c='steelblue', label='train y=+1', s=60, zorder=3)
    ax.scatter(X_tr[y_tr == -1, 0], X_tr[y_tr == -1, 1],
               c='tomato',    label='train y=-1', s=60, zorder=3)

    # also draw Task-2 decision boundary  w·x + b = 0
    x0_range = np.linspace(X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1, 200)
    # w[0]*x0 + w[1]*x1 + b = 0  =>  x1 = -(w[0]*x0 + b) / w[1]
    x1_boundary = -(w[0] * x0_range + b) / w[1]
    ax.plot(x0_range, x1_boundary, 'k--', lw=2, label='Task-2 boundary')

    ax.set_xlim(X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1)
    ax.set_ylim(X_tr[:, 1].min() - 1, X_tr[:, 1].max() + 1)
    ax.set_title('Task 4 Bonus: 1-NN Voronoi regions vs Task-2 boundary')
    ax.legend()
    plt.tight_layout()
    plt.savefig('task4_voronoi.png', dpi=120)
    plt.show()
    print("Task 4 bonus – Voronoi plot saved as task4_voronoi.png")
except ImportError:
    print("scipy not installed – skipping Voronoi bonus plot")
