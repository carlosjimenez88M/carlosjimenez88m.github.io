"""Generate all 6 plots for the statistical learning foundations post."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── 1. Bias-Variance Tradeoff ─────────────────────────────────────────────────
print("Generating bias_variance_tradeoff.png ...")
np.random.seed(42)

diabetes = load_diabetes()
X_raw = diabetes.data[:, 2].reshape(-1, 1)
y_raw = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42
)

degrees = range(1, 16)
n_bootstrap = 100

bias2_list, var_list, mse_list = [], [], []

for deg in degrees:
    predictions = np.zeros((n_bootstrap, len(X_test)))
    for b in range(n_bootstrap):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_b, y_b = X_train[idx], y_train[idx]
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1e-3))
        ])
        pipe.fit(X_b, y_b)
        predictions[b, :] = pipe.predict(X_test)

    mean_pred = predictions.mean(axis=0)
    bias2 = np.mean((mean_pred - y_test) ** 2)
    var = np.mean(predictions.var(axis=0))
    mse = np.mean((predictions - y_test[np.newaxis, :]) ** 2)

    bias2_list.append(bias2)
    var_list.append(var)
    mse_list.append(mse)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(degrees, bias2_list, 'b-o', label='Bias²', linewidth=2)
ax.plot(degrees, var_list, 'r-s', label='Variance', linewidth=2)
ax.plot(degrees, mse_list, 'k-^', label='Total MSE (≈ Bias² + Var)', linewidth=2, linestyle='--')
ax.axhline(y=min(mse_list), color='gray', linestyle=':', alpha=0.7, label='Optimal MSE')
ax.set_xlabel('Polynomial Degree (Model Complexity)', fontsize=13)
ax.set_ylabel('Error (Bootstrap Estimate)', fontsize=13)
ax.set_title('Bias-Variance Decomposition: Diabetes Dataset', fontsize=14)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'bias_variance_tradeoff.png'), dpi=150, bbox_inches='tight')
plt.close()

opt_deg = list(degrees)[np.argmin(mse_list)]
print(f"  Optimal polynomial degree: {opt_deg}")
print(f"  Min test MSE: {min(mse_list):.2f}")
print(f"  At degree {opt_deg}: Bias²={bias2_list[opt_deg-1]:.2f}, Var={var_list[opt_deg-1]:.2f}")

# ── 2. Spaghetti Plot ─────────────────────────────────────────────────────────
print("Generating spaghetti_variance.png ...")
np.random.seed(42)

diabetes = load_diabetes()
X_raw = diabetes.data[:, 2].reshape(-1, 1)
y_raw = diabetes.target

n_bootstrap = 30
deg_list = [1, 5, 12]
x_range = np.linspace(X_raw.min(), X_raw.max(), 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, deg in zip(axes, deg_list):
    boot_preds = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_raw), len(X_raw), replace=True)
        X_b, y_b = X_raw[idx], y_raw[idx]
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1e-3))
        ])
        pipe.fit(X_b, y_b)
        boot_preds.append(pipe.predict(x_range))
    boot_preds = np.array(boot_preds)

    for curve in boot_preds:
        ax.plot(x_range, curve, alpha=0.2, color='steelblue', linewidth=0.8)
    ax.plot(x_range, boot_preds.mean(axis=0), color='navy', linewidth=2, label='Mean prediction')
    ax.scatter(X_raw, y_raw, alpha=0.2, color='gray', s=10)
    ax.set_title(f'Degree {deg}', fontsize=13)
    ax.set_xlabel('BMI (standardized)', fontsize=11)
    ax.set_ylim(-100, 400)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Diabetes Progression', fontsize=11)
fig.suptitle('Spaghetti Plot: Each Curve = One Bootstrap Training Set', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'spaghetti_variance.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 3. Point-wise Variance ────────────────────────────────────────────────────
print("Generating pointwise_variance.png ...")
np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 5))
colors_pw = {'1': 'steelblue', '5': 'darkorange', '12': 'crimson'}

for deg in deg_list:
    boot_preds = []
    for _ in range(100):
        idx = np.random.choice(len(X_raw), len(X_raw), replace=True)
        X_b, y_b = X_raw[idx], y_raw[idx]
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1e-3))
        ])
        pipe.fit(X_b, y_b)
        boot_preds.append(pipe.predict(x_range))
    pw_var = np.array(boot_preds).var(axis=0)
    ax.plot(x_range.ravel(), pw_var, label=f'Degree {deg}',
            color=colors_pw[str(deg)], linewidth=2)

ax2 = ax.twinx()
ax2.hist(X_raw.ravel(), bins=30, alpha=0.15, color='gray', density=True)
ax2.set_ylabel('Training data density', fontsize=10, color='gray')
ax2.tick_params(axis='y', colors='gray')

ax.set_xlabel('BMI (standardized)', fontsize=12)
ax.set_ylabel('Point-wise Var($\\hat{f}(x_0)$)', fontsize=12)
ax.set_title('Variance is Heteroskedastic: Highest at Sparse Extremes', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'pointwise_variance.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 4. KNN Iris ───────────────────────────────────────────────────────────────
print("Generating knn_iris.png ...")
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = range(1, 51)
train_errors, cv_errors = [], []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)
    train_err = 1 - knn.score(X_scaled, y)
    cv_err = 1 - cross_val_score(knn, X_scaled, y, cv=5).mean()
    train_errors.append(train_err)
    cv_errors.append(cv_err)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(k_values, train_errors, 'b-', label='Training Error', linewidth=2)
ax.plot(k_values, cv_errors, 'r--', label='CV Test Error', linewidth=2)
ax.axvline(x=k_values[np.argmin(cv_errors)], color='green', linestyle=':', alpha=0.8,
           label=f'Optimal K={k_values[np.argmin(cv_errors)]}')
ax.set_xlabel('K (Number of Neighbors)', fontsize=12)
ax.set_ylabel('Error Rate', fontsize=12)
ax.set_title('KNN: Training vs. Cross-Validated Error — Iris', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

ax2 = axes[1]
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

colors = ['#AADAFF', '#FFAAAA', '#AAFFAA']
cmap_light = ListedColormap(colors)
cmap_bold = ListedColormap(['#1F77B4', '#D62728', '#2CA02C'])

k_plot = 10
knn_plot = KNeighborsClassifier(n_neighbors=k_plot)
knn_plot.fit(X_scaled, y)
Z = knn_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax2.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold,
            edgecolors='k', s=40)
ax2.set_title(f'Decision Boundary — KNN (K={k_plot})', fontsize=13)
ax2.set_xlabel('Sepal Length (scaled)', fontsize=11)
ax2.set_ylabel('Sepal Width (scaled)', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'knn_iris.png'), dpi=150, bbox_inches='tight')
plt.close()

opt_k = list(k_values)[np.argmin(cv_errors)]
print(f"  Optimal K: {opt_k}")
print(f"  Min CV error: {min(cv_errors):.4f}")

# ── 5. Learning Curves ────────────────────────────────────────────────────────
print("Generating learning_curves.png ...")
url = "https://www.statlearning.com/s/Advertising.csv"
ads = pd.read_csv(url, index_col=0)

X_ads = ads[["TV", "radio", "newspaper"]].values
y_ads = ads["sales"].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, deg, title in zip(
    axes,
    [1, 3],
    ['Linear (degree=1)', 'Polynomial (degree=3)']
):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
        ("lr", LinearRegression())
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X_ads, y_ads,
        train_sizes=np.linspace(0.1, 1.0, 15),
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    train_mse = -train_scores.mean(axis=1)
    test_mse = -test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.plot(train_sizes, train_mse, 'b-o', label='Train MSE', linewidth=2)
    ax.plot(train_sizes, test_mse, 'r-s', label='CV Test MSE', linewidth=2)
    ax.fill_between(train_sizes,
                    test_mse - test_std,
                    test_mse + test_std,
                    alpha=0.15, color='red')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Learning Curves: Advertising Dataset', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 6. Stratified Split MSE ───────────────────────────────────────────────────
print("Generating stratified_split_mse.png ...")
np.random.seed(0)

url = "https://www.statlearning.com/s/Advertising.csv"
ads = pd.read_csv(url, index_col=0)

X_ads = ads[["TV", "radio", "newspaper"]].values
y_ads = ads["sales"].values

sales_quartile = pd.qcut(y_ads, q=4, labels=False)

n_reps = 200
mse_random, mse_stratified = [], []

for _ in range(n_reps):
    X_tr, X_te, y_tr, y_te = train_test_split(X_ads, y_ads, test_size=40)
    lr = LinearRegression().fit(X_tr, y_tr)
    mse_random.append(mean_squared_error(y_te, lr.predict(X_te)))

    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_ads, y_ads, test_size=40, stratify=sales_quartile
    )
    lr_s = LinearRegression().fit(X_tr_s, y_tr_s)
    mse_stratified.append(mean_squared_error(y_te_s, lr_s.predict(X_te_s)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(mse_random, bins=30, alpha=0.6, color='steelblue', label='Random split')
ax.hist(mse_stratified, bins=30, alpha=0.6, color='darkorange', label='Stratified split')
ax.axvline(np.mean(mse_random), color='steelblue', linestyle='--', linewidth=1.5,
           label=f'Random mean={np.mean(mse_random):.2f}, σ={np.std(mse_random):.2f}')
ax.axvline(np.mean(mse_stratified), color='darkorange', linestyle='--', linewidth=1.5,
           label=f'Stratified mean={np.mean(mse_stratified):.2f}, σ={np.std(mse_stratified):.2f}')
ax.set_xlabel('Test MSE', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Evaluation Variance: Random vs. Stratified Train/Test Split\n(200 repetitions, n=40 test)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'stratified_split_mse.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Random split:     mean MSE={np.mean(mse_random):.3f}, σ={np.std(mse_random):.3f}")
print(f"  Stratified split: mean MSE={np.mean(mse_stratified):.3f}, σ={np.std(mse_stratified):.3f}")
print(f"  Variance reduction: {(1 - np.std(mse_stratified)/np.std(mse_random))*100:.1f}%")

print("\nDone! All 6 PNGs generated.")
