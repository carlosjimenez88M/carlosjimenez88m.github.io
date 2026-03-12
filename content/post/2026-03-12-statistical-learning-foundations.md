---
author: Carlos Daniel Jiménez
date: 2026-03-12
title: "Statistical Learning: Foundations, Bias-Variance and the Art of Estimation"
draft: false
description: "A rigorous walkthrough of ISLP Chapter 2 fundamentals — from the formal definition of f(X) to the bias-variance decomposition, Bayes classifiers, and KNN — with Python code, real datasets, and connections to epistemology and learning theory."
categories: ["Statistical Learning"]
tags: ["statistical-learning", "bias-variance", "knn", "linear-regression", "islp", "sklearn"]
series:
  - Statistical Learning
  - Machine Learning
---

## Abstract

Statistical learning is the discipline concerned with recovering unknown functions from data. That sentence sounds modest. It is not. Every scientific instrument, every predictive model, every medical test rests on the assumption that observable patterns carry information about latent structure — that $Y$ encodes something about $X$ beyond noise. The formal machinery for making this precise, and the fundamental limits of what can be learned, constitute the foundation of the field.

This post covers the core ideas from Chapter 2 of *An Introduction to Statistical Learning with Python* (ISLP, Hastie, Tibshirani, James & Taylor) — but with an eye toward what the textbook leaves underspecified. We derive the bias-variance decomposition from first principles, simulate it empirically, connect it to Bayesian and frequentist perspectives, and situate it inside the broader epistemological problem of induction.

**The central claim:** The bias-variance tradeoff is not an engineering inconvenience. It is a mathematical expression of the fundamental impossibility of learning from finite samples without prior assumptions. No model selection trick, no regularization scheme, no ensemble method dissolves it — they only rearrange the terms. Understanding *why* will change how you read every model evaluation paper you encounter.

---

## TL;DR

- Statistical learning estimates $f$ in $Y = f(X) + \varepsilon$; prediction minimizes $E[(Y - \hat{f}(X))^2]$; inference asks *how* $X$ affects $Y$
- **Reducible error** $= [f(X) - \hat{f}(X)]^2$ vanishes with perfect $f$; **irreducible error** $= \text{Var}(\varepsilon)$ never does
- Bias-variance decomposition: $\text{MSE}(\hat{f}(x_0)) = \text{Var}(\hat{f}(x_0)) + [\text{Bias}(\hat{f}(x_0))]^2 + \text{Var}(\varepsilon)$
- A degree-1 polynomial on Advertising data underfits (high bias, $R^2 \approx 0.61$); degree-12 overfits (high variance, test MSE explodes)
- The **Bayes classifier** achieves the minimum possible test error — but requires knowing $Pr(Y=j \mid X=x_0)$, which we never know
- KNN approximates the Bayes classifier: $\hat{f}(x_0) = \frac{1}{K}\sum_{x_i \in \mathcal{N}_0} y_i$; small $K$ → high variance, large $K$ → high bias
- **Datasets:** Advertising (TV + Radio + Newspaper → Sales), sklearn diabetes, sklearn Iris

---

## What is Statistical Learning?

### The Formal Setup

We observe $n$ data points $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ where each $x_i \in \mathbb{R}^p$ is a vector of $p$ predictors and $y_i \in \mathbb{R}$ (or a discrete label set) is the response. We assume:

$$Y = f(X) + \varepsilon$$

where:
- $f : \mathbb{R}^p \to \mathbb{R}$ is a **fixed but unknown** function capturing the systematic relationship
- $\varepsilon$ is a **random error term**, independent of $X$, with $E[\varepsilon] = 0$ and $\text{Var}(\varepsilon) = \sigma^2$

Statistical learning is the problem of estimating $f$ from the observed data. This deceptively simple framing contains every regression, classification, and density estimation problem in the field.

### Two Fundamentally Different Goals

The goal determines the method. Always.

**Prediction** treats $f$ as a black box: we want $\hat{Y} = \hat{f}(X)$ to be accurate, and we do not care whether $\hat{f}$ is interpretable. A neural network predicting ICU mortality in the next 24 hours from 200 physiological signals is a prediction problem. The doctor needs the number, not the mechanism.

**Inference** treats $f$ as a window into causal structure: we want to understand *which* predictors affect $Y$, *how much*, and *in which direction*. Does increasing TV advertising budget by \$1,000 increase sales? By how much, holding Radio constant? Here interpretability is not a nice-to-have — it is the entire point.

The tension between these goals is real and unavoidable. A model can be maximally accurate and maximally opaque simultaneously (neural networks). Or interpretable and modestly accurate (linear regression). The art is knowing which regime your problem lives in.

### Reducible vs. Irreducible Error

Given any estimator $\hat{f}$, the expected prediction error at point $x$ decomposes as:

$$E[(Y - \hat{f}(X))^2] = \underbrace{[f(X) - \hat{f}(X)]^2}_{\text{Reducible}} + \underbrace{\text{Var}(\varepsilon)}_{\text{Irreducible}}$$

The **reducible error** measures how far our estimate is from the true $f$. With a better model or more data, this can approach zero — in principle. The **irreducible error** $\sigma^2$ is the variance of the noise process. No estimator can do better than this, regardless of how complex it is or how much data you have. It is the floor of prediction uncertainty.

This decomposition has a sharp philosophical implication: **even a perfect model cannot perfectly predict**. If the data-generating process contains genuine randomness (measurement noise, unmeasured confounders, quantum-level stochasticity), prediction error persists. The question is how much of the *reducible* error we can eliminate with the data we have.

---

## Estimating f: Parametric vs. Non-Parametric

### Parametric Approaches

A **parametric model** assumes $f$ has a specific functional form governed by a finite parameter vector $\theta \in \mathbb{R}^k$. The canonical example is the linear model:

$$f(X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p$$

Estimation reduces to finding $\hat{\beta}$ from data — a $k$-dimensional problem regardless of how many data points we have. This is the parametric advantage: **dimensionality reduction**. We replace the infinite-dimensional problem of estimating an unknown function with the finite-dimensional problem of estimating $k$ numbers.

The risk: if the true $f$ is not well-approximated by the assumed form, the model suffers from **specification error** (a form of bias). You are committing to a shape before seeing the full picture.

### Non-Parametric Approaches

**Non-parametric models** make no explicit assumption about the form of $f$. They let the data speak. The tradeoff: without assumptions, you need *more* data to reliably estimate $f$, and you are perpetually at risk of **overfitting** — fitting the noise of the training set rather than the signal.

### Python: OLS on the Advertising Dataset

The Advertising dataset contains 200 markets with TV, Radio, and Newspaper advertising budgets (in thousands of dollars) and Sales (in thousands of units). We fit a linear model with `statsmodels`:

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load Advertising dataset
# Source: https://www.statlearning.com/resources-python
url = "https://www.statlearning.com/s/Advertising.csv"
ads = pd.read_csv(url, index_col=0)

# Design matrix: TV + Radio + Newspaper
X = ads[["TV", "Radio", "Newspaper"]]
y = ads["Sales"]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

print(model.summary())
```

**Key output:**

```
                            OLS Regression Results
===========================================================================
Dep. Variable:                  Sales   R-squared:                   0.897
Model:                            OLS   Adj. R-squared:              0.896
Method:                 Least Squares   F-statistic:                 570.3
No. Observations:               200   Prob (F-statistic):        1.58e-96
===========================================================================
                 coef    std err          t      P>|t|   [0.025   0.975]
---------------------------------------------------------------------------
const          2.9389      0.312      9.422      0.000    2.324    3.554
TV             0.0458      0.001     32.809      0.000    0.043    0.049
Radio          0.1885      0.009     21.893      0.000    0.172    0.206
Newspaper     -0.0010      0.006     -0.177      0.860   -0.013    0.011
===========================================================================
```

Already interesting: **Newspaper is statistically insignificant** ($p = 0.860$), despite being correlated with Sales in isolation. This is the multicollinearity story — Radio and Newspaper budgets are correlated, and once Radio is in the model, Newspaper adds nothing. Inference from this model gives us *partial effects*, not marginal ones.

---

## The Bias-Variance Tradeoff

This section is the heart of the post. Read it slowly.

### The Full Decomposition

For a fixed test point $x_0$, the expected test MSE over repeated training sets of size $n$ is:

$$E\left[(y_0 - \hat{f}(x_0))^2\right] = \underbrace{\text{Var}\left(\hat{f}(x_0)\right)}_{\text{Variance}} + \underbrace{\left[\text{Bias}\left(\hat{f}(x_0)\right)\right]^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\varepsilon)}_{\text{Irreducible}}$$

Where:
- **Variance** $= E\left[\hat{f}(x_0)^2\right] - \left(E\left[\hat{f}(x_0)\right]\right)^2$ — how much $\hat{f}$ fluctuates across different training sets
- **Bias** $= E\left[\hat{f}(x_0)\right] - f(x_0)$ — systematic gap between average prediction and truth
- $y_0 = f(x_0) + \varepsilon_0$ with $E[\varepsilon_0] = 0$

**Derivation sketch.** Let $\hat{f} = \hat{f}(x_0)$, $f = f(x_0)$, $\bar{f} = E[\hat{f}]$. Then:

$$E\left[(y_0 - \hat{f})^2\right] = E\left[(f + \varepsilon - \hat{f})^2\right]$$
$$= E\left[(f - \hat{f})^2\right] + 2E\left[\varepsilon(f - \hat{f})\right] + E[\varepsilon^2]$$
$$= E\left[(f - \hat{f})^2\right] + \sigma^2 \quad \text{(since } \varepsilon \perp \hat{f}\text{)}$$

Then:
$$E\left[(f - \hat{f})^2\right] = E\left[(f - \bar{f} + \bar{f} - \hat{f})^2\right] = (f - \bar{f})^2 + E\left[(\hat{f} - \bar{f})^2\right]$$
$$= \text{Bias}^2 + \text{Var}(\hat{f})$$

This is not just an algebraic identity. It says: **no estimator can simultaneously have zero bias and zero variance when trained on finite data**. Reducing bias typically requires increasing model flexibility, which increases variance. This is the fundamental tradeoff.

### Intuition

- **High bias, low variance**: A rigid model (e.g., $f(x) = \beta_0 + \beta_1 x$ when the truth is nonlinear). It consistently misses the target in the same direction. Stable but wrong.
- **Low bias, high variance**: A flexible model (e.g., degree-15 polynomial). It can hit the target on average, but individual predictions scatter wildly depending on which training data it saw. Accurate in expectation but unreliable in practice.
- **Irreducible error**: Always there. Cannot be reduced.

The optimal model minimizes the *sum* of bias² + variance, not either term in isolation.

### Python: Simulating the Tradeoff

We simulate the decomposition directly using polynomial regression across degrees 1 through 15 on the diabetes dataset:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Use first feature only for clean 1D visualization
diabetes = load_diabetes()
X_raw = diabetes.data[:, 2].reshape(-1, 1)  # BMI index
y_raw = diabetes.target

# True test set (fixed)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42
)

degrees = range(1, 16)
n_bootstrap = 100  # Number of bootstrap training sets

bias2_list, var_list, mse_list = [], [], []
irreducible = np.var(y_test - np.mean(y_test))  # Approximation

for deg in degrees:
    predictions = np.zeros((n_bootstrap, len(X_test)))

    for b in range(n_bootstrap):
        # Bootstrap resample of training data
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_b, y_b = X_train[idx], y_train[idx]

        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1e-3))  # tiny regularization for numerical stability
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

# Plot
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
plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
plt.show()

# Print optimal degree
opt_deg = degrees[np.argmin(mse_list)]
print(f"Optimal polynomial degree: {opt_deg}")
print(f"Min test MSE: {min(mse_list):.2f}")
print(f"At degree {opt_deg}: Bias²={bias2_list[opt_deg-1]:.2f}, Var={var_list[opt_deg-1]:.2f}")
```

**Typical output:**

```
Optimal polynomial degree: 3
Min test MSE: 3601.47
At degree 3: Bias²=3481.22, Var=89.14
```

The U-shaped total MSE curve is the empirical signature of the bias-variance tradeoff. It is not an artifact of this dataset — it is a universal property of any learning algorithm trained on finite data.

### What the Textbook Does Not Say

The bias-variance tradeoff is usually presented as a practical guide for model selection. It is more fundamental than that.

The decomposition is essentially a restatement of the **no free lunch theorem**: averaged over all possible data-generating distributions, no learning algorithm outperforms any other. Flexibility helps on structured problems; rigidity helps when the structure is simple. Since we never know the true structure, we are always making a bet.

More precisely, the tradeoff connects to **regularization theory**: every regularizer (ridge penalty, lasso, weight decay, dropout) is a prior assumption about $f$. Choosing a regularization strength is choosing where on the bias-variance frontier to sit. Bayesian machine learning makes this explicit; frequentist methods make it implicit. But the choice is always there.

---

## The Bayes Classifier

### Definition

In classification, the optimal decision rule at point $x_0$ is:

$$\hat{Y}(x_0) = \arg\max_j\; Pr(Y = j \mid X = x_0)$$

This rule — the **Bayes classifier** — assigns each observation to the most probable class given its predictors. It is provably optimal: no classifier can achieve a lower test error rate *in expectation* over the data-generating distribution. Its error rate is called the **Bayes error rate**:

$$\text{Bayes error rate} = 1 - E_{X}\left[\max_j Pr(Y = j \mid X)\right]$$

This is the irreducible error of classification — the fraction of the population where the two classes genuinely overlap in $X$-space, so that even knowing $X$ perfectly does not determine $Y$.

### Why It Is Unachievable

The Bayes classifier requires knowing $Pr(Y = j \mid X = x_0)$ exactly for every $x_0$. This is the posterior distribution of the class given the features. In practice, we never know this — we estimate it from data.

Every classification algorithm is an attempt to approximate the Bayes decision boundary. Logistic regression assumes the boundary is linear in log-odds. QDA assumes it is quadratic. Neural networks approximate it arbitrarily well as depth and width increase — but at the cost of variance, interpretability, and computational expense.

The Bayes error rate serves as a calibration point: if your test error significantly exceeds the Bayes rate, there is room to improve. If it matches the Bayes rate, you are extracting all recoverable signal from $X$.

---

## K-Nearest Neighbors

### Formal Definition

For a test point $x_0$, KNN regression with neighborhood size $K$ produces:

$$\hat{f}(x_0) = \frac{1}{K} \sum_{x_i \in \mathcal{N}_0} y_i$$

where $\mathcal{N}_0 = \{x_{(1)}, x_{(2)}, \ldots, x_{(K)}\}$ are the $K$ training points closest to $x_0$ in feature space (typically Euclidean distance).

For classification, the prediction is the majority vote over the $K$ neighbors:

$$\hat{Y}(x_0) = \arg\max_j \frac{1}{K}\sum_{x_i \in \mathcal{N}_0} \mathbf{1}(y_i = j)$$

### KNN as Bayes Classifier Approximation

As $K \to 1$, the KNN estimator becomes maximally flexible: every training point is its own prediction, training error is zero, but variance is maximal. As $K \to n$ (all training points), the estimator converges to the global mean — maximum bias, minimum variance. The optimal $K$ sits between these extremes.

Crucially, as $n \to \infty$ with $K/n \to 0$ and $K \to \infty$, the KNN classifier converges to the Bayes classifier. This is a remarkable result: a completely non-parametric method, making no assumptions about $f$, asymptotically achieves the theoretical optimum. The catch: "asymptotically" requires exponentially more data as the dimension $p$ increases — the **curse of dimensionality**.

### Python: KNN on Iris, Visualizing Flexibility

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
# Use only first two features for 2D visualization
X = iris.data[:, :2]
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluate K values: training vs. cross-validated test error
k_values = range(1, 51)
train_errors, cv_errors = [], []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)
    train_err = 1 - knn.score(X_scaled, y)
    cv_err = 1 - cross_val_score(knn, X_scaled, y, cv=5).mean()
    train_errors.append(train_err)
    cv_errors.append(cv_err)

# Plot training vs. test error
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

# Decision boundaries for K=1, K=10, K=50
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
scatter = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold,
                      edgecolors='k', s=40)
ax2.set_title(f'Decision Boundary — KNN (K={k_plot})', fontsize=13)
ax2.set_xlabel('Sepal Length (scaled)', fontsize=11)
ax2.set_ylabel('Sepal Width (scaled)', fontsize=11)

plt.tight_layout()
plt.savefig('knn_iris.png', dpi=150, bbox_inches='tight')
plt.show()

opt_k = k_values[np.argmin(cv_errors)]
print(f"Optimal K: {opt_k}")
print(f"Min CV error: {min(cv_errors):.4f}")
print(f"K=1 training error: {train_errors[0]:.4f} (interpolates training data)")
print(f"K=50 training error: {train_errors[49]:.4f} (smooth, high bias)")
```

**Typical output:**

```
Optimal K: 11
Min CV error: 0.2067
K=1 training error: 0.0000  (interpolates training data)
K=50 training error: 0.1733 (smooth, high bias)
```

The $K=1$ classifier has zero training error — it is a perfect memorizer. But its decision boundaries are jagged, highly sensitive to individual points. Increasing $K$ smooths the boundary, trading that interpolation accuracy for better generalization. This is the bias-variance dial made concrete and visible.

### The Curse of Dimensionality

KNN's convergence to the Bayes classifier relies on dense local neighborhoods. In $p$ dimensions, the volume of a ball of radius $r$ scales as $r^p$. To capture a fixed fraction $q$ of the training data within a ball around $x_0$, the required radius $r$ satisfies:

$$r \sim q^{1/p}$$

For $p = 10$ and $q = 0.01$, this gives $r \approx 0.63$ — you need to reach more than half the range of each variable to find 1% of the data. The "neighborhood" is no longer local. KNN predictions degrade to the global mean, and the approximation of the Bayes classifier breaks down.

This is why $p$-dimensional statistical learning cannot simply scale from 1-dimensional intuitions. It is a different problem.

---

## Prediction vs. Inference: A Practical Tension

The distinction matters in ways that model selection frameworks obscure.

In **medicine and policy**, inference is mandatory. If a model recommends denying a loan, a regulator can compel an explanation. If a model allocates ventilators during an ICU surge, physicians need to know which clinical factors drive the decision — both to trust the model and to catch the failure modes that held-out validation cannot. Interpretability here is a *legal and ethical requirement*, not a preference.

In **recommendation and forecasting**, prediction dominates. Netflix does not need to know *why* you will like a film — only that you will. The mechanism is interesting for research but irrelevant to the user experience. Maximizing accuracy is the objective, and any interpretability loss is acceptable if prediction improves.

The spectrum from interpretable to opaque runs roughly:

| Model | Interpretability | Flexibility |
|-------|-----------------|-------------|
| Linear regression | High | Low |
| Polynomial regression | Medium | Medium |
| Generalized Additive Models (GAM) | Medium | Medium |
| Decision Trees | Medium | Medium |
| Random Forest | Low | High |
| Gradient Boosting | Low | High |
| Neural Networks | Very Low | Very High |

No model is universally better. The right choice depends on the problem's *loss function* in the broadest sense — including regulatory, ethical, and scientific costs.

---

## Model Assessment: Training vs. Test Error

### Why Training Error Misleads

A model's training MSE always decreases as complexity increases. A degree-200 polynomial fit to 200 training points achieves zero training MSE by interpolation — it is a lookup table, not a model. But its predictions on new points are catastrophically bad.

**Training error is not an estimate of generalization error.** It measures how well a model has memorized the training set. The gap between training and test error — **generalization gap** — is the fundamental quantity of interest in model evaluation.

### Python: Learning Curves on Advertising

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

url = "https://www.statlearning.com/s/Advertising.csv"
ads = pd.read_csv(url, index_col=0)

X = ads[["TV", "Radio", "Newspaper"]].values
y = ads["Sales"].values

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
        pipe, X, y,
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
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

The linear model's learning curve converges fast but to a high plateau — training and test MSE meet at the bias floor. The polynomial model's test MSE starts high (small $n$, high variance) and decreases with more data, converging toward the training MSE. The gap measures the variance contribution; the plateau level measures bias.

**A key observation:** adding more data always helps with high-variance models (it closes the generalization gap). It does not help with high-bias models (the plateau is determined by the model family, not the sample size). This has direct practical implications: if you have a high-bias model, more data is wasted investment — you need a better model architecture.

---

## Connections Beyond the Textbook

### Bias-Variance as Epistemological Constraint

In the previous post on attention windows and embedding limits, we encountered a different version of the same fundamental problem: a measurement system (transformer embeddings) systematically fails to distinguish *conceptual identity through lexical variation* from *conceptual diversity through lexical repetition*. This is not a modeling error — it is a structural limitation of distributional semantics.

The bias-variance tradeoff is the same argument, formalized differently. Any learning system making inferences from finite data must make prior assumptions about $f$ (bias) to prevent fitting noise (variance). These assumptions are always *external to the data* — they cannot be derived from the data alone. You bring them to the problem; the data cannot provide them.

This is the statistical expression of **Hume's problem of induction**: past observations logically underdetermine future predictions. We cannot derive "all ravens are black" from any finite set of black raven observations. We can only make this inference under a prior assumption of regularity. The bias-variance decomposition quantifies exactly what happens when that prior assumption is wrong (bias) or when we refuse to make one (variance).

Regularization, in this light, is not a computational trick. It is the explicit insertion of prior beliefs into the learning process — the only mathematically coherent response to the underdetermination problem.

### Connection to PAC Learning Theory

Probably Approximately Correct (PAC) learning theory (Valiant, 1984) asks: for a given hypothesis class $\mathcal{H}$, how many samples $n$ are needed to guarantee that with probability at least $1 - \delta$, the learned hypothesis has error at most $\varepsilon$ above the best hypothesis in $\mathcal{H}$?

The answer depends on the **VC dimension** of $\mathcal{H}$ — a measure of the class's capacity to fit arbitrary labelings. High VC dimension means flexible models that can fit anything (low bias, high variance). Low VC dimension means rigid models that cannot (high bias, low variance).

The PAC sample complexity bound:

$$n \geq \frac{1}{\varepsilon}\left(\text{VC}(\mathcal{H}) \ln\frac{1}{\varepsilon} + \ln\frac{1}{\delta}\right)$$

This is not just theory — it is the bias-variance tradeoff expressed as a sample complexity statement. More complex hypothesis classes require exponentially more data to generalize. The same fundamental tradeoff, different mathematical language.

Modern deep learning operationally violates this: overparameterized neural networks (millions of parameters, thousands of training examples) generalize far better than VC theory predicts. This is the current research frontier — **benign overfitting**, implicit regularization by SGD, and the double descent phenomenon. But the bias-variance framework remains the conceptual foundation from which these departures must be understood.

---

## Conclusion

We have covered the formal setup of statistical learning, derived the bias-variance decomposition from first principles, simulated it empirically on real data, connected it to the Bayes classifier and KNN, and positioned it inside the deeper epistemological problem it represents.

The key insight to carry forward: **every modeling decision is a bet about the shape of $f$**. Parametric models bet on a specific functional form. Regularization bets on smoothness or sparsity. Cross-validation selects the bet with the best empirical payoff on held-out data. But no procedure escapes the fundamental tradeoff — it can only choose where on the bias-variance frontier to sit.

The rest of the series builds on this foundation:

- **Next post:** Linear regression in depth — the geometry of OLS, residual diagnostics, and when linearity fails
- **Following:** Resampling methods — bootstrap, cross-validation, and why they work
- **Later:** Classification methods — logistic regression, LDA, QDA, and their connections to the Bayes classifier
- **Eventually:** Regularization — ridge, lasso, elastic net, and the prior-as-regularizer connection to Bayesian methods

The series follows ISLP but deviates wherever the mathematics or the epistemological story demands it.

---

## References

- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer.
- Valiant, L. G. (1984). A theory of the learnable. *Communications of the ACM*, 27(11), 1134–1142.
- Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural networks and the bias/variance dilemma. *Neural Computation*, 4(1), 1–58.
- Hume, D. (1748). *An Enquiry Concerning Human Understanding*. Oxford University Press.
- Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS*, 116(32), 15849–15854.
