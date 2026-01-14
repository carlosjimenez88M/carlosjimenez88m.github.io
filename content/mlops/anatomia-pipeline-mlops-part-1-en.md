---
title: "Anatomy of an MLOps Pipeline - Part 1: Pipeline and Orchestration"
date: 2026-01-13
draft: false
tags: ["mlops", "machine-learning", "python", "gcp", "mlflow", "wandb"]
categories: ["MLOps", "Engineering"]
author: "Carlos Daniel Jiménez"
description: "Part 1: Philosophy, project architecture and orchestration with Hydra + MLflow. Steps for preprocessing, feature engineering, hyperparameter tuning and model registry."
---

> **Complete MLOps Series:** [Part 1 (current)](/mlops/anatomia-pipeline-mlops-parte-1/) | [Part 2: Deployment →](/mlops/anatomia-pipeline-mlops-parte-2/) | [Part 3: Production →](/mlops/anatomia-pipeline-mlops-parte-3/)

# Anatomy of an MLOps Pipeline - Part 1: Pipeline and Orchestration

## Why This Post Is Not Another Scikit-Learn Tutorial

Most MLOps posts teach you how to train a Random Forest in a notebook and tell you "now put it in production." This post assumes you already know how to train models. What you probably don't know is how to build a system where:

- A GitHub commit triggers a complete 7-step pipeline
- Every preprocessing decision is backed by quantifiable metrics
- Models are versioned with rich metadata, not with filenames like `model_final_v3_REAL.pkl`
- Deployment doesn't require SSH to a server to copy a pickle
- Rollback from a defective version takes 30 seconds, not 3 hours of panic debugging

This post dissects a real pipeline that implements all of that. It's not theory, it's code running in production. Based on Chapter 2 of "Hands-On Machine Learning" by Aurélien Géron, but with the infrastructure the book doesn't cover.

**Complete repository:** [github](https://github.com/carlosjimenez88M/mlops-hand-on-ML-and-pytorch/tree/cap2-end_to_end/cap2-end_to_end)

---

## Table of Contents

1. [The Philosophy: Why Being Organized Is More Important Than Being Smart](#philosophy)
2. [Project Structure: Architecture That Scales](#structure)
3. [Orchestration with Hydra + MLflow](#orchestration)
4. [Step 02: Automated Imputation - Data-Backed Decisions](#step-02)
5. [Step 03: Feature Engineering - KMeans As Feature, Not Just Clustering](#step-03)
6. [Step 06: Hyperparameter Sweep - Bayesian Optimization with W&B](#step-06)
7. [Step 07: Model Registry - Versioning in MLflow](#step-07)
8. [CI/CD with GitHub Actions: Complete Pipeline Automation](#github-actions)
9. [The Value of MLOps: Why This Matters](#mlops-value-proposition)
    - W&B vs MLflow: Why Both, Not One or the Other (#wandb-vs-mlflow)
10. [Docker and MLflow: Complete Ecosystem Containerization](#docker-mlflow)
    - Pipeline Container with MLflow Tracking
    - API Container for Inference
    - Streamlit Container for Frontend
    - Docker Compose: Orchestration of Three Containers
    - API Architecture: FastAPI in Production (#api-architecture)
11. [Model and Parameter Selection Strategies](#model-strategies)
    - Model Selection: Comparison of 5 Algorithms
    - Parameter Grids and GridSearch
    - Evaluation Metrics: MAPE, SMAPE, wMAPE
12. [Testing: Fixtures, Mocking and Real Coverage](#testing)
13. [Production Patterns Nobody Tells You About](#production-patterns)
    - The Transform Pattern: The Synthetic KMeans Trick
    - Training/Serving Skew: The Silent Killer
    - Data Drift: The Enemy This Project (Still) Doesn't Monitor
    - Model Monitoring: Beyond Accuracy
    - The Cascade Pattern: Fallback Resilience
    - Feature Store Anti-Pattern: When You DON'T Need One
    - Production Readiness: An Honest Checklist
14. [Conclusions: MLOps As Engineering Discipline](#conclusions)

---

<a name="philosophy"></a>
## 1. The Philosophy: Why Being Organized Is More Important Than Being Smart

### The Real Problem of MLOps

Being an MLOps engineer has two important things in its work:

**First, and what I feel is most important: being organized.** It sounds redundant, but everything must go in its place. A notebook with 50 cells executed in random order is not a pipeline—it's a time bomb. When that model needs to be retrained at 3 AM because data drift triggered an alert, who remembers the correct order of cells?

**Second: what is not tested, remains a mock or a prototype.** Far from thinking about using only design patterns, the focus and what I will try to plant as a central idea of this post is **the usability of products and seeing this as software design.**

### The Right Mindset

This project treats Machine Learning as what it really is: **software with probabilistic components**. It's not magic, it's engineering. And as engineering, it needs:

- **Versioning:** Of data, code, models and configuration
- **Testing:** Unit, integration and end-to-end
- **Observability:** Logs, metrics and traces
- **Reproducibility:** Running today and in 6 months should give the same result
- **Deployment:** Automated, not manual

### Reference: Hands-On Machine Learning by Géron

This post is based on **Chapter 2 of Géron's book**, a classic we should all read. But the book focuses on the model—how to train a good predictor. This post focuses on the **system around the model**—how to get that predictor into production reliably.

**What Géron teaches:** Data imputation, feature engineering, model selection, evaluation.

**What this post adds:** GCS for storage, W&B for experimentation, MLflow for model registry, FastAPI for serving, Docker for deployment, GitHub Actions for CI/CD.

---

<a name="structure"></a>
## 2. Project Structure: Architecture That Scales

### The Complete Tree (200+ Files)

```
cap2-end_to_end/
├── main.py                                # Hydra + MLflow orchestrator
├── config.yaml                            # Single source of truth
├── pyproject.toml                         # Dependencies with UV
├── Makefile                               # CLI for common operations
├── Dockerfile                             # Containerized pipeline
├── docker-compose.yaml                    # API + Streamlit + MLflow
├── pytest.ini                             # Test configuration
├── .env.example                           # Secrets template
│
├── src/
│   ├── data/                              # Processing steps (01-04)
│   │   ├── 01_download_data/
│   │   │   ├── main.py                    # Download from URL → GCS
│   │   │   ├── downloader.py              # Download logic
│   │   │   ├── models.py                  # Pydantic schemas
│   │   │   ├── MLproject                  # MLflow entry point
│   │   │   └── conda.yaml                 # Isolated dependencies
│   │   │
│   │   ├── 02_preprocessing_and_imputation/
│   │   │   ├── main.py
│   │   │   ├── preprocessor.py
│   │   │   ├── imputation_analyzer.py     # (critical) Strategy comparison
│   │   │   └── utils.py
│   │   │
│   │   ├── 03_feature_engineering/
│   │   │   ├── main.py
│   │   │   ├── feature_engineer.py        # (critical) KMeans clustering
│   │   │   └── utils.py                   # Optimize n_clusters
│   │   │
│   │   └── 04_segregation/
│   │       ├── main.py
│   │       ├── segregator.py              # Train/test split
│   │       └── models.py
│   │
│   ├── model/                             # Modeling steps (05-07)
│   │   ├── 05_model_selection/
│   │   │   ├── main.py                    # Comparison of 5 algorithms
│   │   │   ├── model_selector.py          # (critical) GridSearch per model
│   │   │   └── utils.py
│   │   │
│   │   ├── 06_sweep/
│   │   │   ├── main.py                    # (critical) W&B Bayesian optimization
│   │   │   ├── sweep_config.yaml          # Search space
│   │   │   └── best_params.yaml           # Output (generated)
│   │   │
│   │   └── 07_registration/
│   │       ├── main.py                    # (critical) MLflow registration
│   │       └── configs/
│   │           └── model_config.yaml      # Metadata (generated)
│   │
│   └── utils/
│       └── colored_logger.py              # Structured logging
│
├── api/                                   # FastAPI REST API
│   ├── app/
│   │   ├── main.py                        # FastAPI + lifespan
│   │   ├── core/
│   │   │   ├── config.py                  # Pydantic Settings
│   │   │   ├── model_loader.py            # Load from MLflow/GCS/Local
│   │   │   └── wandb_logger.py            # Log predictions
│   │   ├── models/
│   │   │   └── schemas.py                 # Request/Response schemas
│   │   └── routers/
│   │       └── predict.py                 # POST /api/v1/predict
│   ├── Dockerfile                         # API image (port 8080)
│   └── requirements.txt
│
├── streamlit_app/                         # Interactive frontend
│   ├── app.py                             # Streamlit application (450+ lines)
│   ├── Dockerfile                         # Streamlit image (port 8501)
│   └── requirements.txt
│
├── tests/                                 # Test suite
│   ├── conftest.py                        # Shared fixtures
│   ├── fixtures/
│   │   └── test_data_generator.py         # Synthetic data
│   ├── test_pipeline.py                   # Orchestration test
│   ├── test_downloader.py
│   ├── test_preprocessor.py
│   ├── test_imputation_analyzer.py        # (critical) Imputation tests
│   ├── test_feature_engineering.py
│   ├── test_segregation.py
│   └── test_integration_simple.py         # End-to-end
│
└── docs/
    ├── API_ARCHITECTURE_POST.md
    ├── QUICKSTART_GUIDE.md
    └── TESTING_IMPROVEMENTS.md
```

**Files marked with (critical) are the most critical** to understand the architecture.

### Fundamental Architectural Decisions

#### 1. Separation `src/data` vs `src/model`

**Why:** Data steps (01-04) produce **reusable** artifacts—preprocessing, features, splits. Model steps (05-07) **consume** them but can be retrained without rerunning everything upstream.

**Benefit:** If you change hyperparameters, you only rerun 06-07. If you change feature engineering, you rerun 03-07. You don't re-download data every time.

**Cost:** More verbosity, more files. But in real pipelines with multiple data scientists, isolation is worth gold.

#### 2. MLproject + conda.yaml per Step

Each subdirectory is an independent MLflow project:

```yaml
# src/data/02_preprocessing/MLproject
name: preprocessing_and_imputation

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      gcs_input_path: {type: string}
      gcs_output_path: {type: string}
    command: "python main.py --gcs_input_path={gcs_input_path} --gcs_output_path={gcs_output_path}"
```

**Advantages:**
- Isolated dependencies (step 03 uses scikit-learn 1.3, step 06 could use 1.4)
- Independent execution: `mlflow run src/data/02_preprocessing`
- Granular tracking: each step is a separate run

**Disadvantage:** File overhead. But it's the same overhead as having microservices—each with its own Dockerfile.

#### 3. `api/` As Separate Project

The API is not in `src/api/`. It's a sibling project with its own `requirements.txt`, Dockerfile and tests.

**Reason:** The API is deployed **independently** from the pipeline. It doesn't need full pandas, full scikit-learn or W&B client. Only FastAPI, pydantic and the model pickle.

**Result:** Docker image of 200MB vs 1.5GB if you included the entire pipeline.

#### 4. Tests at Root

Tests test the **complete system**, not isolated modules. `test_integration_simple.py` runs the pipeline end-to-end. It doesn't fit conceptually in `src/`.

#### 5. Absence of `notebooks/`

**Deliberate decision.** Notebooks are excellent for exploration, terrible for production. This project prioritizes **reproducibility** over rapid iteration.

If you need to explore, use them locally but **don't commit them**. Notebooks in git are:
- Hard to review (incomprehensible diffs)
- Impossible to test
- Prone to out-of-order execution

---

<a name="orchestration"></a>
## 3. Orchestration with Hydra + MLflow

### Why Not Simple Bash Scripts

Running sequential Python commands works for simple pipelines:

```bash
python src/data/01_download_data/main.py
python src/data/02_preprocessing/main.py
python src/data/03_feature_engineering/main.py
# ...
```

**This approach fails when you need to:**
- Run only specific steps (debugging)
- Change parameters without editing code
- Version configuration alongside code
- Structured logs of what ran with what params
- Track dependencies between steps

**Hydra + MLflow solves all these problems.**

### The Orchestrator: main.py

```python
"""
MLOps Pipeline Orchestrator
Executes steps sequentially using MLflow + Hydra
"""
import os
import sys
import mlflow
import hydra
from omegaconf import DictConfig
from pathlib import Path
import time

def validate_environment_variables() -> None:
    """Fail fast if critical secrets are missing."""
    required_vars = {
        "GCP_PROJECT_ID": "Google Cloud Project ID",
        "GCS_BUCKET_NAME": "GCS Bucket name",
        "WANDB_API_KEY": "Weights & Biases API Key",
    }

    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value in ["your-project-id", "your-key"]:
            missing.append(f"  ERROR: {var}: {description}")

    if missing:
        print("\n" + "="*70)
        print("ERROR: MISSING REQUIRED ENVIRONMENT VARIABLES")
        print("="*70)
        print("\n".join(missing))
        print("\nCreate .env file with:")
        print("  GCP_PROJECT_ID=your-project-id")
        print("  GCS_BUCKET_NAME=your-bucket")
        print("  WANDB_API_KEY=your-key")
        sys.exit(1)

def get_steps_to_execute(config: DictConfig) -> list[str]:
    """Converts execute_steps from config to list."""
    steps = config['main']['execute_steps']
    if isinstance(steps, str):
        return [s.strip() for s in steps.split(',')]
    return list(steps)

def run_step(step_name: str, step_path: Path, entry_point: str, parameters: dict):
    """Executes a step as MLflow project."""
    print(f"\n{'='*70}")
    print(f"EXECUTING: {step_name}")
    print(f"{'='*70}")

    mlflow.run(
        uri=str(step_path),
        entry_point=entry_point,
        env_manager="local",
        parameters=parameters
    )

@hydra.main(config_path='.', config_name="config", version_base="1.3")
def go(config: DictConfig) -> None:
    """Main pipeline entry point."""

    validate_environment_variables()

    mlflow.set_experiment(config['main']['experiment_name'])
    steps_to_execute = get_steps_to_execute(config)
    root_path = Path(__file__).parent

    start_time = time.time()

    try:
        # Step 01: Download Data
        if "01_download_data" in steps_to_execute:
            run_step(
                "01 - Download Data",
                root_path / "src" / "data" / "01_download_data",
                "main",
                {
                    "file_url": config["download_data"]["file_url"],
                    "gcs_output_path": config["download_data"]["gcs_output_path"],
                }
            )

        # ... Steps 02-07 similar pattern ...

        elapsed = time.time() - start_time
        print(f"\nSUCCESS: PIPELINE COMPLETED ({elapsed:.1f}s)")

    except Exception as e:
        print(f"\nERROR: PIPELINE FAILED: {e}")
        raise

if __name__ == "__main__":
    go()
```

### config.yaml: Single Source of Truth

```yaml
main:
  project_name: "housing-mlops-gcp"
  experiment_name: "end_to_end_pipeline"
  execute_steps:
    - "01_download_data"
    - "02_preprocessing_and_imputation"
    - "03_feature_engineering"
    - "04_segregation"
    - "05_model_selection"
    - "06_sweep"
    - "07_registration"

download_data:
  file_url: "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
  gcs_output_path: "data/01-raw/housing.parquet"

preprocessing:
  gcs_input_path: "data/01-raw/housing.parquet"
  gcs_output_path: "data/02-processed/housing_processed.parquet"
  imputation_strategy: "auto"  # Will compare 4 strategies

feature_engineering:
  gcs_input_path: "data/02-processed/housing_processed.parquet"
  gcs_output_path: "data/03-features/housing_features.parquet"
  n_clusters: 10
  optimize_hyperparams: true  # Find best K

segregation:
  gcs_input_path: "data/03-features/housing_features.parquet"
  gcs_train_output_path: "data/04-split/train/train.parquet"
  gcs_test_output_path: "data/04-split/test/test.parquet"
  test_size: 0.2
  target_column: "median_house_value"

model_selection:
  gcs_train_path: "data/04-split/train/train.parquet"
  gcs_test_path: "data/04-split/test/test.parquet"

sweep:
  sweep_count: 50  # 50 Bayesian optimization runs
  metric_name: "mape"
  metric_goal: "minimize"

registration:
  registered_model_name: "housing_price_model"
  model_stage: "Staging"  # Or "Production"
```

### What This Code Does Well

**1. Fail Fast with Environment Validation**

Before spending CPU, verify all secrets exist. The error message includes **instructions** on how to get each value.

```
ERROR: MISSING REQUIRED ENVIRONMENT VARIABLES
===============================================
  ERROR: WANDB_API_KEY: Weights & Biases API Key

Create .env file with:
  WANDB_API_KEY=your-key
```

This saves **frustration**—especially for new contributors.

**2. Selective Execution Without Commenting Code**

You change `config.yaml`:

```yaml
execute_steps: ["03_feature_engineering", "05_model_selection"]
```

And only those steps run. You don't edit Python, you don't comment imports.

**3. Separation Between Orchestration and Logic**

`main.py` doesn't know how to download data or train models. It only knows how to **invoke** scripts that do. Each step can be developed/tested independently.

**4. Structured Logging with Visual Hierarchy**

The separators (`"="*70`) and emojis aren't cosmetic—in a pipeline that runs 2 hours, visual sections allow you to **quickly scan** to find which step failed.

---

<a name="step-02"></a>
## 4. Step 02: Automated Imputation - Data-Backed Decisions

### The Real Problem

California Housing has ~1% of `total_bedrooms` missing. Obvious options:

1. **Drop rows** → lose data
2. **Fill with median** → assume distribution without verification
3. **Fill with KNN** → assume similarity in feature space
4. **Fill with IterativeImputer** → assume modelable relationships

**Question:** Which is better?

**Incorrect answer:** "KNN always works"

**Correct answer:** "I tested all 4, median had RMSE of 0.8, KNN of 0.6, Iterative of 0.5. I use Iterative because it minimizes reconstruction error. Here's the plot in W&B."

### imputation_analyzer.py: The Core

```python
"""
Imputation Analyzer - Automatically compares strategies
Author: Carlos Daniel Jiménez
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

@dataclass
class ImputationResult:
    """Result of an imputation strategy."""
    method_name: str
    rmse: float
    imputed_values: np.ndarray
    imputer: object

class ImputationAnalyzer:
    """
    Analyzes and compares imputation strategies.
    Automatically selects the best based on RMSE.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "total_bedrooms",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.results: Dict[str, ImputationResult] = {}
        self.best_method: str = None
        self.best_imputer: object = None

    def prepare_validation_set(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Creates masked validation set to compare strategies.

        Strategy:
        1. Remove rows with missing target (can't validate against NaN)
        2. Split into train/val
        3. Mask target in val set (simulate missing values)
        4. Save ground truth

        Returns:
            (train_set, val_set_missing, y_val_true)
        """
        housing_numeric = self.df.select_dtypes(include=[np.number])
        housing_known = housing_numeric.dropna(subset=[self.target_column])

        train_set, val_set = train_test_split(
            housing_known,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Mask target in val
        val_set_missing = val_set.copy()
        val_set_missing[self.target_column] = np.nan

        # Ground truth
        y_val_true = val_set[self.target_column].copy()

        return train_set, val_set_missing, y_val_true

    def evaluate_simple_imputer(
        self,
        train_set: pd.DataFrame,
        val_set_missing: pd.DataFrame,
        y_val_true: pd.Series,
        strategy: str = "median"
    ) -> ImputationResult:
        """Evaluates SimpleImputer with given strategy."""
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(train_set)

        val_imputed = imputer.transform(val_set_missing)

        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        return ImputationResult(
            method_name=f"Simple Imputer ({strategy})",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=imputer
        )

    def evaluate_knn_imputer(
        self,
        train_set: pd.DataFrame,
        val_set_missing: pd.DataFrame,
        y_val_true: pd.Series,
        n_neighbors: int = 5
    ) -> ImputationResult:
        """
        Evaluates KNNImputer with scaling.

        CRITICAL: KNN requires scaled features or explodes with overflow.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            # Scale data
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_set)
            val_scaled = scaler.transform(val_set_missing)

            # KNN imputation
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputer.fit(train_scaled)
            val_imputed_scaled = imputer.transform(val_scaled)

            # Inverse scale
            val_imputed = scaler.inverse_transform(val_imputed_scaled)

        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        return ImputationResult(
            method_name=f"KNN Imputer (k={n_neighbors})",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=(scaler, imputer)  # Store tuple!
        )

    def evaluate_iterative_imputer(
        self,
        train_set: pd.DataFrame,
        val_set_missing: pd.DataFrame,
        y_val_true: pd.Series
    ) -> ImputationResult:
        """Evaluates IterativeImputer with RandomForest estimator."""
        estimator = RandomForestRegressor(
            n_jobs=-1,
            random_state=self.random_state
        )
        imputer = IterativeImputer(
            estimator=estimator,
            random_state=self.random_state
        )

        imputer.fit(train_set)
        val_imputed = imputer.transform(val_set_missing)

        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        return ImputationResult(
            method_name="Iterative Imputer (RF)",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=imputer
        )

    def compare_all_methods(self) -> Dict[str, ImputationResult]:
        """Compares all strategies and selects the best."""
        train_set, val_set_missing, y_val_true = self.prepare_validation_set()

        # Evaluate all
        self.results['simple_median'] = self.evaluate_simple_imputer(
            train_set, val_set_missing, y_val_true, strategy="median"
        )

        self.results['simple_mean'] = self.evaluate_simple_imputer(
            train_set, val_set_missing, y_val_true, strategy="mean"
        )

        self.results['knn'] = self.evaluate_knn_imputer(
            train_set, val_set_missing, y_val_true, n_neighbors=5
        )

        self.results['iterative_rf'] = self.evaluate_iterative_imputer(
            train_set, val_set_missing, y_val_true
        )

        # Select best
        best_key = min(self.results, key=lambda k: self.results[k].rmse)
        self.best_method = best_key
        self.best_imputer = self.results[best_key].imputer

        # Print summary
        print("\n" + "="*70)
        print("IMPUTATION METHODS COMPARISON")
        print("="*70)
        for key, result in sorted(self.results.items(), key=lambda x: x[1].rmse):
            status = "[BEST]" if key == best_key else ""
            print(f"  {result.method_name:30s} RMSE: {result.rmse:8.4f} {status}")
        print("="*70)

        return self.results

    def apply_best_imputer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the best imputer to the complete dataset."""
        if self.best_imputer is None:
            raise ValueError("Run compare_all_methods() first")

        df_out = df.copy()
        numeric_df = df_out.select_dtypes(include=[np.number])

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            # Check if it's a tuple (KNN with scaler)
            if isinstance(self.best_imputer, tuple):
                scaler, imputer = self.best_imputer
                numeric_scaled = scaler.transform(numeric_df)
                imputed_scaled = imputer.transform(numeric_scaled)
                imputed_array = scaler.inverse_transform(imputed_scaled)
            else:
                imputed_array = self.best_imputer.transform(numeric_df)

        target_col_idx = numeric_df.columns.get_loc(self.target_column)
        df_out[self.target_column] = imputed_array[:, target_col_idx]

        return df_out

    def create_comparison_plot(self) -> plt.Figure:
        """Creates bar plot comparing RMSE of methods."""
        methods = [r.method_name for r in self.results.values()]
        rmses = [r.rmse for r in self.results.values()]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if i == np.argmin(rmses) else 'skyblue'
                  for i in range(len(rmses))]
        bars = ax.bar(methods, rmses, color=colors)

        ax.set_xlabel('Imputation Method', fontweight='bold')
        ax.set_ylabel('RMSE', fontweight='bold')
        ax.set_title('Comparison of Imputation Strategies', fontsize=14)
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for bar, rmse in zip(bars, rmses):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{rmse:.4f}',
                ha='center',
                va='bottom'
            )

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig
```

### Critical Technical Decisions

#### 1. The Metric: Reconstruction RMSE

**Why RMSE and not MAE?**

MAE treats all errors equally. RMSE penalizes large errors more strongly.

If a method imputes 100 bedrooms when the truth is 3, that's **problematic**. RMSE punishes it more than MAE. In imputation, large errors distort the dataset more than many small errors.

#### 2. The Masked Validation Set

```python
train_set, val_set = train_test_split(housing_known, test_size=0.2)
val_set_missing = val_set.copy()
val_set_missing[self.target_column] = np.nan
y_val_true = val_set[self.target_column].copy()
```

This **trick is critical**. You can't evaluate imputation strategies on real missing values—you don't know the truth. So:

1. Take rows where the target is NOT missing
2. Split into train/val
3. Artificially mask the target in val
4. Compare how well each imputer reconstructs the values you knew

It's **cross-validation for preprocessing**, not just for models.

#### 3. Why KNN Needs Scaling

```python
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_set)
```

KNN calculates euclidean distances between observations. If one feature is in range [0, 1] and another in [0, 10000], **the second dominates completely**.

StandardScaler normalizes everything to mean 0, std 1. Now all features contribute equally.

**IterativeImputer with RandomForest does NOT need scaling**—trees are scale-invariant.

#### 4. The Imputer As Tuple

```python
if isinstance(self.best_imputer, tuple):
    scaler, imputer = self.best_imputer
    # ... apply both
```

If KNN won, you need to save **both the scaler and the imputer**. In production, when new data arrives:

1. Scale with the same scaler fitted in training
2. Apply KNN imputer
3. Inverse transform to return to original scale

Saving only the imputer without the scaler **would break everything**.

### Usage in the Pipeline

```python
# In Step 02 main.py
import wandb
import mlflow

analyzer = ImputationAnalyzer(df, target_column="total_bedrooms")
results = analyzer.compare_all_methods()

# Log to W&B
comparison_plot = analyzer.create_comparison_plot()
wandb.log({
    "imputation/comparison": wandb.Image(comparison_plot),
    "imputation/best_method": analyzer.best_method,
    "imputation/best_rmse": results[analyzer.best_method].rmse,
})

# Apply to complete dataset
housing_clean = analyzer.apply_best_imputer(housing_df)

# Save imputer
import joblib
joblib.dump(analyzer.best_imputer, "artifacts/imputer.pkl")
mlflow.log_artifact("artifacts/imputer.pkl")
```

### What This Achieves

**Without this:** "I used median because that's what everyone does."

**With this:** "I compared 4 strategies. IterativeImputer with RandomForest had 15% lower RMSE than median. Here's the plot in W&B dashboard run `abc123`. The imputer is serialized in MLflow."

Now you have **quantifiable evidence** of why you chose what you chose. Six months later, when someone asks, **the data is there**.

---

<a name="step-03"></a>
## 5. Step 03: Feature Engineering - KMeans As Feature, Not Just Clustering

### The Real Problem

California has strong geographic patterns. Houses in San Francisco behave differently than houses in the central valley. But latitude/longitude as raw features don't capture this well—a linear model can't learn "this area is expensive".

**Solution:** Geographic clustering. But not to segment data, rather to **create a categorical feature**: `cluster_label`.

### ClusterSimilarity: Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Custom transformer for geographic clustering.

    Design: Scikit-learn transformer to integrate into Pipeline.
    """

    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma  # Placeholder for RBF kernel (not currently used)
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        """Fit KMeans on geographic coordinates."""
        self.kmeans_ = KMeans(
            self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        """Transform coordinates to cluster labels."""
        cluster_labels = self.kmeans_.predict(X)
        return np.expand_dims(cluster_labels, axis=1)

    def get_feature_names_out(self, names=None):
        """Returns feature names for Pipeline."""
        return ["cluster_label"]
```

### The Complete Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def create_preprocessing_pipeline(n_clusters=10):
    """
    Creates pipeline that processes:
    - Numeric: impute + scale
    - Categorical: impute + one-hot
    - Geo: clustering
    """

    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_attribs = [
        "longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population",
        "households", "median_income"
    ]

    cat_attribs = ["ocean_proximity"]

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
        ("geo", ClusterSimilarity(n_clusters=n_clusters),
         ["latitude", "longitude"]),
    ])

    return preprocessing
```

### Automatic Optimization of n_clusters

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

def optimize_n_clusters(
    df: pd.DataFrame,
    min_clusters=2,
    max_clusters=20
) -> Tuple[int, Dict]:
    """
    Finds the best K for KMeans using silhouette score.

    Metrics:
    - Silhouette score (0 to 1): Cluster separation. Maximize.
    - Davies-Bouldin index: Internal dispersion vs separation. Minimize.
    """
    geo_features = df[["latitude", "longitude"]].values

    cluster_range = range(min_clusters, max_clusters + 1)
    silhouette_scores = []
    davies_bouldin_scores = []
    inertias = []

    for n in cluster_range:
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=42)
        labels = kmeans.fit_predict(geo_features)

        silhouette_scores.append(silhouette_score(geo_features, labels))
        davies_bouldin_scores.append(davies_bouldin_score(geo_features, labels))
        inertias.append(kmeans.inertia_)

    # Select K with best silhouette
    optimal_n = cluster_range[np.argmax(silhouette_scores)]

    metrics = {
        "optimal_n_clusters": optimal_n,
        "best_silhouette": max(silhouette_scores),
        "cluster_range": list(cluster_range),
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "inertias": inertias,
    }

    return optimal_n, metrics
```

### Visualization: Elbow Method + Silhouette

```python
def create_optimization_plots(metrics: Dict) -> Dict[str, plt.Figure]:
    """Creates K optimization plots."""

    # Plot 1: Elbow Method (Inertia)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(metrics["cluster_range"], metrics["inertias"], 'bo-')
    ax1.axvline(
        metrics["optimal_n_clusters"],
        color='r',
        linestyle='--',
        label=f'Optimal K={metrics["optimal_n_clusters"]}'
    )
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method - KMeans Optimization')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Silhouette Score
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(metrics["cluster_range"], metrics["silhouette_scores"], 'go-')
    ax2.axvline(
        metrics["optimal_n_clusters"],
        color='r',
        linestyle='--'
    )
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs K')
    ax2.grid(True)

    return {"elbow_method": fig1, "silhouette_scores": fig2}
```

### Usage in the Pipeline

```python
# In Step 03 main.py
import wandb
import mlflow
import joblib

# Download data from GCS
df = download_from_gcs(bucket, "data/02-processed/housing_processed.parquet")

# Optimize K
optimal_k, metrics = optimize_n_clusters(df, min_clusters=5, max_clusters=15)

print(f"Optimal K: {optimal_k}")
print(f"   Silhouette: {metrics['best_silhouette']:.4f}")

# Create plots
plots = create_optimization_plots(metrics)

# Log to W&B
wandb.log({
    "optimization/optimal_k": optimal_k,
    "optimization/silhouette": metrics["best_silhouette"],
    "optimization/elbow_plot": wandb.Image(plots["elbow_method"]),
    "optimization/silhouette_plot": wandb.Image(plots["silhouette_scores"]),
})

# Create pipeline with optimal K
preprocessing_pipeline = create_preprocessing_pipeline(n_clusters=optimal_k)

# Fit pipeline
target_column = "median_house_value"
y = df[target_column]
X = df.drop(columns=[target_column])

preprocessing_pipeline.fit(X, y)

# Transform data
X_transformed = preprocessing_pipeline.transform(X)

# Reconstruct DataFrame with target
df_transformed = pd.DataFrame(
    X_transformed,
    columns=preprocessing_pipeline.get_feature_names_out()
)
df_transformed[target_column] = y.values

# Upload to GCS
upload_to_gcs(df_transformed, bucket, "data/03-features/housing_features.parquet")

# Save pipeline
joblib.dump(preprocessing_pipeline, "artifacts/preprocessing_pipeline.pkl")
mlflow.log_artifact("artifacts/preprocessing_pipeline.pkl")
```

### Critical Technical Decisions

#### 1. Why Silhouette Score

**Silhouette score** (range 0 to 1) measures how well separated the clusters are:

- **1.0:** Perfectly separated clusters
- **0.5:** Moderate overlap
- **0.0:** Random clusters

It's **interpretable** and generally correlates well with visual quality of clusters.

**Davies-Bouldin index:** We also calculate it but don't use it for decision—it's more sensitive to outliers.

#### 2. The Obvious Criticism

This code optimizes `n_clusters` based on **clustering metrics**, not on **final model performance**.

A more rigorous approach would be:

```python
for k in range(5, 15):
    pipeline = create_preprocessing_pipeline(n_clusters=k)
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)

    model = RandomForestRegressor()
    model.fit(X_train_transformed, y_train)

    mape = calculate_mape(model, X_test_transformed, y_test)
    # Select K with best MAPE
```

This would take **10x more time** but would be more rigorous.

**Trade-off:** This pipeline prioritizes speed over absolute rigor. For California Housing, silhouette score is good enough. For more complex datasets, consider the full cross-validation approach.

#### 3. handle_unknown="ignore" in OneHotEncoder

```python
OneHotEncoder(handle_unknown="ignore")
```

**Critical for production.** If in training you have categories `["<1H OCEAN", "INLAND", "NEAR BAY"]` but in production `"ISLAND"` arrives (which you didn't see), the encoder:

- **Without `handle_unknown`:** Explodes with ValueError
- **With `handle_unknown="ignore"`:** Generates zero vector for that observation

You lose information for that observation, but the **API doesn't return HTTP 500**.

#### 4. Why Save the Pipeline, Not Just the Model

```python
joblib.dump(preprocessing_pipeline, "artifacts/preprocessing_pipeline.pkl")
```

In production, you need to:

1. Load the pipeline
2. Transform new data
3. Predict with the model

If you only save the model, you don't know:
- What features it expects
- In what order
- What transformations to apply

The pipeline **encapsulates all that**.

### What This Achieves

**Without this:** "I used KMeans with K=10 because I read that 10 clusters is good."

**With this:** "I tested K from 5 to 15. K=8 maximized silhouette score (0.64). Here are the elbow method and silhouette plots. The pipeline with K=8 is serialized in MLflow."

**Quantifiable evidence + reproducible artifact.**

---

<a name="step-06"></a>
## 6. Step 06: Hyperparameter Sweep - Bayesian Optimization with W&B

### The Problem of Model Selection vs Hyperparameter Tuning

Most ML projects make this mistake: they train a Random Forest in a notebook, adjust some hyperparameters until R² looks "good" and declare victory. Three months later, when someone asks "why Random Forest and not XGBoost?", the answer is awkward silence.

**This pipeline separates two phases:**

1. **Model Selection (Step 05):** Compares algorithms with fast GridSearch (5-10 combos per model)
2. **Hyperparameter Sweep (Step 06):** Optimizes the winner with exhaustive Bayesian search (50+ runs)

**Reason:** You don't have time or compute to do exhaustive sweep of 5 algorithms. First you decide **strategy** (which algorithm), then **tactics** (which hyperparameters).

### sweep_config.yaml: The Search Space

```yaml
# =================================================================
# W&B Sweep Configuration for Random Forest
# Author: Carlos Daniel Jiménez
# =================================================================

program: main.py
method: bayes  # Bayesian optimization, not random, not grid

metric:
  name: wmape  # Weighted MAPE (less biased than MAPE)
  goal: minimize

parameters:
  n_estimators:
    min: 50
    max: 500

  max_depth:
    min: 5
    max: 30

  min_samples_split:
    min: 2
    max: 20

  min_samples_leaf:
    min: 1
    max: 10

  max_features:
    values: ['sqrt', 'log2']

# Early stopping: eliminates poor runs early
early_terminate:
  type: hyperband
  min_iter: 10   # Minimum 10 runs before terminating
  eta: 3         # Eliminates 1/3 of poor runs
  s: 2

name: housing-rf-sweep-improved
description: "Optimize Random Forest with wmape + feature tracking"
```

### main.py from Step 06: The Real Sweep

```python
"""
W&B Sweep for Random Forest Hyperparameter Optimization.
Author: Carlos Daniel Jiménez
"""
import argparse
import yaml
import wandb
import logging
from pathlib import Path

from utils import (
    download_data_from_gcs,
    prepare_data,
    train_random_forest,
    evaluate_model,
    log_feature_importances
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level data cache (loaded once, reused in all runs)
_data_cache = {
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "feature_names": None
}

def train():
    """
    Training function called by W&B Sweep agent.
    Executed for each hyperparameter combination.

    Uses module-level cache to avoid reloading data on each run.
    """
    run = wandb.init()
    config = wandb.config

    logger.info("="*70)
    logger.info(f"SWEEP RUN: {run.name}")
    logger.info("="*70)

    try:
        # Prepare parameters
        params = {
            'n_estimators': int(config.n_estimators),
            'max_depth': int(config.max_depth) if config.max_depth else None,
            'min_samples_split': int(config.min_samples_split),
            'min_samples_leaf': int(config.min_samples_leaf),
            'max_features': config.max_features,
            'random_state': 42
        }

        # Train model using cached data
        model = train_random_forest(
            _data_cache["X_train"],
            _data_cache["y_train"],
            params
        )

        # Evaluate model
        metrics = evaluate_model(
            model,
            _data_cache["X_test"],
            _data_cache["y_test"]
        )

        # Log feature importances
        feature_importances = log_feature_importances(
            model,
            _data_cache["feature_names"]
        )

        # Log everything to W&B
        wandb.log({
            **params,
            **metrics,
            **{f"feature_importance_{k}": v
               for k, v in list(feature_importances.items())[:10]}
        })

        logger.info(f"SUCCESS: MAPE={metrics['mape']:.2f}% | "
                   f"WMAPE={metrics['wmape']:.2f}%")

    except Exception as e:
        logger.error(f"ERROR: Run failed: {str(e)}")
        wandb.log({
            "error": str(e),
            "mape": 999.9,
            "wmape": 999.9
        })
        raise

    finally:
        run.finish()

def main():
    """Main function to initialize and run the sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_train_path", type=str, required=True)
    parser.add_argument("--gcs_test_path", type=str, required=True)
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--target_column", type=str, default="median_house_value")
    parser.add_argument("--sweep_count", type=int, default=50)
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml")
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("W&B SWEEP - HYPERPARAMETER OPTIMIZATION")
    logger.info("="*70)

    # Load data ONCE into module-level cache
    logger.info("\nLoading data into cache...")
    train_df = download_data_from_gcs(args.bucket_name, args.gcs_train_path)
    X_train, y_train = prepare_data(train_df, args.target_column)

    test_df = download_data_from_gcs(args.bucket_name, args.gcs_test_path)
    X_test, y_test = prepare_data(test_df, args.target_column)

    # Store in cache
    _data_cache["X_train"] = X_train
    _data_cache["X_test"] = X_test
    _data_cache["y_train"] = y_train
    _data_cache["y_test"] = y_test
    _data_cache["feature_names"] = X_train.columns.tolist()

    logger.info(f"Data cached: Train {X_train.shape}, Test {X_test.shape}")

    # Load sweep configuration
    sweep_config_path = Path(__file__).parent / args.sweep_config

    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    logger.info(f"\nSweep config:")
    logger.info(f"  Method: {sweep_config['method']}")
    logger.info(f"  Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.wandb_project
    )

    logger.info(f"\nSweep created: {sweep_id}")
    logger.info(f"  View at: https://wandb.ai/{args.wandb_project}/sweeps/{sweep_id}")

    # Run sweep agent
    logger.info(f"\nStarting sweep agent ({args.sweep_count} runs)...")
    wandb.agent(
        sweep_id,
        function=train,
        count=args.sweep_count,
        project=args.wandb_project
    )

    logger.info("\n" + "="*70)
    logger.info("SWEEP COMPLETED")
    logger.info("="*70)

    # Save best params
    api = wandb.Api()
    sweep = api.sweep(f"{args.wandb_project}/{sweep_id}")
    best_run = sweep.best_run()

    best_params = {
        "hyperparameters": {
            "n_estimators": int(best_run.config.get('n_estimators')),
            "max_depth": int(best_run.config.get('max_depth')) if best_run.config.get('max_depth') else None,
            "min_samples_split": int(best_run.config.get('min_samples_split')),
            "min_samples_leaf": int(best_run.config.get('min_samples_leaf')),
            "max_features": best_run.config.get('max_features'),
        },
        "metrics": {
            "mape": float(best_run.summary.get('mape')),
            "wmape": float(best_run.summary.get('wmape')),
            "r2": float(best_run.summary.get('r2')),
        },
        "sweep_id": sweep_id,
        "best_run_id": best_run.id
    }

    # Save to YAML
    best_params_path = Path(__file__).parent / "best_params.yaml"
    with open(best_params_path, 'w') as f:
        yaml.dump(best_params, f)

    logger.info(f"\nBest params saved to: {best_params_path}")
    logger.info(f"   MAPE: {best_params['metrics']['mape']:.2f}%")

if __name__ == "__main__":
    main()
```

### Critical Technical Decisions

#### 1. Bayesian Optimization, Not Random Search

```yaml
method: bayes  # Not random, not grid
```

**Random search:** Tests random combinations. Doesn't learn from previous runs.

**Grid search:** Tests all combinations. Exhaustive but **expensive** (5 × 4 × 3 × 3 × 2 = 360 combos).

**Bayesian optimization:** Builds a probabilistic model of the function you're optimizing (MAPE as a function of hyperparameters) and uses that model to decide what to test next.

If it detects that `max_depth=None` consistently gives better MAPE, **it explores more in that region** of the space.

**50 runs is <15% of the total space**, but captures 80% of the possible benefit.

#### 2. wMAPE, Not MAPE

```yaml
metric:
  name: wmape  # Weighted MAPE
```

**Standard MAPE:** Penalizes errors on cheap houses more than on expensive houses.

If a house is worth $10,000 and you predict $12,000, error = 20%.
If a house is worth $500,000 and you predict $510,000, error = 2%.

Both errors are **$10,000**, but MAPE sees them radically different.

**wMAPE (Weighted MAPE):** Weights by actual value. Less biased toward low values.

**Why it works here:** California Housing doesn't have $0 houses. Range is between $15k and $500k—reasonably bounded.

#### 3. Global Variables For Data Cache

```python
_data_cache = {
    "X_train": None,
    "X_test": None,
    # ...
}
```

Global variables are generally dirty code. **Here they're the right decision.**

Each sweep run needs the same data. Without cache, you'd load from GCS **50 times**. With California Housing (20k rows), that's wasted seconds. With larger datasets, it's **minutes or hours**.

**"Clean" alternative:** Pass data as argument to each function. But W&B Sweeps has a fixed interface—the function you pass to `wandb.agent()` can't receive additional arguments.

Global variables here have **limited scope**—they only exist during the sweep process.

#### 4. Early Stopping with Hyperband

```yaml
early_terminate:
  type: hyperband
  min_iter: 10
  eta: 3
```

**Hyperband** eliminates poor runs early. If after 10 runs a set of hyperparameters shows MAPE of 25% while others are at 8%, Hyperband **stops it**.

**eta=3:** Eliminates the worst third of runs in each iteration.

**Benefit:** You save compute on obviously bad hyperparameters.

#### 5. Logged Feature Importances

```python
feature_importances = log_feature_importances(model, feature_names)
wandb.log({
    **{f"feature_importance_{k}": v
       for k, v in list(feature_importances.items())[:10]}
})
```

Random Forest calculates feature importances **for free**. It would be valuable to log it to understand which features dominate the model.

In W&B dashboard, you can compare runs and see "in the best run, `median_income` had importance of 0.45".

### The Critical Output: best_params.yaml

**Note:** Below are the **real values** from the actual production sweep, not example values:

```yaml
sweep_id: f73ao31m
best_run_id: 5q1840qa
best_run_name: dry-sweep-5

hyperparameters:
  n_estimators: 128
  max_depth: 23
  min_samples_split: 9
  min_samples_leaf: 9
  max_features: log2
  random_state: 42

metrics:
  # Primary metrics
  mae: 38901.45
  rmse: 53277.02
  r2: 0.78339

  # Percentage error metrics
  mape: 20.4002
  smape: 18.85
  wmape: 19.12
  median_ape: 16.73

  # Accuracy within thresholds
  within_5pct: 12.4
  within_10pct: 36.2
  within_15pct: 52.8

sweep_url: https://wandb.ai/danieljimenez88m-carlosdanieljimenez-com/housing-mlops-gcp/sweeps/f73ao31m
```

**Key insights from these real metrics:**

- **MAPE of 20.4%** indicates the model predicts prices within ±20% on average
- **36.2% of predictions are within 10%** - acceptable for real estate valuation
- **R² of 0.78** means the model explains 78% of variance in house prices
- The optimal configuration found by Bayesian optimization used:
  - Moderate tree depth (`max_depth=23`) to balance bias/variance
  - Higher leaf requirements (`min_samples_leaf=9`) to prevent overfitting
  - `log2` feature sampling for better generalization than `sqrt`

Optimal hyperparameters are saved in **YAML**, not pickle. Reason:

**YAML is readable and git-friendly.** If in the next retraining you change from `n_estimators=128` to `n_estimators=150`, a `git diff` shows it clearly.

With pickle, it's an **opaque binary blob**.

### What This Achieves

**Without this:** "I used `n_estimators=100` because it's scikit-learn's default."

**With this:** "I ran Bayesian sweep of 50 runs. Optimal config: `n_estimators=128, max_depth=23, max_features=log2`. Final MAPE: 20.4% with 36.2% of predictions within 10%. Here's the sweep in W&B: [f73ao31m](https://wandb.ai/danieljimenez88m-carlosdanieljimenez-com/housing-mlops-gcp/sweeps/f73ao31m)."

**Quantifiable evidence** of why you chose each hyperparameter.

---

<a name="step-07"></a>
## 7. Step 07: Model Registry - Versioning in MLflow

### Why Just Saving the Pickle Isn't Enough

The temptation is:

```python
import joblib
joblib.dump(model, "best_model.pkl")
```

This works until you need to answer:

- What hyperparameters did it use?
- What data was it trained on?
- What metrics did it achieve?
- How do I rollback to the previous version?

**MLflow Model Registry** solves this.

### register_model_to_mlflow(): The Core

```python
"""
Model registration in MLflow Model Registry.
Author: Carlos Daniel Jiménez
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def register_model_to_mlflow(
    model,
    model_name: str,
    model_stage: str,
    params: dict,
    metrics: dict,
    feature_columns: list,
    target_column: str,
    gcs_train_path: str,
    gcs_test_path: str
) -> tuple:
    """
    Registers model in MLflow Model Registry with rich metadata.

    Args:
        model: Trained sklearn model
        model_name: Name for registered model
        model_stage: Stage (Staging/Production)
        params: Hyperparameters
        metrics: Evaluation metrics
        feature_columns: List of features
        target_column: Target column name
        gcs_train_path: Path to training data
        gcs_test_path: Path to test data

    Returns:
        (model_uri, model_version, run_id)
    """
    logger.info("="*70)
    logger.info("REGISTERING MODEL TO MLFLOW")
    logger.info("="*70)

    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")
    logger.info(f"Model logged: {model_uri}")

    # Create or get registered model
    try:
        client.create_registered_model(
            name=model_name,
            description="Housing price prediction - Random Forest"
        )
        logger.info(f"Created new registered model: {model_name}")
    except Exception as e:
        if "already exists" in str(e):
            logger.info(f"Model already exists: {model_name}")
        else:
            raise

    # Create model version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )
    logger.info(f"Version created: {model_version.version}")

    # Transition to stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=model_stage  # "Staging" or "Production"
    )
    logger.info(f"Transitioned to: {model_stage}")

    # Create comprehensive description (MARKDOWN)
    description = f"""
# Housing Price Prediction Model

**Algorithm:** Random Forest Regressor

## Hyperparameters
- n_estimators: {params['n_estimators']}
- max_depth: {params.get('max_depth', 'None')}
- min_samples_split: {params['min_samples_split']}
- min_samples_leaf: {params['min_samples_leaf']}
- max_features: {params.get('max_features', 'sqrt')}

## Performance Metrics
- MAPE: {metrics['mape']:.2f}%
- Median APE: {metrics['median_ape']:.2f}%
- Within 10%: {metrics['within_10pct']:.1f}%
- RMSE: {metrics['rmse']:.2f}
- R²: {metrics['r2']:.4f}

## Features
- Number of features: {len(feature_columns)}
- Target: {target_column}

## Data Sources
- Training: {gcs_train_path}
- Testing: {gcs_test_path}
"""

    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=description
    )

    # Add searchable tags
    tags = {
        "algorithm": "RandomForest",
        "framework": "sklearn",
        "mape": f"{metrics['mape']:.2f}",
        "within_10pct": f"{metrics['within_10pct']:.1f}",
        "rmse": f"{metrics['rmse']:.2f}",
        "r2": f"{metrics['r2']:.4f}",
        "n_features": str(len(feature_columns)),
        "target": target_column,
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            model_name,
            model_version.version,
            key,
            value
        )

    logger.info("Tags added to model version")
    logger.info("="*70)

    return model_uri, model_version.version, run_id
```

### Critical Technical Decisions

#### 1. Artifact vs Registered Model

**Artifact:** Pickle saved in a specific run. To use it, you need the `run_id`.

```python
mlflow.sklearn.log_model(model, "model")  # Only artifact
# Usage: mlflow.sklearn.load_model(f"runs://{run_id}/model")
```

**Registered Model:** Versioned with semantic name, stages and metadata.

```python
client.create_model_version(name="housing_price_model", source=model_uri)
# Usage: mlflow.pyfunc.load_model("models:/housing_price_model/Production")
```

In production, your API loads `models:/housing_price_model/Production`, **not `runs:/abc123/model`**.

When you register a new version, you transition it to Production and the deployment **automatically** takes the new version.

#### 2. Rich Metadata in Markdown

```python
description = f"""
# Housing Price Prediction Model

**Algorithm:** Random Forest

## Hyperparameters
- n_estimators: {params['n_estimators']}
...

## Performance Metrics
- MAPE: {metrics['mape']:.2f}%
...
"""
```

This saves **markdown in the model description**. When you open MLflow UI and navigate to `housing_price_model v3`, you see:

- What hyperparameters it used
- What metrics it achieved
- Where the data came from

**Why it's gold:** Six months later, when someone asks "why does model v3 have better MAPE than v2?", you open MLflow and **the answer is there**.

You don't need to search in logs or ask who trained it.

#### 3. Tags For Search

```python
tags = {
    "algorithm": "RandomForest",
    "mape": f"{metrics['mape']:.2f}",
    "r2": f"{metrics['r2']:.4f}",
}

for key, value in tags.items():
    client.set_model_version_tag(model_name, model_version.version, key, value)
```

In MLflow you can **filter models by tags**. "Show me all models with MAPE < 8%" is a query that works if you tagged consistently.

#### 4. Model Config File: Single Source of Truth

```python
model_config = {
    'model': {
        'name': model_name,
        'version': str(model_version),
        'stage': model_stage,
        'parameters': params,
        'metrics': metrics,
        'feature_columns': feature_columns,
        'mlflow_run_id': run_id,
        'sweep_id': sweep_id
    }
}

config_path = Path("configs/model_config.yaml")
with open(config_path, 'w') as f:
    yaml.dump(model_config, f)

mlflow.log_artifact(str(config_path), artifact_path="config")
```

This YAML is logged to MLflow **AND** saved in the repo (in `configs/model_config.yaml`).

**Why YAML and not just MLflow:** Your FastAPI app needs to read configuration at startup. It can do `mlflow.load_model()` for the pickle, but needs to know the **feature names** for input validation.

The YAML is that **single source of truth**.

#### 5. Versioning in Git

When you commit `model_config.yaml`, the diff shows:

```diff
- version: 2
+ version: 3
- mape: 22.1
+ mape: 20.4
- n_estimators: 100
+ n_estimators: 128
+ max_features: log2
```

It's **auditable**. You know exactly what changed between versions.

### The Complete Flow: Sweep → Registration → Production

```bash
# 1. Model Selection (Step 05)
python src/model/05_model_selection/main.py
# Output: "Best: RandomForestRegressor (MAPE: 8.2%)"

# 2. Hyperparameter Sweep (Step 06)
python src/model/06_sweep/main.py --sweep_count=50
# Output: best_params.yaml with optimal hyperparameters

# 3. Model Registration (Step 07)
python src/model/07_registration/main.py --params_file=best_params.yaml
# Output: Model registered in MLflow Registry

# 4. Transition to Production (manual)
mlflow models transition \
  --name housing_price_model \
  --version 3 \
  --stage Production
```

### What This Approach Solves

**Without Model Registry:**
- Pickles in folders: `model_v3_final_FINAL_2.pkl`
- You don't know what hyperparameters each uses
- Rollback = find the correct pickle in GCS

**With Model Registry:**
- Models with semantic versions: v1, v2, v3
- Embedded metadata: params, metrics, data sources
- Rollback = `transition v3 to Archived` + `transition v2 to Production`

---

<a name="github-actions"></a>
## 8. CI/CD with GitHub Actions: Complete Pipeline Automation

---

## Navigation

**[← Home](/mlops/)** | **[Part 2: Deployment and Infrastructure →](/mlops/anatomia-pipeline-mlops-parte-2/)**

In Part 2 we will cover:
- CI/CD with GitHub Actions
- W&B vs MLflow: Complementary strategies
- Complete containerization with Docker
- FastAPI architecture in production
