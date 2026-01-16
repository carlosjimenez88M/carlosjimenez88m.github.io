---
title: "Anatomy of an MLOps Pipeline - Part 2: Deployment and Infrastructure"
date: 2026-01-13
draft: false
tags: ["mlops", "ci-cd", "docker", "fastapi", "github-actions", "wandb", "mlflow"]
categories: ["MLOps", "Engineering"]
author: "Carlos Daniel Jiménez"
description: "Part 2: CI/CD with GitHub Actions, W&B vs MLflow comparison, complete containerization with Docker, and production-ready API architecture with FastAPI."
series: ["Anatomy of an MLOps Pipeline"]
aliases:
  - /mlops/anatomia-pipeline-mlops-part-2-en/
---

> **Complete MLOps Series:** [← Part 1: Pipeline](/post/anatomia-pipeline-mlops-part-1-en/) | **Part 2 (current)** | [Part 3: Production →](/post/anatomia-pipeline-mlops-part-3-en/)

# Anatomy of an MLOps Pipeline - Part 2: Deployment and Infrastructure

<a name="github-actions"></a>
## 8. CI/CD with GitHub Actions: The Philosophy of Automated MLOps

### The Philosophical Foundation: Why Automation Isn't Optional

Before diving into YAML, let's address the fundamental question: **why do we automate ML pipelines?**

The naive answer is "to save time." The real answer is more profound: **because human memory is unreliable, manual processes don't scale, and production systems demand reproducibility.**

Consider this scenario: You trained a model 6 months ago. It's in production. Now there's a bug. You need to retrain with the exact same configuration. Can you do it?

- **Without CI/CD:** You search through Slack messages, local notebooks, commented code. You find 3 different sets of hyperparameters. You don't know which one is the real one. You guess. You're wrong. The model behaves differently. Production breaks.

- **With CI/CD:** You look at the GitHub Actions run from 6 months ago. Every parameter, every dependency version, every data hash is there. You trigger the same workflow. The model is identical. Production is stable.

**This is not about convenience. This is about engineering rigor.**

### The Complete Pipeline: From Commit to Production

This project implements a **single, comprehensive workflow** that orchestrates the entire ML lifecycle. Let's dissect it.

#### The Architecture: 11 Jobs, 1 Pipeline

```
Commit to master/main
    ↓
┌─────────────────────────────────────────────┐
│ 01. SETUP & VALIDATION                      │
│     - Validate project structure            │
│     - Install dependencies with uv          │
│     - Cache for speed                       │
└─────────────────────────────────────────────┘
         ↓
    ┌────────────┬────────────┐
    ↓            ↓            ↓
┌────────┐  ┌────────┐  ┌──────────────┐
│  LINT  │  │ TESTS  │  │ STEP 01:     │
│  CODE  │  │ & COV  │  │ DOWNLOAD     │
└────────┘  └────────┘  └──────────────┘
                             ↓
                        ┌──────────────┐
                        │ STEP 02:     │
                        │ PREPROCESS   │
                        └──────────────┘
                             ↓
                        ┌──────────────┐
                        │ STEP 03:     │
                        │ FEATURE ENG  │
                        └──────────────┘
                             ↓
                        ┌──────────────┐
                        │ STEP 04:     │
                        │ SEGREGATION  │
                        └──────────────┘
                             ↓
                        ┌──────────────┐
                        │ STEP 05:     │
                        │ MODEL SELECT │
                        └──────────────┘
                             ↓
                        ┌──────────────┐
                        │ STEP 06:     │
                        │ SWEEP        │
                        └──────────────┘
                             ↓
                        ┌──────────────┐
                        │ STEP 07:     │
                        │ REGISTRATION │
                        └──────────────┘
                             ↓
                        ┌──────────────┐
                        │ DOCKER BUILD │
                        │ & TEST       │
                        └──────────────┘
                             ↓
                        ┌──────────────┐
                        │ SUMMARY      │
                        └──────────────┘
```

**Key insight:** The pipeline jobs are **sequential where necessary** (you can't train before downloading data) and **parallel where possible** (linting doesn't need to wait for tests).

### Anatomy of the Workflow

Let's examine the critical sections of `.github/workflows/mlops_pipeline.yaml`:

#### 1. Triggers: When Does This Run?

```yaml
on:
  push:
    branches:
      - master
      - main
      - cap2-end_to_end
  pull_request:
    branches:
      - master
      - main
      - cap2-end_to_end
  workflow_dispatch:
    inputs:
      run_full_pipeline:
        description: 'Run complete 7-step pipeline'
        required: true
        default: true
        type: boolean
      sweep_count:
        description: 'Number of sweep runs'
        required: true
        default: '5'
        type: string
```

**Three trigger modes:**

1. **Push to master/main:** Automatic full pipeline on every merge
2. **Pull Request:** Validation before merge (runs linting, tests, but may skip expensive steps)
3. **Manual (workflow_dispatch):** You can trigger manually from GitHub UI with custom parameters

**Philosophy:** Automation doesn't mean "always run everything." It means "make it trivial to run everything when needed."

#### 2. Environment Variables: Single Source of Truth

```yaml
env:
  PYTHON_VERSION: "3.12"
  WANDB_PROJECT: "housing-mlops-gcp"
  WANDB_CONSOLE: "wrap"
  WANDB__SERVICE_WAIT: "300"
  WANDB_MODE: "online"
  WANDB_SILENT: "false"
  WORKING_DIR: "cap2-end_to_end"
```

**Why define these globally?**

- **Python version:** Change once, apply everywhere. When Python 3.13 is stable, you update one line.
- **W&B config:** Consistent logging behavior across all jobs.
- **Working directory:** If the repo structure changes, one update fixes all jobs.

**This is not DRY for the sake of DRY. This is defense against configuration drift.**

#### 3. Job 1: Setup & Validation

```yaml
setup:
  name: Setup and Validate
  runs-on: ubuntu-latest
  outputs:
    python-version: ${{ env.PYTHON_VERSION }}

  steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Validate project structure
      run: |
        echo "Validating project structure..."
        if [ ! -d "${{ env.WORKING_DIR }}" ]; then
          echo "ERROR: Working directory ${{ env.WORKING_DIR }} not found!"
          exit 1
        fi

        if [ ! -f "${{ env.WORKING_DIR }}/pyproject.toml" ]; then
          echo "ERROR: pyproject.toml not found!"
          exit 1
        fi

        if [ ! -f "${{ env.WORKING_DIR }}/main.py" ]; then
          echo "ERROR: main.py not found!"
          exit 1
        fi

        echo "✓ Project structure validated"

    - name: Set up Python ${{ env.PYTHON_VERSION }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cache/pip
          ~/.cache/uv
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install uv package manager
      run: |
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo "$HOME/.cargo/bin" >> $GITHUB_PATH

    - name: Install project dependencies
      working-directory: ${{ env.WORKING_DIR }}
      run: |
        export PATH="$HOME/.cargo/bin:$PATH"
        echo "Installing dependencies..."
        uv pip install --system -e .
        echo "✓ Dependencies installed successfully"
```

**What's happening here:**

1. **Validation first:** Before installing anything, verify the repo structure is correct. Fail fast.
2. **Dependency caching:** `~/.cache/pip` is cached based on `pyproject.toml` hash. If dependencies haven't changed, installation is ~10x faster.
3. **uv package manager:** Traditional `pip` is slow. `uv` is a Rust-based alternative that's 10-100x faster for large dependency trees.

**When to use uv vs pip:**
- **Use uv:** Projects with 50+ dependencies, CI/CD environments, when speed matters
- **Use pip:** Simple projects, maximum compatibility, when you don't control the environment

#### 4. Parallel Validation: Linting & Tests

```yaml
lint-and-format:
  name: Code Quality Checks
  needs: setup
  runs-on: ubuntu-latest

  steps:
    - name: Run Ruff linter
      working-directory: ${{ env.WORKING_DIR }}
      run: |
        echo "Running Ruff linter..."
        ruff check src/ --output-format=github || true
        echo "✓ Linting completed"

unit-tests:
  name: Unit Tests & Coverage
  needs: setup
  runs-on: ubuntu-latest
  if: always()  # Run even if linting fails

  steps:
    - name: Run tests with coverage
      working-directory: ${{ env.WORKING_DIR }}
      run: |
        export PATH="$HOME/.cargo/bin:$PATH"
        uv run pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing || true
        echo "✓ Tests completed"

    - name: Upload coverage
      if: always()
      uses: codecov/codecov-action@v4
      with:
        files: ./${{ env.WORKING_DIR }}/coverage.xml
        flags: unittests
        fail_ci_if_error: false
```
![](img/action2.png)

**Key patterns:**

- **`needs: setup`:** These jobs wait for setup to finish, but run in parallel with each other.
- **`if: always()`:** Tests run even if linting fails. You want to see all failures, not just the first one.
- **`|| true`:** Linting and tests don't fail the pipeline (notice the `|| true`). Why? Because in some orgs, code quality is aspirational, not blocking. Adjust based on your team's culture.

**Philosophical note:** Should bad tests block deployment? **Yes, in mature teams. No, in research-heavy teams where velocity matters more than rigor.** This workflow is configured for research. In production, remove the `|| true`.

#### 5. The ML Pipeline: Steps 01-07

Each ML step is a separate job. Let's examine Step 06 (Hyperparameter Sweep) as representative:

```yaml
step-06-sweep:
  name: Step 6 - Hyperparameter Sweep
  needs: step-05-model-selection
  runs-on: ubuntu-latest
  if: always() && needs.step-05-model-selection.result == 'success'

  steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Configure environment
      run: |
        echo "GCS_BUCKET_NAME=${{ secrets.GCS_BUCKET_NAME }}" >> $GITHUB_ENV
        echo "GCP_PROJECT_ID=${{ secrets.GCP_PROJECT_ID }}" >> $GITHUB_ENV
        echo "WANDB_API_KEY=${{ secrets.WANDB_API_KEY }}" >> $GITHUB_ENV
        echo "WANDB_PROJECT=${{ env.WANDB_PROJECT }}" >> $GITHUB_ENV

    - name: Run Step 06 - Hyperparameter Sweep
      working-directory: ${{ env.WORKING_DIR }}
      run: |
        python main.py main.execute_steps='["06_sweep"]' sweep.sweep_count=${{ github.event.inputs.sweep_count || '5' }}

    - name: Verify best_params.yaml generated
      working-directory: ${{ env.WORKING_DIR }}
      run: |
        if [ -f "src/model/06_sweep/best_params.yaml" ]; then
          echo "✓ best_params.yaml generated successfully"
          cat src/model/06_sweep/best_params.yaml
        else
          echo "ERROR: best_params.yaml not found!"
          exit 1
        fi

    - name: Upload sweep results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: step-06-sweep-results
        path: |
          ${{ env.WORKING_DIR }}/src/model/06_sweep/best_params.yaml
          ${{ env.WORKING_DIR }}/mlruns/
        retention-days: 30
```

**Critical patterns:**

1. **Conditional execution:** `if: always() && needs.step-05-model-selection.result == 'success'`
   - **Translation:** Run this job even if earlier jobs failed (`always()`), but ONLY if step-05 succeeded.
   - **Why:** If model selection failed, there's no point in running hyperparameter sweep. But if linting failed, we still want to see ML results.

2. **Secret management:** `${{ secrets.GCP_SA_KEY }}`
   - Secrets are configured in GitHub Settings → Secrets and variables → Actions
   - They're encrypted at rest and only exposed during job execution
   - **Never log secrets.** GitHub masks them automatically in logs, but be careful with `cat` or `echo`.

3. **Dynamic parameters:** `sweep.sweep_count=${{ github.event.inputs.sweep_count || '5' }}`
   - If triggered manually, use the user-provided `sweep_count`
   - If triggered by push, default to `5`
   - This lets you run quick validation sweeps (sweep_count=3) vs production sweeps (sweep_count=50)

4. **Artifact verification:** The job explicitly checks that `best_params.yaml` was created
   - **Why:** Fail loudly if the step silently failed to produce output
   - Silent failures are the worst kind of failure

5. **Artifact upload:** `actions/upload-artifact@v4`
   - Artifacts are stored by GitHub for 30 days (configurable)
   - Subsequent jobs can download them with `actions/download-artifact@v4`
   - **Use case:** Step 07 (registration) downloads `best_params.yaml` from Step 06

#### 6. Step 07: Model Registration with Artifact Dependency

```yaml
step-07-registration:
  name: Step 7 - Model Registration
  needs: step-06-sweep
  runs-on: ubuntu-latest

  steps:
    - name: Download sweep results
      uses: actions/download-artifact@v4
      with:
        name: step-06-sweep-results
        path: ${{ env.WORKING_DIR }}

    - name: Verify best_params.yaml exists
      working-directory: ${{ env.WORKING_DIR }}
      run: |
        if [ -f "src/model/06_sweep/best_params.yaml" ]; then
          echo "✓ best_params.yaml found"
          cat src/model/06_sweep/best_params.yaml
        else
          echo "ERROR: best_params.yaml not found after download!"
          ls -R src/model/06_sweep/
          exit 1
        fi

    - name: Run Step 07 - Model Registration
      working-directory: ${{ env.WORKING_DIR }}
      run: |
        python main.py main.execute_steps='["07_registration"]'

    - name: Upload registered model artifacts
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: step-07-registered-model
        path: |
          ${{ env.WORKING_DIR }}/models/trained/
          ${{ env.WORKING_DIR }}/configs/model_config.yaml
          ${{ env.WORKING_DIR }}/mlruns/
        retention-days: 90  # Models kept longer than intermediate artifacts
```

**Key insight:** Artifacts enable **stateful pipelines** in stateless environments.

GitHub Actions runners are ephemeral—each job gets a fresh VM. But ML pipelines have dependencies: you can't register a model without hyperparameters from the sweep.

**Artifact flow:**
1. Step 06 uploads `best_params.yaml`
2. GitHub stores it temporarily
3. Step 07 downloads it
4. Step 07 uses it to train the final model
5. Step 07 uploads the trained model (retention: 90 days, because models are more important than intermediate data)

#### 7. Docker Build & Test

```yaml
docker-test:
  name: Docker Build & Test
  needs: step-07-registration
  runs-on: ubuntu-latest

  steps:
    - name: Download model artifacts
      uses: actions/download-artifact@v4
      with:
        name: step-07-registered-model
        path: ${{ env.WORKING_DIR }}

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Build API Docker image
      uses: docker/build-push-action@v5
      with:
        context: ./${{ env.WORKING_DIR }}/api
        push: false  # Don't push to registry, just build and test
        tags: housing-mlops-api:test
        cache-from: type=gha  # Use GitHub Actions cache
        cache-to: type=gha,mode=max

    - name: Test Docker image
      run: |
        docker run -d --name test-api -p 8000:8000 housing-mlops-api:test
        sleep 10
        curl -f http://localhost:8000/health || exit 1
        echo "✓ Docker image works correctly"
        docker stop test-api
```

**Why test Docker in CI?**

**Scenario:** You add a new Python dependency. You update `pyproject.toml`. You test locally. It works. You push to prod. The Docker build fails because you forgot to rebuild the `requirements.txt` that Docker uses.

**With Docker testing in CI:** This fails before it reaches production.

**Buildx caching:** Docker layer caching (via GitHub Actions cache) speeds up builds by ~5-10x. Only changed layers are rebuilt.

#### 8. Pipeline Summary

```yaml
pipeline-summary:
  name: Pipeline Summary
  runs-on: ubuntu-latest
  needs:
    - setup
    - lint-and-format
    - unit-tests
    - step-01-download
    - step-02-preprocessing
    - step-03-feature-engineering
    - step-04-segregation
    - step-05-model-selection
    - step-06-sweep
    - step-07-registration
    - docker-test
  if: always()

  steps:
    - name: Generate summary
      run: |
        echo "## MLOps Pipeline Execution Summary" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "**Branch:** ${{ github.ref_name }}" >> $GITHUB_STEP_SUMMARY
        echo "**Commit:** ${{ github.sha }}" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo "### Pipeline Steps Status" >> $GITHUB_STEP_SUMMARY
        echo "| Step | Status |" >> $GITHUB_STEP_SUMMARY
        echo "|------|--------|" >> $GITHUB_STEP_SUMMARY
        echo "| Setup & Validation | ${{ needs.setup.result }} |" >> $GITHUB_STEP_SUMMARY
        echo "| Code Quality | ${{ needs.lint-and-format.result }} |" >> $GITHUB_STEP_SUMMARY
        echo "| 01 - Download Data | ${{ needs.step-01-download.result }} |" >> $GITHUB_STEP_SUMMARY
        # ... etc

    - name: Check overall status
      run: |
        if [ "${{ needs.step-07-registration.result }}" == "success" ] && \
           [ "${{ needs.docker-test.result }}" == "success" ]; then
          echo "Pipeline completed successfully!"
          exit 0
        else
          echo "Pipeline failed or incomplete"
          exit 1
        fi
```

**Value:** This generates a markdown summary visible in the GitHub Actions UI. One glance tells you:
- Which steps succeeded/failed
- Links to W&B dashboard
- Links to MLflow UI
- Commit SHA (for reproducibility)

**This is the "control tower" view of your pipeline.**

### Version Management: When to Update GitHub Actions

You've probably noticed version tags like `@v4`, `@v5`, `@v2`. Let's address this philosophically.

#### The Actions We Use

| Action | Current Version | Purpose | Update Frequency |
|--------|----------------|---------|------------------|
| `actions/checkout` | `@v4` | Clone the repository | Stable, update yearly |
| `actions/setup-python` | `@v5` | Install Python | Stable, update yearly |
| `actions/cache` | `@v3` | Cache dependencies | Stable, update when cache bugs occur |
| `actions/upload-artifact` | `@v4` | Upload job artifacts | Critical, update when retention/size policies change |
| `actions/download-artifact` | `@v4` | Download job artifacts | Critical, update with upload-artifact |
| `google-github-actions/auth` | `@v2` | GCP authentication | Security-critical, update quarterly |
| `docker/setup-buildx-action` | `@v3` | Set up Docker Buildx | Update when Docker features needed |
| `docker/build-push-action` | `@v5` | Build/push Docker images | Update when new cache strategies available |
| `codecov/codecov-action` | `@v4` | Upload test coverage | Update when Codecov API changes |

#### The Philosophy of Version Pinning

**Two schools of thought:**

1. **"Always use latest" (`@main` or no version tag)**
   - **Pro:** Automatic security patches, new features
   - **Con:** Breaking changes can silently break your pipeline at 3 AM

2. **"Pin to major version" (`@v4`, `@v5`)**
   - **Pro:** Stability, predictable behavior
   - **Con:** You manually update, which means you might miss security patches

**This pipeline uses pinned major versions.** Why?

**Because ML pipelines should break loudly during development, not silently in production.**

#### When to Update Versions

**Update immediately:**
1. **Security vulnerability announced** (GitHub Dependabot alerts you)
2. **Action is deprecated** (GitHub shows warnings in Actions UI)
3. **Bug affecting your workflow** (check action's GitHub Issues)

**Update proactively (quarterly):**
1. Read release notes for major actions (`checkout`, `setup-python`, `auth`)
2. Test in a dev branch
3. If stable, update in main

**Never update blindly:**
- `actions/checkout@v4` → `@v5`: Read breaking changes first
- `google-github-actions/auth@v2` → `@v3`: Authentication changes can break cloud access

**Example update process:**

```bash
# Create update branch
git checkout -b chore/update-actions

# Edit .github/workflows/mlops_pipeline.yaml
# Change: actions/checkout@v4 → actions/checkout@v5

# Push and test
git push origin chore/update-actions
# Watch GitHub Actions run
# If green, merge. If red, investigate.
```

#### Specific Version Recommendations

**`actions/setup-python@v5`:**
- Use `@v5` (current stable as of 2024-2025)
- Update when Python 3.13+ requires it
- Avoid `@v4` (deprecated)

**`google-github-actions/auth@v2`:**
- Use `@v2` (supports Workload Identity Federation)
- Critical for security: avoids long-lived service account keys
- Update when GCP releases new auth methods

**`docker/build-push-action@v5`:**
- Use `@v5` (supports new cache modes)
- Update when you need new BuildKit features
- The `cache-from: type=gha` requires `@v5`

**`actions/cache@v3` vs `@v4`:**
- Use `@v3` (more stable)
- `@v4` exists but has edge cases with cache corruption
- Update when `@v4` is widely adopted (check GitHub's own workflows)

### What CI/CD Solves in MLOps

**Without CI/CD:**
- Manual deployment prone to errors ("Did I use the right Python version?")
- "Works on my machine" syndrome
- Inconsistent testing (someone skips pytest to save time)
- Rollback requires panic debugging at 2 AM
- No history of what was deployed when
- Each team member has their own deployment ritual

**With CI/CD:**
- One command: `git push`
- Tests run automatically (no one can skip them)
- Rollback is `git revert` + `git push`
- Complete audit trail in GitHub Actions UI
- Every deployment is reproducible (same code + same workflow = same result)
- New team members deploy on day 1 without specialized knowledge

### The Real Value: Time Machine for ML Models

Here's the ultimate test of an MLOps system:

**Scenario:** A model deployed 8 months ago is behaving strangely. You need to debug it.

**Question:** Can you recreate the exact model?

**With this CI/CD pipeline:**
1. Go to GitHub Actions → Workflows → MLOps CI/CD Pipeline
2. Find the run from 8 months ago (search by commit SHA or date)
3. Click on it
4. See:
   - Exact Python version (3.12)
   - Exact dependency versions (from `pyproject.toml` in that commit)
   - Exact hyperparameters (from `best_params.yaml` artifact)
   - Exact data hash (logged to W&B)
   - Exact W&B sweep ID (`f73ao31m` in the logs)
5. Download the artifacts
6. Trigger the workflow with `workflow_dispatch`
7. Compare the new run to the old run

**This is time-travel debugging.** You can reproduce any model from any point in history.

**Without this setup:** You search through Slack, local files, old notebooks. You guess. You're wrong. You give up.

### Final Philosophy: Automate Everything You'll Regret Doing Manually

**A good rule of thumb:**

If a task:
1. Must be done more than twice, AND
2. Has more than 3 steps, AND
3. Failures are costly (wrong model in prod, broken deployment)

**→ Automate it.**

This workflow automates:
- Data download
- Preprocessing
- Feature engineering
- Model selection
- Hyperparameter tuning
- Model registration
- Docker packaging
- Deployment verification

**What it doesn't automate (intentionally):**
- Deployment to production (requires manual approval in most orgs)
- Data labeling (still requires human judgment)
- Model interpretability analysis (exploratory, not deterministic)

**The line between automated and manual is not arbitrary—it's where deterministic processes end and human judgment begins.**

---

<a name="mlops-value-proposition"></a>
## 9. The Value of MLOps: Why This Matters

### The Central Question

"Why should I invest time in all this when I can train a model in a notebook in 30 minutes?"

This is the question every MLOps engineer has heard. The short answer: **because notebooks don't scale.**

The long answer is what this section covers.

### The Real Problem: Research Code vs Production Code

#### Research Code (Notebook)

```python
# notebook.ipynb

# Cell 1
import pandas as pd
df = pd.read_csv('housing.csv')

# Cell 2
df = df.dropna()

# Cell 3
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Cell 4
import pickle
pickle.dump(model, open('model.pkl', 'wb'))

# Cell 5
# Wait, did I drop the right columns?
# Let me rerun cell 2... oh no, I ran it twice
# Now I have 0 rows, what happened?
```

**Problems:**
- Not reproducible (execution order matters)
- Not testable
- Not versionable (git diffs are unreadable)
- Not scalable (what happens with 100GB of data?)
- Not auditable (what params did you use?)

#### Production Code (This Pipeline)

```python
# src/model/05_model_selection/main.py

@hydra.main(config_path=".", config_name="config")
def train(config: DictConfig) -> None:
    """Train model with versioned configuration."""

    # Load data from GCS (single source of truth)
    df = load_from_gcs(config.gcs_train_path)

    # Apply serialized preprocessing pipeline
    pipeline = joblib.load('artifacts/preprocessing_pipeline.pkl')
    X = pipeline.transform(df)

    # Train with config params
    model = RandomForestRegressor(**config.hyperparameters)
    model.fit(X, y)

    # Log to MLflow
    mlflow.log_params(config.hyperparameters)
    mlflow.log_metrics(evaluate(model, X_test, y_test))
    mlflow.sklearn.log_model(model, "model")

    return model
```

**Benefits:**
- Reproducible (same config = same output)
- Testable (pure functions, mocking)
- Versionable (readable git diff)
- Scalable (runs locally or on cluster)
- Auditable (MLflow tracking)

### Value #1: Code Modularization

#### Why It Matters

**Scenario:** Your model has a bug in preprocessing. In a notebook, preprocessing is mixed with feature engineering, training, and evaluation in 300 lines.

**In this pipeline:**

```bash
# Bug is in preprocessing → only edit src/data/02_preprocessing/
# Tests fail → pytest tests/test_preprocessor.py
# Fix → re-run only steps 02-07, not 01
```

**Time saved:** Hours per bug.

#### Separation of Concerns

This pipeline separates:

1. **Data steps (01-04):** Produce reusable artifacts
2. **Model steps (05-07):** Consume artifacts, produce models
3. **API:** Consumes models, produces predictions
4. **Frontend:** Consumes API, produces UX

**Benefit:** Teams can work in parallel. The data scientist modifies feature engineering without touching the API. The frontend engineer modifies UI without understanding Random Forests.

### Value #2: Working with Artifacts

#### The Problem: "Where is model_final_v3.pkl?"

Without artifact management:

```
models/
├── model_v1.pkl
├── model_v2.pkl
├── model_final.pkl
├── model_final_FINAL.pkl
├── model_final_REAL.pkl
├── model_production_2024_01_15.pkl  # Is this the production one?
└── model_old_backup.pkl  # Can I delete this?
```

**Problems:**
- You don't know what hyperparameters each one uses
- You don't know what metrics it achieved
- You don't know what data it was trained on
- Rollback = searching for the correct file

#### The Solution: Artifact Storage + Metadata

**1. Google Cloud Storage for data:**

```
gs://bucket-name/
├── data/
│   ├── 01-raw/housing.parquet                    # Immutable
│   ├── 02-processed/housing_processed.parquet    # Versioned by date
│   ├── 03-features/housing_features.parquet
│   └── 04-split/
│       ├── train/train.parquet
│       └── test/test.parquet
├── artifacts/
│   ├── imputer.pkl                               # Preprocessing artifacts
│   ├── preprocessing_pipeline.pkl
│   └── scaler.pkl
└── models/
    └── trained/housing_price_model.pkl           # Latest trained
```

**Benefits:**
- **Immutability:** `01-raw/` never changes, you can always re-run the pipeline
- **Versioning:** Each run has a timestamp, you can compare versions
- **Sharing:** The whole team accesses the same data, not "send me the CSV via Slack"

**2. MLflow for models:**

```python
# Register model
mlflow.sklearn.log_model(model, "model")

# MLflow automatically saves:
# - The model pickle
# - The hyperparameters (n_estimators=200, max_depth=20)
# - The metrics (MAPE=7.8%, R²=0.87)
# - Metadata (date, duration, user)
# - Code (git commit SHA)

# Load model in production
model = mlflow.pyfunc.load_model("models:/housing_price_model/Production")
```

**Benefits:**
- **Semantic versioning:** v1, v2, v3 with stages (Staging/Production)
- **Rich metadata:** You know exactly what each version is
- **Trivial rollback:** `transition v2 to Production`
- **Comparison:** MLflow UI shows table comparing all versions

**3. W&B for experiments:**

```python
# Each sweep run logs:
wandb.log({
    "hyperparameters/n_estimators": 200,
    "hyperparameters/max_depth": 20,
    "metrics/mape": 7.8,
    "metrics/r2": 0.87,
    "plots/feature_importances": wandb.Image(fig),
    "dataset/train_size": 16512,
})

# W&B dashboard:
# - Table with 50 sweep runs
# - Filter by MAPE < 8%
# - Parallel coordinates plot showing relationship between hyperparameters and MAPE
# - Compare top 5 runs side-by-side
```

**Benefits:**
- **Visualization:** Interactive plots of how each hyperparameter affects metrics
- **Collaboration:** Your team sees your experiments in real-time
- **Reproducibility:** Each run has a permanent link with all context

### Value #3: Pipeline Architecture

#### Why A Pipeline, Not A Script

**Single script (run_all.py):**

```python
# run_all.py (500 lines)

def main():
    # Download data
    df = download_data()

    # Preprocess
    df = preprocess(df)

    # Feature engineering
    df = add_features(df)

    # Train model
    model = train_model(df)

    # Deploy
    deploy_model(model)
```

**Problems:**
- If it fails in train_model(), you re-run EVERYTHING (including slow download)
- You can't run just feature engineering to experiment
- Changing preprocessing requires retraining everything
- No intermediate checkpoints

**Modular pipeline:**

```bash
# Run everything
make run-pipeline

# Run only preprocessing
make run-preprocessing

# Run from feature engineering onwards
python main.py main.execute_steps=[03_feature_engineering,04_segregation,05_model_selection]

# Debugging: run only the step that failed
python src/data/03_feature_engineering/main.py --debug
```

**Benefits:**
- **Selective execution:** Only re-run what changed
- **Fast debugging:** Test a step in isolation
- **Parallelization:** Independent steps can run in parallel
- **Checkpointing:** If step 05 fails, steps 01-04 are already done

#### The Contract Between Steps

Each step:
- **Input:** Path to artifact in GCS (example: `data/02-processed/housing_processed.parquet`)
- **Output:** Path to new artifact in GCS (example: `data/03-features/housing_features.parquet`)
- **Side effects:** Logs to MLflow/W&B

```python
# Step 03: Feature Engineering
def run(config):
    # Input
    df = load_from_gcs(config.gcs_input_path)

    # Transform
    df_transformed = apply_feature_engineering(df)

    # Output
    save_to_gcs(df_transformed, config.gcs_output_path)

    # Side effects
    mlflow.log_artifact("preprocessing_pipeline.pkl")
    wandb.log({"optimization/optimal_k": 8})
```

This **contract** allows each step to be:
- Tested independently
- Developed by different people
- Replaced without affecting other steps

### Value #4: Production-Ready vs Research Code

#### Production-Ready Checklist

| Feature | Research Code | This Pipeline |
|---------|---------------|---------------|
| **Versioning** | Git (badly, notebooks) | Git + GCS + MLflow |
| **Testing** | Manual ("ran it once") | pytest + CI |
| **Configuration** | Hardcoded | Versioned YAML |
| **Secrets** | Exposed in code | .env + GitHub Secrets |
| **Logs** | print() statements | Structured logging |
| **Monitoring** | "Hope it works" | W&B + MLflow tracking |
| **Deployment** | Manual | Automatic CI/CD |
| **Rollback** | Panic debugging | Transition in MLflow |
| **Documentation** | Outdated README | Self-documented code + Markdown in MLflow |
| **Collaboration** | "Run these 10 cells in order" | `make run-pipeline` |

#### The Real Cost of Not Doing MLOps

**Scenario:** A model in production has a bug that causes incorrect predictions.

**Without MLOps (Research Code):**
1. Detect the bug: User reports → 2 hours
2. Reproduce the bug: Search for what code/data was used → 4 hours
3. Fix: Run notebook locally → 1 hour
4. Deploy: SSH, copy pickle, restart server → 30 min
5. Verify: Run manual tests → 1 hour
6. **Total: 8.5 hours of downtime**

**With MLOps (This Pipeline):**
1. Detect the bug: Automatic monitoring alerts → 5 min
2. Rollback: `transition v3 to Archived` + `transition v2 to Production` → 2 min
3. Fix: Identify issue with MLflow metadata, fix code → 1 hour
4. Deploy: Push to GitHub → Automatic CI/CD → 10 min
5. Verify: Automatic smoke tests pass → 1 min
6. **Total: 1 hour 18 min of downtime (>85% reduction)**

**Annual savings:** If this happens 4 times a year, you save 29 hours of engineer time.

### Value #5: Data-Driven Decisions

#### The Anti-Pattern

"I used Random Forest with `n_estimators=100` because that's what everyone does."

**Problem:** You have no evidence it's the best choice.

#### This Pipeline

Every decision has quantifiable metrics:

**1. Imputation:**
- Compared 4 strategies (Simple median, Simple mean, KNN, IterativeImputer)
- IterativeImputer won with RMSE=0.52 (vs 0.78 for median)
- Comparison plot in W&B: `wandb.ai/project/run/imputation_comparison`

**2. Feature Engineering:**
- Optimized K from 5 to 15
- K=8 maximized silhouette score (0.64)
- Elbow method plot in W&B

**3. Hyperparameter Tuning:**
- Bayesian sweep of 50 runs
- Optimal config: `n_estimators=200, max_depth=20`
- MAPE improved from 8.5% to 7.8%
- Link to sweep: `wandb.ai/project/sweeps/abc123`

**Benefit:** Six months later, when the stakeholder asks "why do we use this model?", you open W&B/MLflow and the answer is there with plots and metrics.

### The ROI of MLOps

**Initial investment:**
- Setup of GCS, MLflow, W&B, CI/CD: 2-3 days
- Refactoring code to modular pipeline: 1-2 weeks

**Return:**
- Deployment time: 8 hours → 10 minutes (48x faster)
- Debugging time: 4 hours → 30 min (8x faster)
- Onboarding new engineers: 1 week → 1 day
- Team confidence: "Hope it works" → "I know it works"

**For a team of 5 people, breakeven is ~1 month.**

### The Final Lesson For MLOps Engineers

**It's not about the tools.** You can replace:
- GCS → S3 → Azure Blob
- MLflow → Neptune → Comet
- W&B → TensorBoard → MLflow
- GitHub Actions → GitLab CI → Jenkins

**It's about the principles:**

1. **Modularization:** Code in testable modules, not monolithic notebooks
2. **Artifact Management:** Versioned data and models, not `model_final_v3.pkl`
3. **Automation:** CI/CD eliminates toil
4. **Observability:** Logs, metrics, tracking
5. **Reproducibility:** Same input → same output
6. **Data-driven decisions:** Every choice backed by metrics

**When you understand this, you're an MLOps engineer. When you implement it, you're a good MLOps engineer.**

---

<a name="wandb-vs-mlflow"></a>
## 9.5. W&B vs MLflow: Why Both, Not One or the Other

### The Uncomfortable Question

"Why do you have Weights & Biases AND MLflow? Aren't they the same?"

This question reveals a fundamental misunderstanding about what each tool does. They're not competitors—they're **allies with different responsibilities**. Understanding this separates a data scientist who experiments from an MLOps engineer who builds systems.

The short answer: **W&B is your research lab. MLflow is your production line.**

The long answer is what this section covers, with examples from this project's actual code.

---

### The Real Problem: Experimentation vs Governance

#### Phase 1: Experimentation (50-100 runs/day)

When you're in experimentation phase:
- You run 50 sweep runs testing hyperparameter combinations
- You need to see **in real-time** how each run evolves
- You want to visually compare 20 runs simultaneously
- You need to see convergence plots, feature distributions, confusion matrices
- Logging overhead must be minimal (asynchronous logging)

**Correct tool:** Weights & Biases

#### Phase 2: Governance and Deployment (1-2 models/week)

When you promote a model to production:
- You need semantic versioning (v1, v2, v3)
- You need stages (Staging → Production)
- You need rich metadata (what hyperparameters? what data? what commit?)
- You need an API to load models (`models:/housing_price_model/Production`)
- You need trivial rollback (transition v2 to Production)

**Correct tool:** MLflow Model Registry

**The uncomfortable truth:** No tool does both things well.

---

### How This Project Uses W&B

#### 1. Hyperparameter Sweep (Step 06): Bayesian Optimization

```python
# src/model/06_sweep/main.py

# Sweep configuration (Bayesian optimization)
sweep_config = {
    "method": "bayes",  # Bayesian > Grid > Random
    "metric": {
        "name": "wmape",
        "goal": "minimize"
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3
    },
    "parameters": {
        "n_estimators": {"min": 50, "max": 300},
        "max_depth": {"min": 5, "max": 30},
        "min_samples_split": {"min": 2, "max": 20},
        "min_samples_leaf": {"min": 1, "max": 10}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="housing-mlops-gcp")

# Training function that W&B calls 50 times
def train():
    run = wandb.init()  # W&B automatically assigns hyperparameters

    # Get hyperparameters suggested by Bayesian optimizer
    config = wandb.config

    # Train model
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        # ...
    )
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Log to W&B (asynchronous, non-blocking)
    wandb.log({
        "hyperparameters/n_estimators": config.n_estimators,
        "hyperparameters/max_depth": config.max_depth,
        "metrics/mape": metrics['mape'],
        "metrics/wmape": metrics['wmape'],  # Optimizer uses this
        "metrics/r2": metrics['r2'],
        "plots/feature_importances": wandb.Image(fig),
    })

    run.finish()

# Run 50 runs with Bayesian optimization
wandb.agent(sweep_id, function=train, count=50)
```

**What W&B does here that MLflow can't:**

1. **Bayesian Optimization**: W&B suggests the next hyperparameters based on previous runs. It's not random—it uses Gaussian Processes to efficiently explore the space.

   ```
   Run 1: n_estimators=100, max_depth=15 → wMAPE=8.5%
   Run 2: n_estimators=200, max_depth=20 → wMAPE=7.9%  # Better
   Run 3: n_estimators=250, max_depth=22 → wMAPE=7.8%  # W&B suggests values close to Run 2
   ```

2. **Early Termination (Hyperband)**: If a run is performing badly in the first 3 iterations (epochs), W&B kills it automatically and tries other hyperparameters. Saves ~40% of compute.

   ```python
   "early_terminate": {
       "type": "hyperband",
       "min_iter": 3  # Minimum 3 iterations before terminating
   }
   ```

3. **Parallel Coordinates Plot**: Interactive visualization showing which hyperparameter combination produces the best wMAPE.

   ![W&B Parallel Coordinates](https://docs.wandb.ai/assets/images/parallel-coordinates.png)

   **Interpretation:** Blue lines (runs with low wMAPE) converge at `n_estimators=200-250` and `max_depth=20-25`. This visually tells you where the optimum is.

4. **Asynchronous Logging**: `wandb.log()` doesn't block. While the model trains, W&B uploads metrics in the background. Total overhead: <1% of training time.

**MLflow doesn't have:**
- Bayesian optimization (only Grid/Random search via scikit-learn)
- Intelligent early termination
- Parallel coordinates plots
- Asynchronous logging (mlflow.log is synchronous)

---

#### 2. Real-Time Monitoring: See Runs While They Run

```python
# In W&B dashboard (web UI):
# - See 50 simultaneous runs in interactive table
# - Filter by "wmape < 8.0%" → shows only 12 runs
# - Compare top 5 runs side-by-side
# - See convergence plots (MAPE vs iteration)
```

**Real use case:** You start a 50-run sweep at 9 AM. At 10 AM, from your laptop at the coffee shop:
1. Open W&B dashboard
2. See that 30 runs have finished
3. Filter by `wmape < 8.0%` → 8 runs qualify
4. Compare those 8 runs → identify that `max_depth=20` appears in all
5. **Decision:** Cancel the sweep, adjust `max_depth` range to [18, 25], restart

**Value:** Immediate feedback without SSH to the server, without reading terminal logs. Experimentation is **interactive**, not batch.

---

#### 3. Lightweight Artifact Tracking (References to GCS)

```python
# src/model/05_model_selection/main.py

# Upload model to GCS
model_gcs_uri = upload_model_to_gcs(model, "models/05-selection/randomforest_best.pkl")
# gs://bucket/models/05-selection/randomforest_best.pkl

# Log reference in W&B (does NOT upload the pickle, only the URI)
artifact = wandb.Artifact(
    name="best_model_selection",
    type="model",
    description="Best model selected: RandomForest"
)
artifact.add_reference(model_gcs_uri, name="best_model.pkl")  # Only the URI
run.log_artifact(artifact)
```

**W&B doesn't store the model**—it only saves the URI `gs://...`. The model lives in GCS.

**Advantage:** You don't pay for double storage (GCS + W&B). W&B is the index, GCS is the warehouse.

---

### How This Project Uses MLflow

#### 1. Model Registry (Step 07): Versioning and Stages

```python
# src/model/07_registration/main.py

with mlflow.start_run(run_name="model_registration"):
    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log params and metrics
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2
    })
    mlflow.log_metrics({
        "mape": 7.82,
        "r2": 0.8654
    })

    # Register in Model Registry
    client = MlflowClient()

    # Create registered model (if it doesn't exist)
    client.create_registered_model(
        name="housing_price_model",
        description="Housing price prediction - Random Forest"
    )

    # Create new version
    model_version = client.create_model_version(
        name="housing_price_model",
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )
    # Result: housing_price_model/v3

    # Transition to stage
    client.transition_model_version_stage(
        name="housing_price_model",
        version=model_version.version,
        stage="Staging"  # Staging → Production when validated
    )
```

**What MLflow does here that W&B can't:**

1. **Semantic Versioning**: Each model is `housing_price_model/v1`, `v2`, `v3`. They're not random IDs—they're incremental versions.

2. **Stages**: A model goes through `None → Staging → Production → Archived`. This lifecycle is explicit.

   ```
   v1: Production (current in API)
   v2: Staging (being validated)
   v3: None (just trained)
   v4: Archived (deprecated)
   ```

3. **Model-as-Code API**: Loading a model in the API is trivial:

   ```python
   # api/app/core/model_loader.py

   model = mlflow.pyfunc.load_model("models:/housing_price_model/Production")
   ```

   **You don't need to know:**
   - Where the pickle is physically located
   - What version it is (MLflow resolves "Production" → v1)
   - How to deserialize it (mlflow.pyfunc abstracts this)

4. **Rollback in 10 Seconds**:

   ```bash
   # Problem: v3 in Production has a bug
   # Rollback to v2:
   mlflow models transition \
     --name housing_price_model \
     --version 2 \
     --stage Production

   # API detects the change and automatically reloads v2
   ```

5. **Rich Metadata with Tags and Description**:

   ```python
   # Add searchable tags
   client.set_model_version_tag(
       "housing_price_model",
       version,
       "training_date",
       "2026-01-13"
   )
   client.set_model_version_tag(
       "housing_price_model",
       version,
       "sweep_id",
       "abc123xyz"  # Link to W&B sweep
   )

   # Description in Markdown
   client.update_model_version(
       name="housing_price_model",
       version=version,
       description="""
       # Housing Price Model v3

       **Trained:** 2026-01-13
       **Algorithm:** Random Forest
       **Metrics:** MAPE=7.8%, R²=0.865
       **Sweep:** [W&B Link](https://wandb.ai/project/sweeps/abc123)
       """
   )
   ```

   **Result:** 6 months later, when a stakeholder asks "what model is in Production?", you open MLflow UI and all the info is there—not in a lost Slack thread.

**W&B doesn't have:**
- Model Registry (only basic artifact tracking)
- Stages (Staging/Production)
- Load API (`models:/name/stage`)
- Transition history (who changed v2 to Production, when, why)

---

#### 2. Pipeline Orchestration (main.py)

```python
# main.py

@hydra.main(config_path=".", config_name="config")
def go(config: DictConfig) -> None:
    # MLflow orchestrates steps as sub-runs

    # Step 01: Download
    mlflow.run(
        uri="src/data/01_download_data",
        entry_point="main",
        parameters={
            "file_url": config.download_data.file_url,
            "gcs_output_path": config.download_data.gcs_output_path,
            # ...
        }
    )

    # Step 02: Preprocessing
    mlflow.run(
        uri="src/data/02_preprocessing_and_imputation",
        entry_point="main",
        parameters={
            "gcs_input_path": config.preprocessing.gcs_input_path,
            # ...
        }
    )

    # ... Steps 03-07
```

**MLflow creates a hierarchical run:**

```
Parent Run: end_to_end_pipeline
├── Child Run: 01_download_data
│   ├── params: file_url, gcs_output_path
│   └── artifacts: housing.parquet
├── Child Run: 02_preprocessing_and_imputation
│   ├── params: imputation_strategy
│   └── artifacts: imputer.pkl, housing_processed.parquet
├── Child Run: 03_feature_engineering
│   └── ...
└── Child Run: 07_registration
    └── artifacts: model.pkl, model_config.yaml
```

**Value:** In MLflow UI, you see the entire pipeline execution as a tree. Each step is auditable—what params it used, how long it took, what artifacts it produced.

**W&B doesn't have pipeline orchestration**—only tracking of individual runs.

---

### The Division of Labor in This Project

| Responsibility | W&B | MLflow | Reason |
|-----------------|-----|--------|-------|
| **Bayesian hyperparameter optimization** | ✓ | ✗ | W&B has intelligent sweep, MLflow only Grid/Random |
| **Real-time dashboards** | ✓ | ✗ | W&B UI is interactive, MLflow UI is static |
| **Parallel coordinates plots** | ✓ | ✗ | W&B has advanced visualizations |
| **Early termination (Hyperband)** | ✓ | ✗ | W&B implements Hyperband/ASHA/Median stopping |
| **Model Registry with stages** | ✗ | ✓ | MLflow has Staging/Production, W&B doesn't |
| **Model-as-code API** | ✗ | ✓ | `mlflow.pyfunc.load_model()` is the standard |
| **Model rollback** | ✗ | ✓ | MLflow transition, W&B has no stage concept |
| **Pipeline orchestration** | ✗ | ✓ | `mlflow.run()` executes nested steps |
| **Artifact storage (physical)** | ✗ | ✗ | Both point to GCS, don't duplicate storage |
| **Asynchronous logging** | ✓ | ✗ | W&B doesn't block training, MLflow does |
| **Searchable metadata** | ✓ | ✓ | Both allow tags/search, different implementations |

---

### The Complete Flow: W&B → MLflow

**Day 1-3: Experimentation (W&B)**

```bash
# Run 50-run sweep
make run-sweep

# W&B dashboard shows:
# - 50 runs in table
# - Parallel coordinates plot
# - Best run: n_estimators=200, max_depth=20, wMAPE=7.8%
# - Sweep ID: abc123xyz
```

**Output:** `src/model/06_sweep/best_params.yaml`

```yaml
hyperparameters:
  n_estimators: 128
  max_depth: 23
  min_samples_split: 9
  min_samples_leaf: 9
  max_features: log2
metrics:
  mape: 20.4002
  wmape: 19.12
  r2: 0.78339
  within_10pct: 36.2
sweep_id: f73ao31m  # Link to W&B
best_run_id: 5q1840qa
best_run_name: dry-sweep-5
```

**Day 4: Registration (MLflow)**

```bash
# Step 07 reads best_params.yaml
python main.py main.execute_steps=[07_registration]

# MLflow:
# 1. Trains model with best_params
# 2. Registers as housing_price_model/v3
# 3. Transitions to Staging
# 4. Saves metadata (including sweep_id)
```

**Day 5-7: Staging Validation**

```bash
# API runs with model in Staging
docker run -p 8080:8080 \
  -e MLFLOW_MODEL_NAME=housing_price_model \
  -e MLFLOW_MODEL_STAGE=Staging \
  housing-api:latest

# Run tests, validate metrics, review predictions
```

**Day 8: Promotion to Production**

```bash
mlflow models transition \
  --name housing_price_model \
  --version 3 \
  --stage Production

# Production API auto-reloads v3
# v2 becomes fallback (stage: Archived)
```

**If something fails:**

```bash
# Rollback in 10 seconds
mlflow models transition \
  --name housing_price_model \
  --version 2 \
  --stage Production
```

---

### Why Both, Definitely

**Question:** "Can I use only W&B?"

**Answer:** You can, but you lose:
- Model Registry (versioning, stages, rollback)
- Standard API for loading models in production
- Pipeline orchestration with hierarchical runs

**Result:** You end up building your own model versioning system with custom scripts—reinventing the wheel badly.

**Question:** "Can I use only MLflow?"

**Answer:** You can, but you lose:
- Bayesian optimization (you'll have to do slow Grid Search)
- Interactive visualizations (parallel coordinates, real-time dashboards)
- Intelligent early termination (you waste compute)

**Result:** Your sweeps take 3x longer, and you have no visual feedback on what works.

---

### The Real Cost

**W&B:**
- Free tier: 100GB storage, unlimited collaborators
- Team tier: $50/user/month (for teams >5 people)

**MLflow:**
- Open source, free
- Cost: Hosting the tracking server (Cloud Run: ~$20/month for moderate use)
- Storage: GCS (you already pay for data)

**Total for team of 5:** ~$20-50/month (if using W&B free tier) or ~$270/month (if using W&B Team).

**ROI:** If a more efficient sweep saves 30 minutes of compute/day:
- Compute saved: ~15 hours/month
- In GCP: 15 hours × $2/hour (GPU) = $30/month saved in compute alone
- Plus engineer time (more valuable)

**Breakeven in <1 month.**

---

### The Lesson For MLOps Engineers

**Don't choose tools by hype or popularity.** Choose by **clear responsibilities**:

1. **Fast, interactive experimentation:** W&B, Neptune, Comet
2. **Governance and deployment:** MLflow, Seldon, BentoML
3. **Artifact storage:** GCS, S3, Azure Blob (not tracking tools)

**This project uses:**
- **W&B:** Because it needs efficient Bayesian sweep
- **MLflow:** Because it needs production-ready Model Registry
- **GCS:** Because it needs high-availability storage

**There's no redundancy—there's specialization.**

When you understand this, you stop asking "W&B or MLflow?" and start asking "what problem am I solving?"

**That's the difference between using tools and building systems.**

---

<a name="docker-mlflow"></a>
## 10. Docker and MLflow: Containerizing the Complete Ecosystem

### The Three-Container Architecture

This project uses **three distinct Dockerfiles**, each optimized for its specific purpose:

1. **Pipeline Container (`Dockerfile`)**: Runs the complete training pipeline with MLflow tracking
2. **API Container (`api/Dockerfile`)**: Serves predictions with FastAPI in production
3. **Streamlit Container (`streamlit_app/Dockerfile`)**: Provides interactive web interface

This separation is not accidental—it's an architectural decision that reflects the different requirements of each component.
![](img/app1.png)
---

### 1. Pipeline Container: Training with MLflow Tracking

#### Pipeline Dockerfile

```dockerfile
# =================================================================
# Dockerfile for MLOps Pipeline Execution
# Purpose: Run the complete training pipeline in containerized environment
# =================================================================

FROM python:3.12-slim

LABEL maintainer="danieljimenez88m@gmail.com"
LABEL description="Housing Price Prediction - MLOps Pipeline"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    uv pip install --system -e .

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create necessary directories
RUN mkdir -p mlruns outputs models

# Default command runs the pipeline
CMD ["python", "main.py"]
```

#### Critical Technical Decisions

**1. Why `gcc` and `g++`**

```dockerfile
RUN apt-get install -y gcc g++ git curl
```

Many ML packages (numpy, scipy, scikit-learn) compile C/C++ extensions during installation. Without these compilers, `pip install` fails with cryptic errors like "error: command 'gcc' failed".

**Trade-off:** Larger image (~500MB vs ~150MB of pure Python slim), but guarantees all dependencies install correctly.

**2. Layer Caching Strategy**

```dockerfile
# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt* ./
RUN pip install ...

# Copy application code AFTER
COPY . .
```

Docker caches layers. If you change Python code but not dependencies, Docker reuses the `pip install` layer (takes 5 minutes) and only recopies the code (10 seconds).

**Without this optimization:** Every code change requires reinstalling all dependencies.

**3. Directory Creation for MLflow**

```dockerfile
RUN mkdir -p mlruns outputs models
```

MLflow writes artifacts to `mlruns/` by default if a remote tracking server isn't configured. If this directory doesn't exist with correct permissions, MLflow fails silently.

**`outputs/`**: For plots and intermediate analysis
**`models/`**: For model checkpoints before uploading to GCS

#### How to Enable MLflow Tracking

**Option 1: Local MLflow (Default)**

When you run the pipeline in this container, MLflow writes to `mlruns/` inside the container:

```bash
docker run --env-file .env housing-pipeline:latest

# MLflow writes to /app/mlruns/
# To see the UI:
docker exec -it <container-id> mlflow ui --host 0.0.0.0 --port 5000
```

**Limitation:** Runs are lost when the container stops.

**Option 2: MLflow Remote Tracking Server**

To persist runs, configure a separate MLflow server:

```yaml
# docker-compose.yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    container_name: mlflow-server
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=sqlite:///mlflow.db
      - DEFAULT_ARTIFACT_ROOT=gs://your-bucket/mlflow-artifacts
    volumes:
      - mlflow-data:/mlflow
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root gs://your-bucket/mlflow-artifacts
      --host 0.0.0.0
      --port 5000

  pipeline:
    build: .
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - GCP_PROJECT_ID=${GCP_PROJECT_ID}
      - GCS_BUCKET_NAME=${GCS_BUCKET_NAME}
      - WANDB_API_KEY=${WANDB_API_KEY}
    depends_on:
      - mlflow

volumes:
  mlflow-data:
```

**Configuration in code:**

```python
# main.py
import os
import mlflow

# If MLFLOW_TRACKING_URI is configured, use that server
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(mlflow_uri)

mlflow.set_experiment("housing_price_prediction")

with mlflow.start_run():
    # Log params, metrics, artifacts
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("mape", 7.82)
    mlflow.sklearn.log_model(model, "model")
```

**Option 3: MLflow in Cloud (Production)**

For production, use a managed MLflow server:

```bash
# Deploy MLflow to Cloud Run (serverless)
gcloud run deploy mlflow-server \
  --image ghcr.io/mlflow/mlflow:v2.9.2 \
  --platform managed \
  --region us-central1 \
  --set-env-vars="BACKEND_STORE_URI=postgresql://user:pass@host/mlflow_db,DEFAULT_ARTIFACT_ROOT=gs://bucket/mlflow" \
  --allow-unauthenticated

# Get URL
MLFLOW_URL=$(gcloud run services describe mlflow-server --format 'value(status.url)')

# Configure in pipeline
export MLFLOW_TRACKING_URI=$MLFLOW_URL
```

#### Pipeline Container Execution

```bash
# Build
docker build -t housing-pipeline:latest .

# Run with env vars
docker run \
  --env-file .env \
  -v $(pwd)/mlruns:/app/mlruns \
  housing-pipeline:latest

# Run with specific steps
docker run \
  --env-file .env \
  housing-pipeline:latest \
  python main.py main.execute_steps=[03_feature_engineering,05_model_selection]

# See logs in real-time
docker logs -f <container-id>
```

**Volume Mount (`-v`)**: Mounts `mlruns/` from the host to the container to persist MLflow runs even after the container stops.

---

### 2. API Container: Production Inference

#### API Dockerfile

```dockerfile
# =================================================================
# Dockerfile for Housing Price Prediction API
# Purpose: Production-ready FastAPI service for Cloud Run deployment
# =================================================================

FROM python:3.12-slim

LABEL maintainer="danieljimenez88m@gmail.com"
LABEL description="Housing Price Prediction API - FastAPI Service"

WORKDIR /app

# Install system dependencies (only curl for healthcheck)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create models directory
RUN mkdir -p models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
```

#### Critical Technical Decisions

**1. Lighter Image**

Compared to the pipeline container:
- **Doesn't need `gcc`/`g++`**: Dependencies are already compiled in wheels
- **Doesn't need `git`**: Doesn't clone repos
- **Only `curl`**: For the healthcheck

**Result:** Image of ~200MB vs ~500MB for the pipeline.

**Why it matters:** Cloud Run charges for memory usage. A smaller image = less memory = less cost.

**2. Native Health Check**

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

Docker marks the container as "unhealthy" if the `/health` endpoint fails 3 consecutive times.

**Cloud Run** and **Kubernetes** use this to:
- Not send traffic to unhealthy containers
- Restart failing containers
- Uptime reporting

**start-period=40s**: Gives the API 40 seconds to load the model before starting health checks.

**3. Flexible Port Configuration**

```dockerfile
ENV PORT=8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
```

![](img/action.png)

Cloud Run injects `PORT` as an env var (can be 8080, 8081, etc.). The API must read this value, not hardcode it.

**`exec`**: Replaces the shell process with uvicorn, allowing Docker to send signals (SIGTERM) directly to uvicorn for graceful shutdown.

#### How the API Loads the Model

The API has **three model loading strategies** with automatic fallback:

```python
# api/app/core/model_loader.py

class ModelLoader:
    """Load model from MLflow → GCS → Local with fallback."""

    def load_model(self) -> Any:
        """Priority: MLflow > GCS > Local"""

        # Strategy 1: From MLflow Registry
        if self.mlflow_model_name:
            try:
                model_uri = f"models:/{self.mlflow_model_name}/{self.mlflow_stage}"
                self._model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Loaded from MLflow: {model_uri}")
                return self._model
            except Exception as e:
                logger.warning(f"MLflow load failed: {e}, trying GCS...")

        # Strategy 2: From GCS
        if self.gcs_model_path:
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(self.gcs_bucket)
                blob = bucket.blob(self.gcs_model_path)

                model_bytes = blob.download_as_bytes()
                self._model = pickle.loads(model_bytes)
                logger.info(f"Loaded from GCS: gs://{self.gcs_bucket}/{self.gcs_model_path}")
                return self._model
            except Exception as e:
                logger.warning(f"GCS load failed: {e}, trying local...")

        # Strategy 3: From local file (fallback)
        if self.local_model_path and Path(self.local_model_path).exists():
            with open(self.local_model_path, 'rb') as f:
                self._model = pickle.load(f)
            logger.info(f"Loaded from local: {self.local_model_path}")
            return self._model

        raise RuntimeError("No model could be loaded from any source")
```

**Configuration with env vars:**

```bash
# Production: Load from MLflow
docker run -p 8080:8080 \
  -e MLFLOW_TRACKING_URI=https://mlflow.example.com \
  -e MLFLOW_MODEL_NAME=housing_price_model \
  -e MLFLOW_MODEL_STAGE=Production \
  housing-api:latest

# Staging: Load from GCS
docker run -p 8080:8080 \
  -e GCS_BUCKET=my-bucket \
  -e GCS_MODEL_PATH=models/trained/housing_price_model.pkl \
  housing-api:latest

# Development: Load from local
docker run -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -e LOCAL_MODEL_PATH=/app/models/trained/housing_price_model.pkl \
  housing-api:latest
```

---

### 3. Streamlit Container: Interactive Frontend

#### Streamlit Dockerfile

```dockerfile
# =================================================================
# Dockerfile for Streamlit Frontend
# Purpose: Interactive web interface for housing price predictions
# =================================================================

FROM python:3.12-slim

LABEL maintainer="danieljimenez88m@gmail.com"
LABEL description="Housing Price Prediction - Streamlit Frontend"

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Create .streamlit directory for config
RUN mkdir -p .streamlit

# Streamlit configuration
RUN echo '\
[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = "#FF4B4B"\n\
backgroundColor = "#FFFFFF"\n\
secondaryBackgroundColor = "#F0F2F6"\n\
textColor = "#262730"\n\
font = "sans serif"\n\
' > .streamlit/config.toml

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Critical Technical Decisions

**1. Embedded Configuration**

```dockerfile
RUN echo '...' > .streamlit/config.toml
```

Streamlit requires configuration to run in containers (headless mode, CORS, etc.). Instead of committing a `config.toml` file to the repo, we generate it at build time.

**Advantages:**
- One less file in the repo
- Configuration versioned with the Dockerfile
- No risk of forgetting to commit the config

**2. Streamlit Health Check**

```dockerfile
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1
```

Streamlit automatically exposes `/_stcore/health`. This endpoint returns 200 if the app is running.

**3. Custom Theme**

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

The theme defines button colors, backgrounds, etc. This gives visual consistency without needing custom CSS in each component.

#### How Streamlit Connects to the API

```python
# streamlit_app/app.py

import os
import requests
import streamlit as st

# Read API URL from environment variable
API_URL = os.getenv("API_URL", "http://localhost:8080")
API_PREDICT_ENDPOINT = f"{API_URL}/api/v1/predict"

def make_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Call API to get prediction."""
    payload = {"instances": [features]}

    try:
        response = requests.post(
            API_PREDICT_ENDPOINT,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# Streamlit UI
st.title("Housing Price Prediction")

with st.form("prediction_form"):
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    # ... more inputs

    submitted = st.form_submit_button("Predict")

    if submitted:
        features = {
            "longitude": longitude,
            "latitude": latitude,
            # ...
        }

        result = make_prediction(features)

        if result:
            prediction = result["predictions"][0]["median_house_value"]
            st.success(f"Predicted Price: ${prediction:,.2f}")
```

**API URL configuration:**

```bash
# Docker Compose: Use service name
docker-compose up
# Streamlit automatically receives API_URL=http://api:8080

# Local development: Use localhost
API_URL=http://localhost:8080 streamlit run app.py

# Production: Use Cloud Run URL
API_URL=https://housing-api-xyz.run.app streamlit run app.py
```

---

### Docker Compose: Orchestrating the Three Containers

```yaml
# docker-compose.yaml
services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: housing-price-api
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - LOCAL_MODEL_PATH=/app/models/trained/housing_price_model.pkl
      - WANDB_API_KEY=${WANDB_API_KEY}
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mlops-network

  streamlit:
    build:
      context: ./streamlit_app
      dockerfile: Dockerfile
    container_name: housing-streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8080
    depends_on:
      - api
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mlops-network

networks:
  mlops-network:
    driver: bridge
    name: housing-mlops-network
```

**Critical Decisions:**

**1. Network Isolation**

```yaml
networks:
  - mlops-network
```

Both containers are in the same Docker network, allowing Streamlit to call the API using `http://api:8080` (service name as hostname).

**Without this:** You'd have to use `http://host.docker.internal:8080` (only works on Docker Desktop) or the host IP.

**2. Read-Only Volume Mount**

```yaml
volumes:
  - ./models:/app/models:ro
```

The API mounts `models/` in **read-only mode (`:ro`)**. The container can read the model but not modify it.

**Why:** Security. If the container is compromised, an attacker can't overwrite the model with a malicious one.

**3. Dependency Order**

```yaml
depends_on:
  - api
```

Docker Compose starts the API before Streamlit. This prevents Streamlit from failing when trying to connect to an API that's not yet running.

**Limitation:** `depends_on` only waits for the container to **start**, not for the API to be **ready** (healthcheck pass). For that, you need an init container or retry logic in Streamlit.


![](img/app1.png)
![](img/app2.png)
---

### Complete Execution Commands

```bash
# 1. Build all images
docker-compose build

# 2. Train the model (pipeline container)
docker run --env-file .env -v $(pwd)/models:/app/models housing-pipeline:latest

# 3. Start API + Streamlit
docker-compose up -d

# 4. Verify health
curl http://localhost:8080/health
curl http://localhost:8501/_stcore/health

# 5. See logs
docker-compose logs -f

# 6. Stop everything
docker-compose down
```

---

### What This Architecture Solves

**Without containers:**
- "Works on my machine" syndrome
- Conflicting dependencies (Python 3.9 vs 3.12)
- Manual setup in every environment (dev, staging, prod)

**With this architecture:**
- **Reproducibility:** Same container runs on laptop, CI/CD, and production
- **Isolation:** API doesn't interfere with Streamlit, pipeline doesn't interfere with API
- **Deployment:** `docker push` → `gcloud run deploy` in <5 minutes
- **Rollback:** `docker pull previous-image` → restart
- **Observability:** Automatic health checks, centralized logs

**The real value:** A data scientist without DevOps experience can deploy to production without knowing how to configure nginx, systemd, or virtual environments. Docker abstracts all that complexity.

---

<a name="api-architecture"></a>
## 10.5. API Architecture: FastAPI in Production

### Why This Section Matters

You've seen training pipelines, hyperparameter sweeps, and model registry. But **90% of the time, your model isn't training—it's serving predictions in production.**

A poorly designed API is the bottleneck between an excellent model and a useful product. This section breaks down how this project builds a production-ready API, not a tutorial prototype.

---

### The General Architecture

```
api/
├── app/
│   ├── main.py                    # FastAPI app + lifespan management
│   ├── core/
│   │   ├── config.py              # Pydantic Settings (env vars)
│   │   ├── model_loader.py        # Multi-source model loading
│   │   ├── wandb_logger.py        # Prediction logging
│   │   └── preprocessor.py        # Feature engineering
│   ├── routers/
│   │   └── predict.py             # Prediction endpoints
│   └── models/
│       └── schemas.py             # Pydantic request/response models
├── requirements.txt
├── Dockerfile
└── tests/
```

**Architectural decision:** Separation of concerns by layers:

1. **Core**: Business logic (load model, logging, config)
2. **Routers**: HTTP endpoints (routes, request validation)
3. **Models**: Data schemas (Pydantic)

**Why not everything in `main.py`?** Because when the API grows (adding authentication, rate limiting, multiple models), each layer extends independently without touching the rest.

---

### 1. Lifespan Management: The Pattern That Avoids First-Request Latency

#### The Problem It Solves

**Common anti-pattern:**

```python
# BAD: Load model on every request
@app.post("/predict")
def predict(features):
    model = pickle.load(open("model.pkl", "rb"))  # 5 seconds every request
    return model.predict(features)
```

**Problems:**
- First request takes 5 seconds (load model)
- Every subsequent request also (no caching)
- If 10 concurrent requests → 10 model loads (50 seconds total)

#### The Solution: asynccontextmanager

```python
# api/app/main.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Loads the model on startup and cleans up on shutdown.
    """
    logger.info("Starting up API...")

    # STARTUP: Load model ONCE
    wandb_logger = WandBLogger(
        project=settings.WANDB_PROJECT,
        enabled=True
    )

    model_loader = ModelLoader(
        local_model_path=settings.LOCAL_MODEL_PATH,
        gcs_bucket=settings.GCS_BUCKET,
        gcs_model_path=settings.GCS_MODEL_PATH,
        mlflow_model_name=settings.MLFLOW_MODEL_NAME,
        mlflow_model_stage=settings.MLFLOW_MODEL_STAGE,
        mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI
    )

    try:
        logger.info("Loading model...")
        model_loader.load_model()  # Takes 5 seconds, but ONLY once
        logger.info(f"Model loaded: {model_loader.model_version}")

        # Save in app state (available to all endpoints)
        app.state.model_loader = model_loader
        app.state.wandb_logger = wandb_logger

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.warning("API will start but predictions will fail")

    yield  # API runs here

    # SHUTDOWN: Cleanup
    logger.info("Shutting down API...")
    wandb_logger.close()


# Use lifespan in FastAPI
app = FastAPI(
    title="Housing Price Prediction API",
    version="1.0.0",
    lifespan=lifespan  # CRITICAL
)
```

**What it does:**

1. **Startup (before `yield`):**
   - Load model into memory (5 seconds, **only once**)
   - Initialize W&B logger
   - Save both in `app.state` (singleton pattern)

2. **Running (after `yield`):**
   - All requests use the cached model in `app.state.model_loader`
   - Latency per request: <50ms (only inference, no I/O)

3. **Shutdown (after context manager):**
   - Close W&B run (flush pending logs)
   - Free resources

**Result:**
- First request: <50ms (model already loaded)
- Subsequent requests: <50ms
- 10 concurrent requests: <100ms average (parallelizable)

**Trade-off:** 5-10 second startup time. Acceptable for production—better than 5 seconds per request.

---

### 2. Configuration Management: Pydantic Settings with Priorities

#### The Pattern: Settings-as-Code

```python
# api/app/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Housing Price Prediction API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Model - MLflow (priority 1)
    MLFLOW_MODEL_NAME: str = ""
    MLFLOW_MODEL_STAGE: str = "Production"
    MLFLOW_TRACKING_URI: str = ""

    # Model - GCS (priority 2)
    GCS_BUCKET: str = ""
    GCS_MODEL_PATH: str = "models/trained/housing_price_model.pkl"

    # Model - Local (priority 3, fallback)
    LOCAL_MODEL_PATH: str = "models/trained/housing_price_model.pkl"

    # Weights & Biases
    WANDB_API_KEY: str = ""
    WANDB_PROJECT: str = "housing-mlops-api"

    class Config:
        env_file = ".env"  # Read from .env automatically
        case_sensitive = True  # MLFLOW_MODEL_NAME != mlflow_model_name
```

**Why Pydantic Settings:**

1. **Type Safety**: `settings.VERSION` is `str`, not `Optional[Any]`
2. **Validation**: If `MLFLOW_MODEL_STAGE` is not a string, fails at startup (not on first request)
3. **Auto .env loading**: You don't need `python-dotenv` manually
4. **Default values**: `LOCAL_MODEL_PATH` has a default, `MLFLOW_MODEL_NAME` doesn't

**Usage in code:**

```python
from app.core.config import Settings

settings = Settings()  # Read env vars + .env

if settings.MLFLOW_MODEL_NAME:  # Type-safe check
    model = load_from_mlflow(settings.MLFLOW_MODEL_NAME)
```

#### The Priority Strategy (Cascade Fallback)

```
Try to load from:
1. MLflow Registry (if MLFLOW_MODEL_NAME is configured)
   ↓ If fails
2. GCS (if GCS_BUCKET is configured)
   ↓ If fails
3. Local filesystem (always available as last resort)
   ↓ If fails
4. API starts but `/predict` returns 500
```

**Configuration by environment:**

```bash
# Production (.env.production)
MLFLOW_MODEL_NAME=housing_price_model
MLFLOW_MODEL_STAGE=Production
MLFLOW_TRACKING_URI=https://mlflow.company.com
# GCS and Local stay empty → not used

# Staging (.env.staging)
MLFLOW_MODEL_NAME=housing_price_model
MLFLOW_MODEL_STAGE=Staging
# Same setup, different stage

# Local development (.env.local)
LOCAL_MODEL_PATH=models/trained/housing_price_model.pkl
# Without MLflow or GCS → load from local directly
```

**Value:** One codebase, multiple environments. No `if ENVIRONMENT == "production"` in the code.

---

### 3. Model Loader: Multi-Source with Intelligent Fallback

#### The Loader Architecture

```python
# api/app/core/model_loader.py

class ModelLoader:
    """Handles loading ML models from various sources."""

    def __init__(
        self,
        local_model_path: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        gcs_model_path: Optional[str] = None,
        mlflow_model_name: Optional[str] = None,
        mlflow_model_stage: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None
    ):
        self.local_model_path = local_model_path
        self.gcs_bucket = gcs_bucket
        self.gcs_model_path = gcs_model_path
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_stage = mlflow_model_stage
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._model: Optional[Any] = None  # Cached in memory
        self._model_version: str = "unknown"
        self._preprocessor = HousingPreprocessor()

    def load_model(self) -> Any:
        """Load model with cascade fallback strategy."""

        # Priority 1: MLflow Registry
        if self.mlflow_model_name:
            try:
                logger.info(f"Attempting MLflow load: {self.mlflow_model_name}/{self.mlflow_model_stage}")
                self._model = self.load_from_mlflow(
                    self.mlflow_model_name,
                    self.mlflow_model_stage,
                    self.mlflow_tracking_uri
                )
                return self._model
            except Exception as e:
                logger.warning(f"MLflow load failed: {str(e)}, trying GCS...")

        # Priority 2: GCS
        if self.gcs_bucket and self.gcs_model_path:
            try:
                logger.info(f"Attempting GCS load: gs://{self.gcs_bucket}/{self.gcs_model_path}")
                self._model = self.load_from_gcs(self.gcs_bucket, self.gcs_model_path)
                return self._model
            except Exception as e:
                logger.warning(f"GCS load failed: {str(e)}, trying local...")

        # Priority 3: Local filesystem
        if self.local_model_path and Path(self.local_model_path).exists():
            logger.info(f"Attempting local load: {self.local_model_path}")
            self._model = self.load_from_local(self.local_model_path)
            return self._model

        # All strategies failed
        raise RuntimeError(
            "Could not load model from any source. "
            "Check MLflow/GCS/local configuration."
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions with preprocessing."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Apply same preprocessing as the training pipeline
        processed_features = self._preprocessor.transform(features)

        # Predict
        predictions = self._model.predict(processed_features)

        return predictions

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
```

#### Critical Technical Decisions

**1. Why MLflow Is Priority 1**

```python
# MLflow load
model = mlflow.sklearn.load_model("models:/housing_price_model/Production")
```

**Advantages over GCS/Local:**
- **Model URI abstracts storage**: The model can be in S3, GCS, HDFS—MLflow resolves it
- **Stage resolution**: `Production` automatically resolves to the correct version (v1, v2, etc.)
- **Metadata included**: MLflow also loads `conda.yaml`, `requirements.txt`, feature metadata
- **Trivial rollback**: Change stage in MLflow UI, API automatically reloads on next restart

**2. GCS As Fallback (Not Primary)**

```python
# GCS load
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("my-bucket")
blob = bucket.blob("models/trained/housing_price_model.pkl")
model_bytes = blob.download_as_bytes()
model = pickle.loads(model_bytes)
```

**Why not primary:**
- **No versioning:** `models/trained/housing_price_model.pkl` is always the "latest"—you can't load v1 vs v2 without changing the path
- **No metadata:** You only get the pickle, you don't know what hyperparameters/features it expects
- **No stages:** No concept of Staging vs Production

**When to use GCS as primary:**
- MLflow is unavailable (outage)
- Simple setup (only one model, don't need registry)
- Budget constraint (avoid hosting MLflow)

**3. Local As Last Resort**

```python
# Local load
with open("models/trained/housing_price_model.pkl", "rb") as f:
    model = pickle.load(f)
```

**Only for:**
- Local development (don't want to depend on GCS/MLflow)
- Debugging (broken model in GCS, test with local copy)
- CI/CD tests (GitHub Actions doesn't have access to GCS)

**Never for real production**—if GCS and MLflow are down, you have bigger problems than the model.

**4. Embedded Preprocessing Pipeline**

```python
self._preprocessor = HousingPreprocessor()

def predict(self, features: pd.DataFrame) -> np.ndarray:
    processed_features = self._preprocessor.transform(features)
    predictions = self._model.predict(processed_features)
    return predictions
```

**Why critical:** The model expects processed features (one-hot encoding of `ocean_proximity`, cluster feature engineering). If the client sends raw features, the model fails.

**Implementation options:**

**A) Preprocessing in the API (this project):**
```python
# Client sends raw features
{"ocean_proximity": "NEAR BAY", "longitude": -122.23, ...}

# API applies preprocessing
processed = preprocessor.transform(raw_features)

# Model receives processed features
predictions = model.predict(processed)
```

**B) Preprocessing in the client (bad for public APIs):**
```python
# Client must know exact preprocessing
processed = client_side_preprocessing(raw_features)  # What does this do?

# API only does inference
predictions = model.predict(processed)
```

**Trade-offs:**

| Approach | Advantage | Disadvantage |
|----------|---------|------------|
| **Preprocessing in API** | Client doesn't need to know preprocessing | More complex API, +5ms latency |
| **Preprocessing in client** | Simple API, low latency | Client must replicate exact preprocessing |

**For public APIs:** Always preprocessing in the API. Clients shouldn't know internal model details.

**For internal APIs:** Depends. If the client is another service you control, you can do preprocessing there to reduce latency.

---

### 4. Request/Response Validation: Pydantic Schemas

#### The Anti-Pattern: Manual Validation

```python
# BAD: Manual validation prone to errors
@app.post("/predict")
def predict(request: dict):
    if "longitude" not in request:
        return {"error": "missing longitude"}
    if not isinstance(request["longitude"], (int, float)):
        return {"error": "longitude must be number"}
    if request["longitude"] < -180 or request["longitude"] > 180:
        return {"error": "longitude out of range"}
    # ... 50 more lines of manual validation
```

**Problems:**
- Repetitive and fragile code
- Inconsistent errors (`"missing longitude"` vs `"longitude is required"`)
- No automatic documentation (OpenAPI)
- Hard to test

#### The Solution: Pydantic Schemas

```python
# api/app/models/schemas.py

from pydantic import BaseModel, Field, field_validator

class HousingFeatures(BaseModel):
    """Input features for housing price prediction."""

    longitude: float = Field(
        ...,  # Required
        description="Longitude coordinate",
        ge=-180,  # greater or equal
        le=180    # less or equal
    )
    latitude: float = Field(..., description="Latitude coordinate", ge=-90, le=90)
    housing_median_age: float = Field(..., description="Median age of houses", ge=0)
    total_rooms: float = Field(..., description="Total number of rooms", ge=0)
    total_bedrooms: float = Field(..., description="Total number of bedrooms", ge=0)
    population: float = Field(..., description="Block population", ge=0)
    households: float = Field(..., description="Number of households", ge=0)
    median_income: float = Field(..., description="Median income", ge=0)
    ocean_proximity: str = Field(..., description="Proximity to ocean")

    @field_validator('ocean_proximity')
    @classmethod
    def validate_ocean_proximity(cls, v: str) -> str:
        """Validate ocean proximity values."""
        valid_values = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
        if v.upper() not in valid_values:
            raise ValueError(
                f"ocean_proximity must be one of: {', '.join(valid_values)}"
            )
        return v.upper()  # Normalize to uppercase

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }]
        }
    }
```

**What this automatically provides:**

1. **Type validation:**
   ```json
   {"longitude": "not a number"}  // Rejected: ValidationError
   ```

2. **Range validation:**
   ```json
   {"longitude": -200}  // Rejected: must be >= -180
   ```

3. **Custom validation:**
   ```json
   {"ocean_proximity": "INVALID"}  // Rejected: must be one of [...]
   ```

4. **Automatic documentation at `/docs`:**
   - Swagger UI shows all fields
   - Descriptions, constraints, examples
   - Try-it-out works out-of-the-box

5. **Type-safe serialization:**
   ```python
   features = HousingFeatures(**request_json)
   features.longitude  # Type: float (not Optional[Any])
   ```

#### Batch Predictions Support

```python
class PredictionRequest(BaseModel):
    """Request model for single or batch predictions."""

    instances: List[HousingFeatures] = Field(
        ...,
        description="List of housing features for prediction",
        min_length=1  # At least one instance
    )
```

**Usage:**

```json
{
  "instances": [
    {"longitude": -122.23, ...},  // Predict house 1
    {"longitude": -118.45, ...},  // Predict house 2
    {"longitude": -121.89, ...}   // Predict house 3
  ]
}
```


![](img/api.png)

**Why support batch:**
- **Reduced latency:** 3 individual requests = 150ms. 1 batch of 3 = 60ms.
- **Reduced cost:** Less HTTP overhead (headers, handshake, etc.)
- **Efficient inference:** The model can vectorize operations

**Trade-off:** Very large batch size (>1000) can cause timeouts. Implement limit:

```python
instances: List[HousingFeatures] = Field(
    ...,
    min_length=1,
    max_length=100  # Maximum 100 predictions per request
)
```

#### Response Schema

```python
class PredictionResult(BaseModel):
    """Individual prediction result."""
    predicted_price: float = Field(..., description="Predicted median house value")
    confidence_interval: Optional[dict] = Field(
        None,
        description="Confidence interval (if available)"
    )

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    model_version: str = Field(..., description="Model version used")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "predictions": [{
                    "predicted_price": 452600.0,
                    "confidence_interval": None
                }],
                "model_version": "randomforest_v1"
            }]
        }
    }
```

**`model_version` in response:** Crucial for debugging. If a client reports incorrect predictions, the `model_version` tells you which model was used (v1, v2, Production, etc.).

---

### 5. Router Pattern: Endpoints and Error Handling

#### The Router Structure

```python
# api/app/routers/predict.py

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Global instances (set by main.py)
model_loader: ModelLoader = None
wandb_logger: WandBLogger = None

def set_model_loader(loader: ModelLoader) -> None:
    """Dependency injection pattern."""
    global model_loader
    model_loader = loader
```

**Why `prefix="/api/v1"`:**

```
/api/v1/predict  ← Version 1 of the API
/api/v2/predict  ← Version 2 (breaking changes)
```

You can run both versions simultaneously during migration:
- Legacy clients use `/api/v1/`
- New clients use `/api/v2/`
- Deprecate v1 after 6 months

**Without versioning:** Breaking change → all clients break at the same time.

#### The Main Endpoint: POST /api/v1/predict

```python
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
    summary="Predict housing prices",
    description="Make predictions for housing prices based on input features"
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict housing prices for given features.
    """
    # 1. Check model loaded
    if model_loader is None or not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded"
        )

    start_time = time.time()

    try:
        # 2. Convert Pydantic models to DataFrame
        features_list = [instance.model_dump() for instance in request.instances]
        df = pd.DataFrame(features_list)

        # 3. Make predictions
        predictions = model_loader.predict(df)

        # 4. Calculate metrics
        response_time_ms = (time.time() - start_time) * 1000

        # 5. Format response
        results = [
            PredictionResult(predicted_price=float(pred))
            for pred in predictions
        ]

        # 6. Log to W&B (async, non-blocking)
        if wandb_logger:
            wandb_logger.log_prediction(
                features=features_list,
                predictions=[float(p) for p in predictions],
                model_version=model_loader.model_version,
                response_time_ms=response_time_ms
            )

        return PredictionResponse(
            predictions=results,
            model_version=model_loader.model_version
        )

    except ValueError as e:
        # Validation error (e.g., feature out of expected range)
        if wandb_logger:
            wandb_logger.log_error("validation_error", str(e), features_list)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )

    except Exception as e:
        # Unexpected error (e.g., corrupted model, OOM)
        if wandb_logger:
            wandb_logger.log_error("prediction_error", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
```

**Error handling decisions:**

```python
try:
    # Inference
except ValueError:
    # Client sent invalid data → 400 Bad Request
    # Log to W&B for analysis
    return 400
except Exception:
    # Unexpected error (bug in code/model) → 500 Internal Server Error
    # Log to W&B for alerting
    return 500
```

**Why distinguish 400 vs 500:**
- **400:** Client's fault. No automatic retries.
- **500:** Server's fault. Client can retry.

**Error logging to W&B:** Allows detecting patterns. If you see 1000 `validation_error` for `ocean_proximity="INVALID"`, add a clearer error message.

---

### 6. W&B Logging: Production Observability

#### Why Log Predictions

**Question:** "Why log every prediction if I already have uvicorn logs?"

**Answer:** Uvicorn logs tell you:
- What endpoint was called
- HTTP status code
- How long it took

W&B logs tell you:
- **What features** were used
- **What prediction** was made
- **Distribution of predictions** (are they all in $200k-$500k? are there outliers?)
- **Average latency** per request
- **Error rate** (how many requests fail?)

**Real use case:** Stakeholder reports "predictions are too high lately". You open W&B dashboard:

```
prediction/mean: $450k (before: $380k)
features/median_income: 9.2 (before: 7.5)
```

**Conclusion:** No bug—clients are simply querying houses in more expensive areas (`median_income` higher). Without W&B, you'd be debugging code for hours.

#### The Implementation

```python
# api/app/core/wandb_logger.py

class WandBLogger:
    def __init__(self, project: str = "housing-mlops-api", enabled: bool = True):
        self.enabled = enabled and bool(os.getenv("WANDB_API_KEY"))

        if self.enabled:
            self._run = wandb.init(
                project=self.project,
                job_type="api-inference",  # Distinguish from training runs
                config={
                    "environment": os.getenv("ENVIRONMENT", "production"),
                    "model_version": os.getenv("MODEL_VERSION", "unknown")
                },
                reinit=True  # Allow multiple init() in same process
            )

    def log_prediction(
        self,
        features: List[Dict],
        predictions: List[float],
        model_version: str,
        response_time_ms: float
    ) -> None:
        if not self.enabled:
            return

        # Aggregated metrics
        wandb.log({
            "prediction/count": len(predictions),
            "prediction/mean": sum(predictions) / len(predictions),
            "prediction/min": min(predictions),
            "prediction/max": max(predictions),
            "performance/response_time_ms": response_time_ms,
            "model/version": model_version,
            "timestamp": datetime.now().isoformat()
        })

        # Feature distributions (sample first 100)
        if len(features) <= 100:
            for i, (feat, pred) in enumerate(zip(features, predictions)):
                wandb.log({
                    f"features/instance_{i}/median_income": feat["median_income"],
                    f"predictions/instance_{i}": pred
                })
```

**Why `job_type="api-inference"`:**

In W&B dashboard, you can filter by job type:
- `training`: Training pipeline runs
- `sweep`: Hyperparameter sweep runs
- `api-inference`: Production predictions

**Why `reinit=True`:** A uvicorn process can live for days. `reinit=True` allows creating multiple W&B runs within the same process (one per startup/restart).

**Why sample first 100:** Logging 10,000 individual features per request would be too much overhead. Sampling 100 gives a representative distribution without killing performance.

#### W&B Dashboard in Production

```
# Metrics to monitor:

prediction/count: Requests per minute (RPM)
  - Expected: 100-500 RPM
  - Alert: <10 RPM (is it down?) or >2000 RPM (DDoS?)

prediction/mean: Average predicted price
  - Expected: $300k-$450k (depending on market)
  - Alert: >$1M (broken model) or <$50k (data drift)

performance/response_time_ms: Latency
  - Expected: 30-60ms
  - Alert: >200ms (slow model or CPU throttling)

error/count: Errors per minute
  - Expected: 0-5 errors/min
  - Alert: >50 errors/min (investigate immediately)
```

---

### 7. CORS and Security

```python
# api/app/main.py

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Local frontend (React/Streamlit)
        "http://localhost:8080",
        "https://app.company.com",   # Production frontend
    ],
    allow_credentials=False,  # No cookies (API is stateless)
    allow_methods=["GET", "POST"],  # Only necessary methods
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,  # Cache preflight requests for 1 hour
)
```

**Why restricted origins:**

**Anti-pattern (permissive):**
```python
allow_origins=["*"]  # BAD: Any site can call your API
```

**Problem:** A malicious site `evil.com` can make requests to your API from the user's browser, potentially:
- Consuming your GCP quota (if no auth)
- Making spam predictions
- DoS attack

**Correct pattern (restrictive):**
```python
allow_origins=["https://app.company.com"]  # Only your frontend
```

**For local development:** Add `http://localhost:3000` temporarily, remove in production.

**Why `allow_credentials=False`:** This API is stateless—doesn't use cookies or sessions. `allow_credentials=True` would be unnecessary and an additional attack surface.

---

### 8. The Complete Flow of A Request

**Request:**
```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "longitude": -122.23,
      "latitude": 37.88,
      "housing_median_age": 41,
      "total_rooms": 880,
      "total_bedrooms": 129,
      "population": 322,
      "households": 126,
      "median_income": 8.3252,
      "ocean_proximity": "NEAR BAY"
    }]
  }'
```

**The internal journey (< 50ms):**

```
1. FastAPI receives request (1ms)
   ├─ CORS middleware validates origin
   └─ Router match: POST /api/v1/predict

2. Pydantic validation (2ms)
   ├─ Parse JSON → PredictionRequest object
   ├─ Validate types (longitude: float ✓)
   ├─ Validate ranges (longitude: -122.23, within [-180, 180] ✓)
   └─ Custom validator (ocean_proximity: "NEAR BAY" → valid ✓)

3. Endpoint handler: predict() (40ms)
   ├─ Check model_loader.is_loaded (0.1ms)
   ├─ Convert Pydantic → DataFrame (1ms)
   ├─ Preprocessing (5ms)
   │   ├─ One-hot encode ocean_proximity
   │   ├─ Compute cluster similarity features
   │   └─ Scale numerical features
   ├─ Model inference (30ms)
   │   └─ RandomForest.predict(processed_features)
   ├─ Format response (1ms)
   └─ Log to W&B (async, <1ms non-blocking)

4. FastAPI serializes response (2ms)
   └─ PredictionResponse → JSON

5. HTTP response sent (1ms)

Total: ~50ms
```

**Response:**
```json
{
  "predictions": [{
    "predicted_price": 452600.0,
    "confidence_interval": null
  }],
  "model_version": "models:/housing_price_model/Production"
}
```

---

### 9. What This Architecture Achieves

**Without this architecture (naive API):**
- Load model on every request (5 seconds/request)
- Manual validation prone to errors
- No observability (debugging is guessing)
- No API versioning (breaking changes break clients)
- Open CORS (vulnerability)

**With this architecture:**
- **Latency:** <50ms per prediction (cached model)
- **Reliability:** Pydantic guarantees valid requests before reaching the model
- **Observability:** W&B dashboard shows prediction distribution, latency, errors
- **Maintainability:** Separation of concerns (core/routers/models)
- **Security:** Restrictive CORS, robust error handling
- **Versioning:** `/api/v1/` allows evolving the API without breaking clients

**The real value:** This API can scale from 10 requests/min to 10,000 requests/min without code changes—just add more containers with a load balancer. The architecture is already ready.

---

<a name="model-strategies"></a>
## 11. Model Selection and Parameter Strategies

---

## Navigation

**[← Part 1: Pipeline and Orchestration](/post/anatomia-pipeline-mlops-part-1-en/)** | **[Part 3: Production and Best Practices →](/post/anatomia-pipeline-mlops-part-3-en/)**

In Part 3 we will cover:
- Model selection and parameter strategies
- Testing: Fixtures, mocking, and real coverage
- Production patterns (Transform Pattern, Data Drift, Feature Stores)
- Production Readiness Checklist
