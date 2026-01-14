---
title: "Anatomía de un Pipeline MLOps - Parte 2: Deployment e Infraestructura"
date: 2026-01-13
draft: false
tags: ["mlops", "ci-cd", "docker", "fastapi", "github-actions", "wandb", "mlflow"]
categories: ["MLOps", "Engineering"]
author: "Carlos Daniel Jiménez"
description: "Parte 2: CI/CD con GitHub Actions, comparación W&B vs MLflow, containerización completa con Docker, y arquitectura de API con FastAPI en producción."
---

> **Serie MLOps Completo:** [← Parte 1: Pipeline](/mlops/anatomia-pipeline-mlops-parte-1/) | **Parte 2 (actual)** | [Parte 3: Producción →](/mlops/anatomia-pipeline-mlops-parte-3/)


<a name="github-actions"></a>
## 8. CI/CD con GitHub Actions: Automatización del Pipeline Completo

### Por Qué CI/CD Es Crítico en MLOps

Como MLOps engineer, uno de los mayores puntos de fricción es el deployment manual. Has entrenado un modelo excelente en tu laptop, pero llevarlo a producción requiere:

1. SSH a un servidor
2. Copiar archivos manualmente
3. Instalar dependencias
4. Cruzar los dedos
5. Debuggear cuando algo explota

**GitHub Actions elimina esto.** Cada commit dispara un pipeline automatizado que:

- Ejecuta tests
- Valida que el código cumple estándares
- Entrena el modelo (opcional, en pipelines simples)
- Construye imágenes Docker
- Deploya a Cloud Run/ECS/Kubernetes

### La Arquitectura de CI/CD Para Este Proyecto

Este proyecto implementa **dos workflows separados**:

#### 1. PR Validation Workflow

**Trigger:** Cada pull request a `main`

**Propósito:** Asegurar que el código es production-ready antes de mergear

```yaml
# .github/workflows/pr_validation.yaml
name: PR Validation - Tests & Linting

on:
  pull_request:
    branches: [main, master]
    paths:
      - 'src/**'
      - 'api/**'
      - 'tests/**'
      - 'pyproject.toml'
      - 'requirements.txt'

jobs:
  lint:
    name: Lint Code
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: |
          uv venv
          uv pip install -e .
          uv pip install ruff pytest pytest-cov

      - name: Run Ruff linter
        run: |
          source .venv/bin/activate
          ruff check src/ tests/ api/

      - name: Run Ruff formatter check
        run: |
          source .venv/bin/activate
          ruff format --check src/ tests/ api/

  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    env:
      GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
      GCS_BUCKET_NAME: ${{ secrets.GCS_BUCKET_NAME }}
      WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv venv
          uv pip install -e .
          uv pip install pytest pytest-cov pytest-mock

      - name: Run unit tests with coverage
        run: |
          source .venv/bin/activate
          pytest tests/ -v \
            --cov=src \
            --cov=api/app \
            --cov-report=xml \
            --cov-report=term-missing \
            --cov-fail-under=70

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  integration-tests:
    name: Integration Tests (Pipeline E2E)
    runs-on: ubuntu-latest
    env:
      GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
      GCS_BUCKET_NAME: ${{ secrets.GCS_BUCKET_NAME }}
      WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Install dependencies
        run: |
          pip install uv
          uv venv
          uv pip install -e .

      - name: Run integration test (Steps 01-04)
        run: |
          source .venv/bin/activate
          python main.py main.execute_steps=[01_download_data,02_preprocessing_and_imputation,03_feature_engineering,04_segregation]
        timeout-minutes: 30

      - name: Verify artifacts were created
        run: |
          gsutil ls gs://${{ secrets.GCS_BUCKET_NAME }}/data/04-split/train/train.parquet
          gsutil ls gs://${{ secrets.GCS_BUCKET_NAME }}/data/04-split/test/test.parquet
```

**Valor para el MLOps engineer:**

- **Previene merges rotos:** Si los tests fallan, el PR no puede mergearse
- **Estándares de código:** Ruff garantiza consistencia (importa cuando tienes 5+ contributors)
- **Coverage tracking:** Codecov muestra qué porcentaje del código está cubierto por tests
- **Fast feedback:** Sabes en 5 minutos si tu cambio rompió algo, no 3 horas después

#### 2. Deployment Workflow

**Trigger:** Push a `main` (después de merge de PR)

**Propósito:** Construir y deployar el API a producción

```yaml
# .github/workflows/deploy_api.yaml
name: Deploy API to Cloud Run

on:
  push:
    branches: [main]
    paths:
      - 'api/**'
      - 'models/**'
      - '.github/workflows/deploy_api.yaml'

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  SERVICE_NAME: housing-price-api
  REGION: us-central1

jobs:
  build-and-deploy:
    name: Build Docker Image & Deploy
    runs-on: ubuntu-latest

    permissions:
      contents: read
      id-token: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker for GCR
        run: gcloud auth configure-docker gcr.io

      - name: Download trained model from GCS
        run: |
          mkdir -p api/models/trained
          gsutil cp gs://${{ secrets.GCS_BUCKET_NAME }}/models/trained/housing_price_model.pkl \
            api/models/trained/housing_price_model.pkl

      - name: Build Docker image
        run: |
          cd api
          docker build \
            --tag gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:${{ github.sha }} \
            --tag gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:latest \
            .

      - name: Push Docker image to GCR
        run: |
          docker push gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:${{ github.sha }}
          docker push gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:latest

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ${{ env.SERVICE_NAME }} \
            --image gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:${{ github.sha }} \
            --platform managed \
            --region ${{ env.REGION }} \
            --allow-unauthenticated \
            --set-env-vars="GCS_BUCKET=${{ secrets.GCS_BUCKET_NAME }},WANDB_API_KEY=${{ secrets.WANDB_API_KEY }}" \
            --memory 2Gi \
            --cpu 2 \
            --max-instances 10 \
            --min-instances 1 \
            --timeout 300

      - name: Get Cloud Run URL
        id: deploy-url
        run: |
          URL=$(gcloud run services describe ${{ env.SERVICE_NAME }} \
            --platform managed \
            --region ${{ env.REGION }} \
            --format 'value(status.url)')
          echo "url=$URL" >> $GITHUB_OUTPUT

      - name: Run smoke test
        run: |
          curl -X POST "${{ steps.deploy-url.outputs.url }}/api/v1/predict" \
            -H "Content-Type: application/json" \
            -d '{"instances":[{"longitude":-122.23,"latitude":37.88,"housing_median_age":41,"total_rooms":880,"total_bedrooms":129,"population":322,"households":126,"median_income":8.3252,"ocean_proximity":"NEAR BAY"}]}'

      - name: Notify deployment success
        if: success()
        run: |
          echo "Deployment successful! API available at: ${{ steps.deploy-url.outputs.url }}"
```

**Valor para el MLOps engineer:**

- **Zero-downtime deployment:** Cloud Run hace rolling updates automáticamente
- **Rollback fácil:** Si algo explota, haces `gcloud run services update-traffic --to-revisions=PREVIOUS=100`
- **Smoke test automático:** Verifica que el API responde después del deploy
- **Versionado de imágenes:** Cada commit tiene su propia imagen Docker taggeada con SHA

### Secretos y Seguridad

**CRÍTICO:** Nunca commitees secrets al repo. GitHub Actions usa **GitHub Secrets** para guardar:

- `GCP_PROJECT_ID`: ID del proyecto de GCP
- `GCS_BUCKET_NAME`: Nombre del bucket de GCS
- `WANDB_API_KEY`: API key de W&B
- `GCP_SA_KEY`: Service account key (JSON) para autenticar en GCP
- `WIF_PROVIDER` / `WIF_SERVICE_ACCOUNT`: Workload Identity Federation (más seguro que SA keys)

**Configuración en GitHub:**

1. Ve a repo → Settings → Secrets and variables → Actions
2. Crea cada secret
3. Los workflows acceden con `${{ secrets.SECRET_NAME }}`

### Monitoreo de Deployments

**¿Cómo saber si un deployment falló?**

GitHub Actions envía notificaciones a:
- Email (configurado en GitHub profile)
- Slack (con GitHub app)
- Discord/Teams (con webhooks)

**Post-deployment monitoring:**

```yaml
# Agregar step de validación post-deploy
- name: Run API health check
  run: |
    for i in {1..5}; do
      STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${{ steps.deploy-url.outputs.url }}/health")
      if [ $STATUS -eq 200 ]; then
        echo "Health check passed"
        exit 0
      fi
      echo "Attempt $i failed, retrying..."
      sleep 10
    done
    echo "Health check failed after 5 attempts"
    exit 1
```

### Estrategias Avanzadas de CI/CD

#### 1. Pipeline de Reentrenamiento Automático

**Trigger:** Cron schedule (ejemplo: semanalmente)

```yaml
on:
  schedule:
    - cron: '0 2 * * 0'  # Cada domingo a las 2 AM UTC

jobs:
  retrain-model:
    runs-on: ubuntu-latest
    steps:
      - name: Run full pipeline
        run: python main.py

      - name: Compare metrics with production model
        run: |
          NEW_MAPE=$(python scripts/get_latest_mape.py)
          PROD_MAPE=$(python scripts/get_production_mape.py)

          if (( $(echo "$NEW_MAPE < $PROD_MAPE" | bc -l) )); then
            echo "New model is better, promoting to Production"
            mlflow models transition --name housing_price_model --version latest --stage Production
          else
            echo "New model is worse, keeping current Production model"
          fi
```

**Valor:** El modelo se reentrena automáticamente con datos nuevos. Si mejora, se promociona a Production. Si empeora, se descarta.

#### 2. Canary Deployments

**Problema:** Un nuevo modelo puede tener bugs sutiles que no aparecen en tests.

**Solución:** Deployar el nuevo modelo a solo 10% del tráfico, monitorear por 1 hora, luego migrar 100% si no hay errores.

```yaml
- name: Deploy canary (10% traffic)
  run: |
    gcloud run services update-traffic ${{ env.SERVICE_NAME }} \
      --to-revisions=LATEST=10,PREVIOUS=90

- name: Wait and monitor
  run: sleep 3600  # 1 hora

- name: Check error rate
  run: |
    ERROR_RATE=$(python scripts/check_error_rate.py --minutes=60)
    if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
      echo "Error rate too high, rolling back"
      gcloud run services update-traffic ${{ env.SERVICE_NAME }} --to-revisions=PREVIOUS=100
      exit 1
    fi

- name: Promote to 100% traffic
  run: |
    gcloud run services update-traffic ${{ env.SERVICE_NAME }} --to-revisions=LATEST=100
```

### Lo Que CI/CD Resuelve en MLOps

**Sin CI/CD:**
- Deployment manual propenso a errores
- "Funciona en mi máquina" syndrome
- Testing inconsistente
- Rollback requiere pánico debugging
- No hay historial de qué se deployó cuándo

**Con CI/CD:**
- Deployment automático en cada merge
- Tests garantizan que el código funciona
- Rollback es un comando
- Historial completo en GitHub Actions UI
- Cada deployment es reproducible

### El Valor Real Para el MLOps Engineer

**No es sobre automatizar por automatizar.** Es sobre:

1. **Reducir toil:** Gastas tiempo resolviendo problemas interesantes, no copiando archivos manualmente
2. **Confianza:** Sabes que el código funciona antes de llegar a producción
3. **Velocidad:** De commit a producción en <10 minutos
4. **Auditoría:** Cada cambio está loggeado en GitHub
5. **Colaboración:** Tu equipo puede deployar sin depender de ti

**Un MLOps engineer sin CI/CD es como un software engineer sin git—técnicamente posible, pero fundamentalmente broken.**

---

<a name="mlops-value-proposition"></a>
## 9. El Valor de MLOps: Por Qué Esto Importa

### La Pregunta Central

"¿Por qué debería invertir tiempo en todo esto cuando puedo entrenar un modelo en un notebook en 30 minutos?"

Esta es la pregunta que todo MLOps engineer ha escuchado. La respuesta corta: **porque el notebook no escala.**

La respuesta larga es lo que cubre esta sección.

### El Problema Real: Research Code vs Production Code

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

**Problemas:**
- No reproducible (orden de ejecución importa)
- No testeable
- No versionable (git diffs son ilegibles)
- No escalable (qué pasa con 100GB de datos?)
- No auditable (qué params usaste?)

#### Production Code (Este Pipeline)

```python
# src/model/05_model_selection/main.py

@hydra.main(config_path=".", config_name="config")
def train(config: DictConfig) -> None:
    """Entrenar modelo con configuración versionada."""

    # Cargar datos desde GCS (single source of truth)
    df = load_from_gcs(config.gcs_train_path)

    # Aplicar preprocessing pipeline serializado
    pipeline = joblib.load('artifacts/preprocessing_pipeline.pkl')
    X = pipeline.transform(df)

    # Entrenar con params de config
    model = RandomForestRegressor(**config.hyperparameters)
    model.fit(X, y)

    # Loggear a MLflow
    mlflow.log_params(config.hyperparameters)
    mlflow.log_metrics(evaluate(model, X_test, y_test))
    mlflow.sklearn.log_model(model, "model")

    return model
```

**Beneficios:**
- Reproducible (mismo config = mismo output)
- Testeable (funciones puras, mocking)
- Versionable (git diff legible)
- Escalable (corre en local o en cluster)
- Auditable (MLflow tracking)

### Valor #1: Modularización de Código

#### Por Qué Importa

**Escenario:** Tu modelo tiene bug en preprocessing. En notebook, el preprocessing está mezclado con feature engineering, entrenamiento y evaluación en 300 líneas.

**En este pipeline:**

```bash
# Bug está en preprocessing → solo editas src/data/02_preprocessing/
# Tests fallan → pytest tests/test_preprocessor.py
# Fixeas → reejecutas solo step 02-07, no 01
```

**Tiempo ahorrado:** Horas por bug.

#### Separation of Concerns

Este pipeline separa:

1. **Data steps (01-04):** Producen artifacts reutilizables
2. **Model steps (05-07):** Consumen artifacts, producen modelos
3. **API:** Consume modelos, produce predicciones
4. **Frontend:** Consume API, produce UX

**Beneficio:** Equipos pueden trabajar en paralelo. El data scientist modifica feature engineering sin tocar el API. El frontend engineer modifica UI sin entender Random Forests.

### Valor #2: Working with Artifacts

#### El Problema: "¿Dónde está el model_final_v3.pkl?"

Sin artifact management:

```
models/
├── model_v1.pkl
├── model_v2.pkl
├── model_final.pkl
├── model_final_FINAL.pkl
├── model_final_REAL.pkl
├── model_production_2024_01_15.pkl  # ¿Este es el de producción?
└── model_old_backup.pkl  # ¿Puedo borrarlo?
```

**Problemas:**
- No sabes qué hiperparámetros usa cada uno
- No sabes qué métricas logró
- No sabes con qué datos se entrenó
- Rollback = buscar el archivo correcto

#### La Solución: Artifact Storage + Metadata

**1. Google Cloud Storage para datos:**

```
gs://bucket-name/
├── data/
│   ├── 01-raw/housing.parquet                    # Inmutable
│   ├── 02-processed/housing_processed.parquet    # Versionado por fecha
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

**Beneficios:**
- **Inmutabilidad:** `01-raw/` nunca cambia, siempre puedes reejecutar el pipeline
- **Versionamiento:** Cada run tiene timestamp, puedes comparar versiones
- **Compartir:** Todo el equipo accede a los mismos datos, no "enviame el CSV por Slack"

**2. MLflow para modelos:**

```python
# Registrar modelo
mlflow.sklearn.log_model(model, "model")

# MLflow guarda automáticamente:
# - El pickle del modelo
# - Los hiperparámetros (n_estimators=200, max_depth=20)
# - Las métricas (MAPE=7.8%, R²=0.87)
# - Metadata (fecha, duración, usuario)
# - Código (git commit SHA)

# Cargar modelo en producción
model = mlflow.pyfunc.load_model("models:/housing_price_model/Production")
```

**Beneficios:**
- **Versionamiento semántico:** v1, v2, v3 con stages (Staging/Production)
- **Metadata rica:** Sabes exactamente qué es cada versión
- **Rollback trivial:** `transition v2 to Production`
- **Comparación:** MLflow UI muestra tabla comparando todas las versiones

**3. W&B para experimentos:**

```python
# Cada run de sweep loggea:
wandb.log({
    "hyperparameters/n_estimators": 200,
    "hyperparameters/max_depth": 20,
    "metrics/mape": 7.8,
    "metrics/r2": 0.87,
    "plots/feature_importances": wandb.Image(fig),
    "dataset/train_size": 16512,
})

# W&B dashboard:
# - Tabla con 50 runs de sweep
# - Filtrar por MAPE < 8%
# - Parallel coordinates plot mostrando relación entre hiperparámetros y MAPE
# - Comparar top 5 runs side-by-side
```

**Beneficios:**
- **Visualización:** Plots interactivos de cómo cada hiperparámetro afecta métricas
- **Colaboración:** Tu equipo ve tus experimentos en real-time
- **Reproducibilidad:** Cada run tiene link permanente con todo el contexto

### Valor #3: Pipeline Architecture

#### Por Qué Un Pipeline, No Un Script

**Script único (run_all.py):**

```python
# run_all.py (500 líneas)

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

**Problemas:**
- Si falla en train_model(), reejecutas TODO (incluyendo download lento)
- No puedes ejecutar solo feature engineering para experimentar
- Cambiar preprocessing requiere reentrenar todo
- No hay checkpoints intermedios

**Pipeline modular:**

```bash
# Ejecutar todo
make run-pipeline

# Ejecutar solo preprocessing
make run-preprocessing

# Ejecutar desde feature engineering en adelante
python main.py main.execute_steps=[03_feature_engineering,04_segregation,05_model_selection]

# Debugging: ejecutar solo step que falló
python src/data/03_feature_engineering/main.py --debug
```

**Beneficios:**
- **Ejecución selectiva:** Solo reejecutas lo que cambió
- **Debugging rápido:** Testeas un step aislado
- **Paralelización:** Steps independientes pueden correr en paralelo
- **Checkpointing:** Si falla step 05, steps 01-04 ya están hechos

#### El Contrato Entre Steps

Cada step:
- **Input:** Path a artifact en GCS (ejemplo: `data/02-processed/housing_processed.parquet`)
- **Output:** Path a nuevo artifact en GCS (ejemplo: `data/03-features/housing_features.parquet`)
- **Side effects:** Logs a MLflow/W&B

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

Este **contrato** permite que cada step sea:
- Testeado independientemente
- Desarrollado por diferentes personas
- Reemplazado sin afectar otros steps

### Valor #4: Production-Ready vs Research Code

#### Checklist de Production-Ready

| Feature | Research Code | Este Pipeline |
|---------|---------------|---------------|
| **Versionamiento** | Git (mal, notebooks) | Git + GCS + MLflow |
| **Testing** | Manual ("lo corrí una vez") | pytest + CI |
| **Configuración** | Hardcoded | YAML versionado |
| **Secretos** | Expuestos en código | .env + GitHub Secrets |
| **Logs** | print() statements | Logging estructurado |
| **Monitoring** | "Espero que funcione" | W&B + MLflow tracking |
| **Deployment** | Manual | CI/CD automático |
| **Rollback** | Panic debugging | Transition en MLflow |
| **Documentación** | README desactualizado | Código autodocumentado + Markdown en MLflow |
| **Colaboración** | "Ejecuta estas 10 celdas en orden" | `make run-pipeline` |

#### El Costo Real de No Hacer MLOps

**Escenario:** Un modelo en producción tiene bug que causa predicciones incorrectas.

**Sin MLOps (Research Code):**
1. Detectar el bug: Usuario reporta → 2 horas
2. Reproducir el bug: Buscar qué código/datos se usaron → 4 horas
3. Fixear: Correr notebook localmente → 1 hora
4. Deployar: SSH, copiar pickle, restart server → 30 min
5. Verificar: Correr tests manuales → 1 hora
6. **Total: 8.5 horas de downtime**

**Con MLOps (Este Pipeline):**
1. Detectar el bug: Monitoring automático alerta → 5 min
2. Rollback: `transition v3 to Archived` + `transition v2 to Production` → 2 min
3. Fix: Identificar issue con MLflow metadata, fixear código → 1 hora
4. Deployar: Push to GitHub → CI/CD automático → 10 min
5. Verificar: Smoke tests automáticos pasan → 1 min
6. **Total: 1 hora 18 min de downtime (>85% reducción)**

**Ahorro anualizado:** Si esto pasa 4 veces al año, ahorras 29 horas de tiempo de ingeniero.

### Valor #5: Decisiones Respaldadas por Datos

#### El Anti-Pattern

"Usé Random Forest con `n_estimators=100` porque eso es lo que hace todo el mundo."

**Problema:** No tienes evidencia de que es la mejor opción.

#### Este Pipeline

Cada decisión tiene métricas cuantificables:

**1. Imputación:**
- Comparó 4 estrategias (Simple median, Simple mean, KNN, IterativeImputer)
- IterativeImputer ganó con RMSE=0.52 (vs 0.78 de median)
- Plot de comparación en W&B: `wandb.ai/project/run/imputation_comparison`

**2. Feature Engineering:**
- Optimizó K de 5 a 15
- K=8 maximizó silhouette score (0.64)
- Plot de elbow method en W&B

**3. Hyperparameter Tuning:**
- Sweep bayesiano de 50 runs
- Optimal config: `n_estimators=200, max_depth=20`
- MAPE mejoró de 8.5% a 7.8%
- Link a sweep: `wandb.ai/project/sweeps/abc123`

**Beneficio:** Seis meses después, cuando el stakeholder pregunta "¿por qué usamos este modelo?", abres W&B/MLflow y la respuesta está ahí con plots y métricas.

### El ROI de MLOps

**Inversión inicial:**
- Setup de GCS, MLflow, W&B, CI/CD: 2-3 días
- Refactoring de código a pipeline modular: 1-2 semanas

**Retorno:**
- Deployment time: 8 horas → 10 minutos (48x más rápido)
- Debugging time: 4 horas → 30 min (8x más rápido)
- Onboarding nuevos engineers: 1 semana → 1 día
- Confianza del equipo: "Espero que funcione" → "Sé que funciona"

**Para un equipo de 5 personas, el breakeven es ~1 mes.**

### La Lección Final Para MLOps Engineers

**No es sobre las herramientas.** Puedes reemplazar:
- GCS → S3 → Azure Blob
- MLflow → Neptune → Comet
- W&B → TensorBoard → MLflow
- GitHub Actions → GitLab CI → Jenkins

**Es sobre los principios:**

1. **Modularización:** Código en módulos testeables, no notebooks monolíticos
2. **Artifact Management:** Datos y modelos versionados, no `model_final_v3.pkl`
3. **Automatización:** CI/CD elimina toil
4. **Observabilidad:** Logs, métricas, tracking
5. **Reproducibilidad:** Mismo input → mismo output
6. **Decisiones data-driven:** Cada elección respaldada por métricas

**Cuando entiendes esto, eres un MLOps engineer. Cuando lo implementas, eres un buen MLOps engineer.**

---

<a name="wandb-vs-mlflow"></a>
## 9.5. W&B vs MLflow: Por Qué Ambos, No Uno u Otro

### La Pregunta Incómoda

"¿Por qué tienes Weights & Biases Y MLflow? ¿No son lo mismo?"

Esta pregunta revela un malentendido fundamental sobre lo que hace cada herramienta. No son competidores—son **aliados con responsabilidades diferentes**. Entender esto separa un data scientist que experimenta de un MLOps engineer que construye sistemas.

La respuesta corta: **W&B es tu laboratorio de investigación. MLflow es tu cadena de producción.**

La respuesta larga es lo que cubre esta sección, con ejemplos del código real de este proyecto.

---

### El Problema Real: Experimentación vs Governance

#### Fase 1: Experimentación (50-100 runs/día)

Cuando estás en fase de experimentación:
- Corres 50 sweep runs probando combinaciones de hiperparámetros
- Necesitas ver **en tiempo real** cómo evoluciona cada run
- Quieres comparar visualmente 20 runs simultáneos
- Necesitas ver plots de convergencia, distribuciones de features, confusion matrices
- El overhead de logging debe ser mínimo (logging asíncrono)

**Herramienta correcta:** Weights & Biases

#### Fase 2: Governance y Deployment (1-2 modelos/semana)

Cuando subes un modelo a producción:
- Necesitas versionamiento semántico (v1, v2, v3)
- Necesitas stages (Staging → Production)
- Necesitas metadata rica (¿qué hiperparámetros? ¿qué datos? ¿qué commit?)
- Necesitas un API para cargar modelos (`models:/housing_price_model/Production`)
- Necesitas rollback trivial (transition v2 to Production)

**Herramienta correcta:** MLflow Model Registry

**La verdad incómoda:** Ninguna herramienta hace bien ambas cosas.

---

### Cómo Este Proyecto Usa W&B

#### 1. Hyperparameter Sweep (Step 06): Bayesian Optimization

```python
# src/model/06_sweep/main.py

# Configuración del sweep (Bayesian optimization)
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

# Inicializar sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="housing-mlops-gcp")

# Función de training que W&B llama 50 veces
def train():
    run = wandb.init()  # W&B asigna hiperparámetros automáticamente

    # Obtener hiperparámetros sugeridos por Bayesian optimizer
    config = wandb.config

    # Entrenar modelo
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        # ...
    )
    model.fit(X_train, y_train)

    # Evaluar
    metrics = evaluate_model(model, X_test, y_test)

    # Log a W&B (asíncrono, no bloquea)
    wandb.log({
        "hyperparameters/n_estimators": config.n_estimators,
        "hyperparameters/max_depth": config.max_depth,
        "metrics/mape": metrics['mape'],
        "metrics/wmape": metrics['wmape'],  # Optimizer usa esto
        "metrics/r2": metrics['r2'],
        "plots/feature_importances": wandb.Image(fig),
    })

    run.finish()

# Ejecutar 50 runs con Bayesian optimization
wandb.agent(sweep_id, function=train, count=50)
```

**Lo que W&B hace aquí que MLflow no puede:**

1. **Bayesian Optimization**: W&B sugiere los próximos hiperparámetros basándose en runs previos. No es random—usa Gaussian Processes para explorar el espacio eficientemente.

   ```
   Run 1: n_estimators=100, max_depth=15 → wMAPE=8.5%
   Run 2: n_estimators=200, max_depth=20 → wMAPE=7.9%  # Mejor
   Run 3: n_estimators=250, max_depth=22 → wMAPE=7.8%  # W&B sugiere valores cercanos a Run 2
   ```

2. **Early Termination (Hyperband)**: Si un run va mal en las primeras 3 iteraciones (epochs), W&B lo mata automáticamente y prueba otros hiperparámetros. Ahorra ~40% de compute.

   ```python
   "early_terminate": {
       "type": "hyperband",
       "min_iter": 3  # Mínimo 3 iteraciones antes de terminar
   }
   ```

3. **Parallel Coordinates Plot**: Visualización interactiva mostrando qué combinación de hiperparámetros produce mejor wMAPE.

   ![W&B Parallel Coordinates](https://docs.wandb.ai/assets/images/parallel-coordinates.png)

   **Interpretación:** Las líneas azules (runs con wMAPE bajo) convergen en `n_estimators=200-250` y `max_depth=20-25`. Esto te dice visualmente dónde está el óptimo.

4. **Logging Asíncrono**: `wandb.log()` no bloquea. Mientras el modelo entrena, W&B sube métricas en background. Total overhead: <1% del training time.

**MLflow no tiene:**
- Bayesian optimization (solo Grid/Random search vía scikit-learn)
- Early termination inteligente
- Parallel coordinates plots
- Logging asíncrono (mlflow.log es síncrono)

---

#### 2. Real-Time Monitoring: Ver Runs Mientras Corren

```python
# En W&B dashboard (web UI):
# - Ver 50 runs simultáneos en tabla interactiva
# - Filtrar por "wmape < 8.0%" → muestra solo 12 runs
# - Comparar top 5 runs side-by-side
# - Ver plots de convergencia (MAPE vs iteration)
```

**Caso de uso real:** Inicias un sweep de 50 runs a las 9 AM. A las 10 AM, desde tu laptop en la cafetería:
1. Abres W&B dashboard
2. Ves que 30 runs ya terminaron
3. Filtras por `wmape < 8.0%` → 8 runs cumplen
4. Comparas esos 8 runs → identificas que `max_depth=20` aparece en todos
5. **Decisión:** Cancelas el sweep, ajustas el range de `max_depth` a [18, 25], reinicias

**Valor:** Retroalimentación inmediata sin SSH al server, sin leer logs en terminal. La experimentación es **interactiva**, no batch.

---

#### 3. Artifact Tracking Ligero (Referencias a GCS)

```python
# src/model/05_model_selection/main.py

# Upload modelo a GCS
model_gcs_uri = upload_model_to_gcs(model, "models/05-selection/randomforest_best.pkl")
# gs://bucket/models/05-selection/randomforest_best.pkl

# Log referencia en W&B (NO sube el pickle, solo el URI)
artifact = wandb.Artifact(
    name="best_model_selection",
    type="model",
    description="Best model selected: RandomForest"
)
artifact.add_reference(model_gcs_uri, name="best_model.pkl")  # Solo el URI
run.log_artifact(artifact)
```

**W&B no almacena el modelo**—solo guarda el URI `gs://...`. El modelo vive en GCS.

**Ventaja:** No pagas doble storage (GCS + W&B). W&B es el índice, GCS es el almacén.

---

### Cómo Este Proyecto Usa MLflow

#### 1. Model Registry (Step 07): Versionamiento y Stages

```python
# src/model/07_registration/main.py

with mlflow.start_run(run_name="model_registration"):
    # Log modelo
    mlflow.sklearn.log_model(model, "model")

    # Log params y métricas
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2
    })
    mlflow.log_metrics({
        "mape": 7.82,
        "r2": 0.8654
    })

    # Registrar en Model Registry
    client = MlflowClient()

    # Crear modelo registrado (si no existe)
    client.create_registered_model(
        name="housing_price_model",
        description="Housing price prediction - Random Forest"
    )

    # Crear nueva versión
    model_version = client.create_model_version(
        name="housing_price_model",
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )
    # Resultado: housing_price_model/v3

    # Transicionar a stage
    client.transition_model_version_stage(
        name="housing_price_model",
        version=model_version.version,
        stage="Staging"  # Staging → Production cuando se valide
    )
```

**Lo que MLflow hace aquí que W&B no puede:**

1. **Versionamiento Semántico**: Cada modelo es `housing_price_model/v1`, `v2`, `v3`. No son IDs aleatorios—son versiones incrementales.

2. **Stages**: Un modelo pasa por `None → Staging → Production → Archived`. Este lifecycle es explícito.

   ```
   v1: Production (actual en el API)
   v2: Staging (validándose)
   v3: None (recién entrenado)
   v4: Archived (deprecado)
   ```

3. **Model-as-Code API**: Cargar modelo en el API es trivial:

   ```python
   # api/app/core/model_loader.py

   model = mlflow.pyfunc.load_model("models:/housing_price_model/Production")
   ```

   **No necesitas saber:**
   - Dónde está el pickle físicamente
   - Qué versión es (MLflow resuelve "Production" → v1)
   - Cómo deserializarlo (mlflow.pyfunc abstrae esto)

4. **Rollback en 10 Segundos**:

   ```bash
   # Problema: v3 en Production tiene bug
   # Rollback a v2:
   mlflow models transition \
     --name housing_price_model \
     --version 2 \
     --stage Production

   # El API detecta el cambio y recarga v2 automáticamente
   ```

5. **Metadata Rica con Tags y Description**:

   ```python
   # Agregar tags searchables
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
       "abc123xyz"  # Link al W&B sweep
   )

   # Description en Markdown
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

   **Resultado:** 6 meses después, cuando un stakeholder pregunta "¿qué modelo está en Production?", abres MLflow UI y toda la info está ahí—no en un Slack thread perdido.

**W&B no tiene:**
- Model Registry (solo artifact tracking básico)
- Stages (Staging/Production)
- API de carga (`models:/name/stage`)
- Transition history (quién cambió v2 a Production, cuándo, por qué)

---

#### 2. Pipeline Orchestration (main.py)

```python
# main.py

@hydra.main(config_path=".", config_name="config")
def go(config: DictConfig) -> None:
    # MLflow orquesta steps como sub-runs

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

**MLflow crea un run jerárquico:**

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

**Valor:** En MLflow UI, ves toda la ejecución del pipeline como un árbol. Cada step es auditable—qué params usó, cuánto tardó, qué artifacts produjo.

**W&B no tiene orquestación de pipelines**—solo tracking de runs individuales.

---

### La División del Trabajo en Este Proyecto

| Responsabilidad | W&B | MLflow | Razón |
|-----------------|-----|--------|-------|
| **Bayesian hyperparameter optimization** | ✓ | ✗ | W&B tiene sweep inteligente, MLflow solo Grid/Random |
| **Real-time dashboards** | ✓ | ✗ | W&B UI es interactivo, MLflow UI es estático |
| **Parallel coordinates plots** | ✓ | ✗ | W&B tiene visualizaciones avanzadas |
| **Early termination (Hyperband)** | ✓ | ✗ | W&B implementa Hyperband/ASHA/Median stopping |
| **Model Registry con stages** | ✗ | ✓ | MLflow tiene Staging/Production, W&B no |
| **Model-as-code API** | ✗ | ✓ | `mlflow.pyfunc.load_model()` es el estándar |
| **Rollback de modelos** | ✗ | ✓ | MLflow transition, W&B no tiene concepto de stages |
| **Pipeline orchestration** | ✗ | ✓ | `mlflow.run()` ejecuta steps anidados |
| **Artifact storage (físico)** | ✗ | ✗ | Ambos apuntan a GCS, no duplican storage |
| **Logging asíncrono** | ✓ | ✗ | W&B no bloquea training, MLflow sí |
| **Metadata searchable** | ✓ | ✓ | Ambos permiten tags/búsqueda, implementaciones diferentes |

---

### El Flujo Completo: W&B → MLflow

**Día 1-3: Experimentación (W&B)**

```bash
# Ejecutar sweep de 50 runs
make run-sweep

# W&B dashboard muestra:
# - 50 runs en tabla
# - Parallel coordinates plot
# - Best run: n_estimators=200, max_depth=20, wMAPE=7.8%
# - Sweep ID: abc123xyz
```

**Output:** `src/model/06_sweep/best_params.yaml`

```yaml
hyperparameters:
  n_estimators: 200
  max_depth: 20
  min_samples_split: 2
  min_samples_leaf: 1
metrics:
  mape: 7.82
  wmape: 7.76
  r2: 0.8654
sweep_id: abc123xyz  # Link a W&B
best_run_id: def456ghi
```

**Día 4: Registration (MLflow)**

```bash
# Step 07 lee best_params.yaml
python main.py main.execute_steps=[07_registration]

# MLflow:
# 1. Entrena modelo con best_params
# 2. Registra como housing_price_model/v3
# 3. Transiciona a Staging
# 4. Guarda metadata (incluyendo sweep_id)
```

**Día 5-7: Validación en Staging**

```bash
# API corre con modelo en Staging
docker run -p 8080:8080 \
  -e MLFLOW_MODEL_NAME=housing_price_model \
  -e MLFLOW_MODEL_STAGE=Staging \
  housing-api:latest

# Correr tests, validar métricas, revisar predicciones
```

**Día 8: Promoción a Production**

```bash
mlflow models transition \
  --name housing_price_model \
  --version 3 \
  --stage Production

# API en producción auto-recarga v3
# v2 queda como fallback (stage: Archived)
```

**Si algo falla:**

```bash
# Rollback en 10 segundos
mlflow models transition \
  --name housing_price_model \
  --version 2 \
  --stage Production
```

---

### Por Qué Ambos, Definitivamente

**Pregunta:** "¿Puedo usar solo W&B?"

**Respuesta:** Puedes, pero pierdes:
- Model Registry (versionamiento, stages, rollback)
- API estándar para cargar modelos en producción
- Pipeline orchestration con runs jerárquicos

**Resultado:** Terminas construyendo tu propio sistema de versionamiento de modelos con scripts custom—reinventando la rueda mal.

**Pregunta:** "¿Puedo usar solo MLflow?"

**Respuesta:** Puedes, pero pierdes:
- Bayesian optimization (tendrás que hacer Grid Search lento)
- Visualizaciones interactivas (parallel coordinates, real-time dashboards)
- Early termination inteligente (desperdicias compute)

**Resultado:** Tus sweeps toman 3x más tiempo, y no tienes feedback visual de qué funciona.

---

### El Costo Real

**W&B:**
- Free tier: 100GB storage, colaboradores ilimitados
- Team tier: $50/usuario/mes (para equipos >5 personas)

**MLflow:**
- Open source, gratis
- Costo: Hosting del tracking server (Cloud Run: ~$20/mes para uso moderado)
- Storage: GCS (ya lo pagas para datos)

**Total para equipo de 5:** ~$20-50/mes (si usas W&B free tier) o ~$270/mes (si usas W&B Team).

**ROI:** Si un sweep más eficiente ahorra 30 minutos de compute/día:
- Compute ahorrado: ~15 horas/mes
- En GCP: 15 horas × $2/hora (GPU) = $30/mes ahorrado solo en compute
- Más el tiempo de ingeniero (más valioso)

**Breakeven en <1 mes.**

---

### La Lección Para MLOps Engineers

**No elijas herramientas por hype o popularidad.** Elige por **responsabilidades claras**:

1. **Experimentación rápida e interactiva:** W&B, Neptune, Comet
2. **Governance y deployment:** MLflow, Seldon, BentoML
3. **Artifact storage:** GCS, S3, Azure Blob (no herramientas de tracking)

**Este proyecto usa:**
- **W&B:** Porque necesita sweep Bayesiano eficiente
- **MLflow:** Porque necesita Model Registry production-ready
- **GCS:** Porque necesita storage de alta disponibilidad

**No hay redundancia—hay especialización.**

Cuando entiendes esto, dejas de preguntar "¿W&B o MLflow?" y empiezas a preguntar "¿qué problema estoy resolviendo?"

**Esa es la diferencia entre usar herramientas y construir sistemas.**

---

<a name="docker-mlflow"></a>
## 10. Docker y MLflow: Containerización del Ecosistema Completo

### La Arquitectura de Tres Containers

Este proyecto utiliza **tres Dockerfiles distintos**, cada uno optimizado para su propósito específico:

1. **Pipeline Container (`Dockerfile`)**: Ejecuta el pipeline completo de entrenamiento con MLflow tracking
2. **API Container (`api/Dockerfile`)**: Sirve predicciones con FastAPI en producción
3. **Streamlit Container (`streamlit_app/Dockerfile`)**: Proporciona interfaz web interactiva

Esta separación no es accidental—es una decisión arquitectónica que refleja los diferentes requisitos de cada componente.
![](img/app1.png)
---

### 1. Pipeline Container: Entrenamiento con MLflow Tracking

#### Dockerfile del Pipeline

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

#### Decisiones Técnicas Críticas

**1. Por Qué `gcc` y `g++`**

```dockerfile
RUN apt-get install -y gcc g++ git curl
```

Muchos paquetes de ML (numpy, scipy, scikit-learn) compilan extensiones C/C++ durante la instalación. Sin estos compiladores, `pip install` falla con errores crípticos como "error: command 'gcc' failed".

**Trade-off:** Imagen más grande (~500MB vs ~150MB de Python slim puro), pero garantiza que todas las dependencias se instalan correctamente.

**2. Layer Caching Strategy**

```dockerfile
# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt* ./
RUN pip install ...

# Copy application code AFTER
COPY . .
```

Docker cachea layers. Si cambias código Python pero no dependencias, Docker reutiliza la layer de `pip install` (que toma 5 minutos) y solo recopia el código (10 segundos).

**Sin esta optimización:** Cada cambio de código requiere reinstalar todas las dependencias.

**3. Directory Creation for MLflow**

```dockerfile
RUN mkdir -p mlruns outputs models
```

MLflow escribe artifacts a `mlruns/` por defecto si no se configura un tracking server remoto. Si este directorio no existe con permisos correctos, MLflow falla silenciosamente.

**`outputs/`**: Para plots y análisis intermedios
**`models/`**: Para checkpoints de modelos antes de subir a GCS

#### Cómo Habilitar MLflow Tracking

**Opción 1: MLflow Local (Default)**

Cuando ejecutas el pipeline en este container, MLflow escribe a `mlruns/` dentro del container:

```bash
docker run --env-file .env housing-pipeline:latest

# MLflow escribe a /app/mlruns/
# Para ver el UI:
docker exec -it <container-id> mlflow ui --host 0.0.0.0 --port 5000
```

**Limitación:** Los runs se pierden cuando el container se detiene.

**Opción 2: MLflow Remote Tracking Server**

Para persistir runs, configura un servidor MLflow separado:

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

**Configuración en el código:**

```python
# main.py
import os
import mlflow

# Si MLFLOW_TRACKING_URI está configurado, usar ese server
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(mlflow_uri)

mlflow.set_experiment("housing_price_prediction")

with mlflow.start_run():
    # Log params, metrics, artifacts
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("mape", 7.82)
    mlflow.sklearn.log_model(model, "model")
```

**Opción 3: MLflow en Cloud (Production)**

Para producción, usa un servidor MLflow gestionado:

```bash
# Deploy MLflow a Cloud Run (serverless)
gcloud run deploy mlflow-server \
  --image ghcr.io/mlflow/mlflow:v2.9.2 \
  --platform managed \
  --region us-central1 \
  --set-env-vars="BACKEND_STORE_URI=postgresql://user:pass@host/mlflow_db,DEFAULT_ARTIFACT_ROOT=gs://bucket/mlflow" \
  --allow-unauthenticated

# Obtener URL
MLFLOW_URL=$(gcloud run services describe mlflow-server --format 'value(status.url)')

# Configurar en pipeline
export MLFLOW_TRACKING_URI=$MLFLOW_URL
```

#### Ejecución del Pipeline Container

```bash
# Build
docker build -t housing-pipeline:latest .

# Run con env vars
docker run \
  --env-file .env \
  -v $(pwd)/mlruns:/app/mlruns \
  housing-pipeline:latest

# Run con steps específicos
docker run \
  --env-file .env \
  housing-pipeline:latest \
  python main.py main.execute_steps=[03_feature_engineering,05_model_selection]

# Ver logs en tiempo real
docker logs -f <container-id>
```

**Volume Mount (`-v`)**: Monta `mlruns/` desde el host al container para persistir runs MLflow incluso después de que el container se detenga.

---

### 2. API Container: Inference en Producción

#### Dockerfile del API

```dockerfile
# =================================================================
# Dockerfile for Housing Price Prediction API
# Purpose: Production-ready FastAPI service for Cloud Run deployment
# =================================================================

FROM python:3.12-slim

LABEL maintainer="danieljimenez88m@gmail.com"
LABEL description="Housing Price Prediction API - FastAPI Service"

WORKDIR /app

# Install system dependencies (solo curl para healthcheck)
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

#### Decisiones Técnicas Críticas

**1. Imagen Más Ligera**

Comparado con el pipeline container:
- **No necesita `gcc`/`g++`**: Las dependencias ya están compiladas en wheels
- **No necesita `git`**: No clona repos
- **Solo `curl`**: Para el healthcheck

**Resultado:** Imagen de ~200MB vs ~500MB del pipeline.

**Por qué importa:** Cloud Run cobra por uso de memoria. Una imagen más pequeña = menos memoria = menos costo.

**2. Health Check Nativo**

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

Docker marca el container como "unhealthy" si el endpoint `/health` falla 3 veces consecutivas.

**Cloud Run** y **Kubernetes** usan esto para:
- No enviar tráfico a containers unhealthy
- Reiniciar containers que fallan
- Reporting de uptime

**start-period=40s**: Da 40 segundos al API para cargar el modelo antes de empezar health checks.

**3. Port Configuration Flexible**

```dockerfile
ENV PORT=8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
```

Cloud Run inyecta `PORT` como env var (puede ser 8080, 8081, etc.). El API debe leer este valor, no hardcodearlo.

**`exec`**: Reemplaza el shell process con uvicorn, permitiendo que Docker envíe signals (SIGTERM) directamente a uvicorn para graceful shutdown.

#### Cómo el API Carga el Modelo

El API tiene **tres estrategias de carga de modelo** con fallback automático:

```python
# api/app/core/model_loader.py

class ModelLoader:
    """Carga modelo desde MLflow → GCS → Local con fallback."""

    def load_model(self) -> Any:
        """Priority: MLflow > GCS > Local"""

        # Estrategia 1: Desde MLflow Registry
        if self.mlflow_model_name:
            try:
                model_uri = f"models:/{self.mlflow_model_name}/{self.mlflow_stage}"
                self._model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Loaded from MLflow: {model_uri}")
                return self._model
            except Exception as e:
                logger.warning(f"MLflow load failed: {e}, trying GCS...")

        # Estrategia 2: Desde GCS
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

        # Estrategia 3: Desde archivo local (fallback)
        if self.local_model_path and Path(self.local_model_path).exists():
            with open(self.local_model_path, 'rb') as f:
                self._model = pickle.load(f)
            logger.info(f"Loaded from local: {self.local_model_path}")
            return self._model

        raise RuntimeError("No model could be loaded from any source")
```

**Configuración con env vars:**

```bash
# Producción: Cargar desde MLflow
docker run -p 8080:8080 \
  -e MLFLOW_TRACKING_URI=https://mlflow.example.com \
  -e MLFLOW_MODEL_NAME=housing_price_model \
  -e MLFLOW_MODEL_STAGE=Production \
  housing-api:latest

# Staging: Cargar desde GCS
docker run -p 8080:8080 \
  -e GCS_BUCKET=my-bucket \
  -e GCS_MODEL_PATH=models/trained/housing_price_model.pkl \
  housing-api:latest

# Desarrollo: Cargar desde local
docker run -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -e LOCAL_MODEL_PATH=/app/models/trained/housing_price_model.pkl \
  housing-api:latest
```

---

### 3. Streamlit Container: Frontend Interactivo

#### Dockerfile de Streamlit

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

#### Decisiones Técnicas Críticas

**1. Configuración Embedded**

```dockerfile
RUN echo '...' > .streamlit/config.toml
```

Streamlit requiere configuración para correr en containers (headless mode, CORS, etc.). En lugar de commitear un archivo `config.toml` al repo, lo generamos en build time.

**Ventajas:**
- Un archivo menos en el repo
- Configuración versionada con el Dockerfile
- No hay riesgo de olvidar commitear el config

**2. Health Check de Streamlit**

```dockerfile
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1
```

Streamlit expone `/_stcore/health` automáticamente. Este endpoint retorna 200 si la app está running.

**3. Tema Personalizado**

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

El tema define los colores de botones, backgrounds, etc. Esto da consistencia visual sin necesidad de CSS custom en cada componente.

#### Cómo Streamlit Se Conecta al API

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
    # ... más inputs

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

**Configuración de la URL del API:**

```bash
# Docker Compose: Usa service name
docker-compose up
# Streamlit automáticamente recibe API_URL=http://api:8080

# Local development: Usa localhost
API_URL=http://localhost:8080 streamlit run app.py

# Production: Usa Cloud Run URL
API_URL=https://housing-api-xyz.run.app streamlit run app.py
```

---

### Docker Compose: Orquestación de los Tres Containers

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

**Decisiones Críticas:**

**1. Network Isolation**

```yaml
networks:
  - mlops-network
```

Ambos containers están en la misma red Docker, permitiendo que Streamlit llame al API usando `http://api:8080` (service name como hostname).

**Sin esto:** Tendrías que usar `http://host.docker.internal:8080` (solo funciona en Docker Desktop) o la IP del host.

**2. Volume Mount Read-Only**

```yaml
volumes:
  - ./models:/app/models:ro
```

El API monta `models/` en **read-only mode (`:ro`)**. El container puede leer el modelo pero no modificarlo.

**Por qué:** Seguridad. Si el container es comprometido, un atacante no puede sobrescribir el modelo con uno malicioso.

**3. Dependency Order**

```yaml
depends_on:
  - api
```

Docker Compose inicia el API antes que Streamlit. Esto evita que Streamlit falle al intentar conectarse a un API que aún no está corriendo.

**Limitación:** `depends_on` solo espera a que el container **inicie**, no a que el API esté **listo** (healthcheck pass). Para eso, necesitas un init container o retry logic en Streamlit.

---

### Comando Completo de Ejecución

```bash
# 1. Build todas las imágenes
docker-compose build

# 2. Entrenar el modelo (pipeline container)
docker run --env-file .env -v $(pwd)/models:/app/models housing-pipeline:latest

# 3. Iniciar API + Streamlit
docker-compose up -d

# 4. Verificar health
curl http://localhost:8080/health
curl http://localhost:8501/_stcore/health

# 5. Ver logs
docker-compose logs -f

# 6. Detener todo
docker-compose down
```

---

### Lo Que Esta Arquitectura Resuelve

**Sin containers:**
- "Funciona en mi máquina" syndrome
- Dependencias conflictivas (Python 3.9 vs 3.12)
- Setup manual en cada ambiente (dev, staging, prod)

**Con esta arquitectura:**
- **Reproducibilidad:** Mismo container corre en laptop, CI/CD, y producción
- **Isolation:** API no interfiere con Streamlit, pipeline no interfiere con API
- **Deployment:** `docker push` → `gcloud run deploy` en <5 minutos
- **Rollback:** `docker pull previous-image` → restart
- **Observability:** Health checks automáticos, logs centralizados

**El valor real:** Un data scientist sin experiencia en DevOps puede deployar a producción sin saber cómo configurar nginx, systemd, o virtual environments. Docker abstrae toda esa complejidad.

---

<a name="api-architecture"></a>
## 10.5. Arquitectura del API: FastAPI en Producción

### Por Qué Esta Sección Importa

Has visto pipelines de entrenamiento, sweep de hiperparámetros, y model registry. Pero **el 90% del tiempo, tu modelo no está entrenando—está sirviendo predicciones en producción.**

Un API mal diseñado es el cuello de botella entre un modelo excelente y un producto útil. Esta sección desmenuza cómo este proyecto construye un API production-ready, no un prototipo de tutorial.

---

### La Arquitectura General

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

**Decisión arquitectónica:** Separation of concerns por capas:

1. **Core**: Lógica de negocio (cargar modelo, logging, config)
2. **Routers**: Endpoints HTTP (rutas, validación de requests)
3. **Models**: Schemas de datos (Pydantic)

**Por qué no todo en `main.py`?** Porque cuando el API crece (agregar autenticación, rate limiting, múltiples modelos), cada capa se extiende independientemente sin tocar el resto.

---

### 1. Lifespan Management: El Patrón Que Evita Latencia en Primera Request

#### El Problema Que Resuelve

**Anti-pattern común:**

```python
# BAD: Cargar modelo en cada request
@app.post("/predict")
def predict(features):
    model = pickle.load(open("model.pkl", "rb"))  # 5 segundos cada request
    return model.predict(features)
```

**Problemas:**
- Primera request toma 5 segundos (cargar modelo)
- Cada request subsecuente también (no hay caching)
- Si 10 requests concurrentes → 10 cargas del modelo (50 segundos total)

#### La Solución: asynccontextmanager

```python
# api/app/main.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Loads the model on startup and cleans up on shutdown.
    """
    logger.info("Starting up API...")

    # STARTUP: Cargar modelo UNA VEZ
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
        model_loader.load_model()  # Toma 5 segundos, pero SOLO una vez
        logger.info(f"Model loaded: {model_loader.model_version}")

        # Guardar en app state (disponible para todos los endpoints)
        app.state.model_loader = model_loader
        app.state.wandb_logger = wandb_logger

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.warning("API will start but predictions will fail")

    yield  # API corre aquí

    # SHUTDOWN: Cleanup
    logger.info("Shutting down API...")
    wandb_logger.close()


# Usar lifespan en FastAPI
app = FastAPI(
    title="Housing Price Prediction API",
    version="1.0.0",
    lifespan=lifespan  # CRÍTICO
)
```

**Lo que hace:**

1. **Startup (antes de `yield`):**
   - Carga modelo en memoria (5 segundos, **una sola vez**)
   - Inicializa W&B logger
   - Guarda ambos en `app.state` (singleton pattern)

2. **Running (después de `yield`):**
   - Todas las requests usan el modelo cacheado en `app.state.model_loader`
   - Latencia por request: <50ms (solo inference, no I/O)

3. **Shutdown (después del context manager):**
   - Cierra W&B run (flush pending logs)
   - Libera recursos

**Resultado:**
- Primera request: <50ms (modelo ya cargado)
- Requests subsecuentes: <50ms
- 10 requests concurrentes: <100ms promedio (paralelizable)

**Trade-off:** Startup time de 5-10 segundos. Aceptable para producción—mejor que 5 segundos por request.

---

### 2. Configuration Management: Pydantic Settings con Prioridades

#### El Pattern: Settings-as-Code

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
        env_file = ".env"  # Lee de .env automáticamente
        case_sensitive = True  # MLFLOW_MODEL_NAME != mlflow_model_name
```

**Por qué Pydantic Settings:**

1. **Type Safety**: `settings.VERSION` es `str`, no `Optional[Any]`
2. **Validation**: Si `MLFLOW_MODEL_STAGE` no es string, falla en startup (no en la primera request)
3. **Auto .env loading**: No necesitas `python-dotenv` manualmente
4. **Default values**: `LOCAL_MODEL_PATH` tiene default, `MLFLOW_MODEL_NAME` no

**Uso en código:**

```python
from app.core.config import Settings

settings = Settings()  # Lee env vars + .env

if settings.MLFLOW_MODEL_NAME:  # Type-safe check
    model = load_from_mlflow(settings.MLFLOW_MODEL_NAME)
```

#### La Estrategia de Prioridades (Cascade Fallback)

```
Intenta cargar de:
1. MLflow Registry (si MLFLOW_MODEL_NAME está configurado)
   ↓ Si falla
2. GCS (si GCS_BUCKET está configurado)
   ↓ Si falla
3. Local filesystem (siempre disponible como último recurso)
   ↓ Si falla
4. API inicia pero `/predict` retorna 500
```

**Configuración por ambiente:**

```bash
# Producción (.env.production)
MLFLOW_MODEL_NAME=housing_price_model
MLFLOW_MODEL_STAGE=Production
MLFLOW_TRACKING_URI=https://mlflow.company.com
# GCS y Local quedan vacíos → no se usan

# Staging (.env.staging)
MLFLOW_MODEL_NAME=housing_price_model
MLFLOW_MODEL_STAGE=Staging
# Mismo setup, diferente stage

# Desarrollo local (.env.local)
LOCAL_MODEL_PATH=models/trained/housing_price_model.pkl
# Sin MLflow ni GCS → carga de local directo
```

**Valor:** Un solo codebase, múltiples ambientes. No hay `if ENVIRONMENT == "production"` en el código.

---

### 3. Model Loader: Multi-Source con Fallback Inteligente

#### La Arquitectura del Loader

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
        self._model: Optional[Any] = None  # Cacheado en memoria
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

        # Apply same preprocessing que el training pipeline
        processed_features = self._preprocessor.transform(features)

        # Predict
        predictions = self._model.predict(processed_features)

        return predictions

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
```

#### Decisiones Técnicas Críticas

**1. Por Qué MLflow Es Priority 1**

```python
# MLflow load
model = mlflow.sklearn.load_model("models:/housing_price_model/Production")
```

**Ventajas sobre GCS/Local:**
- **Model URI abstrae storage**: El modelo puede estar en S3, GCS, HDFS—MLflow lo resuelve
- **Stage resolution**: `Production` automáticamente resuelve a la versión correcta (v1, v2, etc.)
- **Metadata incluida**: MLflow también carga `conda.yaml`, `requirements.txt`, metadata de features
- **Rollback trivial**: Cambias stage en MLflow UI, API recarga automáticamente en próximo restart

**2. GCS Como Fallback (No Primary)**

```python
# GCS load
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("my-bucket")
blob = bucket.blob("models/trained/housing_price_model.pkl")
model_bytes = blob.download_as_bytes()
model = pickle.loads(model_bytes)
```

**Por qué no primary:**
- **No hay versionamiento:** `models/trained/housing_price_model.pkl` es siempre el "latest"—no puedes cargar v1 vs v2 sin cambiar el path
- **No metadata:** Solo obtienes el pickle, no sabes qué hiperparámetros/features espera
- **No stages:** No existe concepto de Staging vs Production

**Cuándo usar GCS como primary:**
- MLflow no está disponible (outage)
- Setup simple (solo un modelo, no necesitas registry)
- Budget constraint (evitar hosting de MLflow)

**3. Local Como Last Resort**

```python
# Local load
with open("models/trained/housing_price_model.pkl", "rb") as f:
    model = pickle.load(f)
```

**Solo para:**
- Desarrollo local (no quieres depender de GCS/MLflow)
- Debugging (modelo roto en GCS, testeas con una copia local)
- CI/CD tests (GitHub Actions no tiene acceso a GCS)

**Nunca para producción real**—si GCS y MLflow están down, tienes problemas más grandes que el modelo.

**4. Preprocessing Pipeline Embebido**

```python
self._preprocessor = HousingPreprocessor()

def predict(self, features: pd.DataFrame) -> np.ndarray:
    processed_features = self._preprocessor.transform(features)
    predictions = self._model.predict(processed_features)
    return predictions
```

**Por qué crítico:** El modelo espera features procesadas (one-hot encoding de `ocean_proximity`, feature engineering de clusters). Si el cliente envía raw features, el modelo falla.

**Opciones de implementación:**

**A) Preprocessing en el API (este proyecto):**
```python
# Cliente envía raw features
{"ocean_proximity": "NEAR BAY", "longitude": -122.23, ...}

# API aplica preprocessing
processed = preprocessor.transform(raw_features)

# Modelo recibe features procesadas
predictions = model.predict(processed)
```

**B) Preprocessing en el cliente (mal para APIs públicos):**
```python
# Cliente debe saber exact preprocessing
processed = client_side_preprocessing(raw_features)  # ¿Qué hace esto?

# API solo hace inference
predictions = model.predict(processed)
```

**Trade-offs:**

| Approach | Ventaja | Desventaja |
|----------|---------|------------|
| **Preprocessing en API** | Cliente no necesita saber preprocessing | API más complejo, latencia +5ms |
| **Preprocessing en cliente** | API simple, latencia baja | Cliente debe replicar preprocessing exacto |

**Para APIs públicos:** Siempre preprocessing en el API. Los clientes no deben conocer detalles internos del modelo.

**Para APIs internos:** Depende. Si el cliente es otro servicio que controlas, puedes hacer preprocessing ahí para reducir latencia.

---

### 4. Request/Response Validation: Pydantic Schemas

#### El Anti-Pattern: Validación Manual

```python
# BAD: Validación manual propensa a errores
@app.post("/predict")
def predict(request: dict):
    if "longitude" not in request:
        return {"error": "missing longitude"}
    if not isinstance(request["longitude"], (int, float)):
        return {"error": "longitude must be number"}
    if request["longitude"] < -180 or request["longitude"] > 180:
        return {"error": "longitude out of range"}
    # ... 50 líneas más de validación manual
```

**Problemas:**
- Código repetitivo y frágil
- Errores inconsistentes (`"missing longitude"` vs `"longitude is required"`)
- No hay documentación automática (OpenAPI)
- Difícil de testear

#### La Solución: Pydantic Schemas

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
        return v.upper()  # Normaliza a uppercase

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

**Lo que esto da automáticamente:**

1. **Validación de tipos:**
   ```json
   {"longitude": "not a number"}  // Rechazado: ValidationError
   ```

2. **Validación de rangos:**
   ```json
   {"longitude": -200}  // Rechazado: must be >= -180
   ```

3. **Validación custom:**
   ```json
   {"ocean_proximity": "INVALID"}  // Rechazado: must be one of [...]
   ```

4. **Documentación automática en `/docs`:**
   - Swagger UI muestra todos los fields
   - Descriptions, constraints, ejemplos
   - Try-it-out funciona out-of-the-box

5. **Serialización type-safe:**
   ```python
   features = HousingFeatures(**request_json)
   features.longitude  # Type: float (no Optional[Any])
   ```

#### Batch Predictions Support

```python
class PredictionRequest(BaseModel):
    """Request model for single or batch predictions."""

    instances: List[HousingFeatures] = Field(
        ...,
        description="List of housing features for prediction",
        min_length=1  # Al menos una instancia
    )
```

**Uso:**

```json
{
  "instances": [
    {"longitude": -122.23, ...},  // Predict house 1
    {"longitude": -118.45, ...},  // Predict house 2
    {"longitude": -121.89, ...}   // Predict house 3
  ]
}
```

**Por qué soportar batch:**
- **Latencia reducida:** 3 requests individuales = 150ms. 1 batch de 3 = 60ms.
- **Costo reducido:** Menos HTTP overhead (headers, handshake, etc.)
- **Inference eficiente:** El modelo puede vectorizar operaciones

**Trade-off:** Batch size muy grande (>1000) puede causar timeouts. Implementar límite:

```python
instances: List[HousingFeatures] = Field(
    ...,
    min_length=1,
    max_length=100  # Máximo 100 predicciones por request
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

**`model_version` en response:** Crucial para debugging. Si un cliente reporta predicciones incorrectas, el `model_version` te dice qué modelo usó (v1, v2, Production, etc.).

---

### 5. Router Pattern: Endpoints y Error Handling

#### La Estructura del Router

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

**Por qué `prefix="/api/v1"`:**

```
/api/v1/predict  ← Versión 1 del API
/api/v2/predict  ← Versión 2 (breaking changes)
```

Puedes correr ambas versiones simultáneamente durante migración:
- Clientes legacy usan `/api/v1/`
- Clientes nuevos usan `/api/v2/`
- Deprecas v1 después de 6 meses

**Sin versionamiento:** Breaking change → todos los clientes se rompen al mismo tiempo.

#### El Endpoint Principal: POST /api/v1/predict

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

        # 6. Log to W&B (async, no bloquea)
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
        # Validation error (ej: feature fuera de rango esperado)
        if wandb_logger:
            wandb_logger.log_error("validation_error", str(e), features_list)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )

    except Exception as e:
        # Unexpected error (ej: modelo corrupto, OOM)
        if wandb_logger:
            wandb_logger.log_error("prediction_error", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
```

**Decisiones de error handling:**

```python
try:
    # Inference
except ValueError:
    # Cliente envió datos inválidos → 400 Bad Request
    # Loguear a W&B para análisis
    return 400
except Exception:
    # Error inesperado (bug en el código/modelo) → 500 Internal Server Error
    # Loguear a W&B para alerting
    return 500
```

**Por qué distinguir 400 vs 500:**
- **400:** Culpa del cliente. No retries automáticos.
- **500:** Culpa del servidor. Cliente puede retry.

**Logging de errores a W&B:** Permite detectar patrones. Si ves 1000 `validation_error` para `ocean_proximity="INVALID"`, agregas un mensaje de error más claro.

---

### 6. W&B Logging: Observability en Producción

#### Por Qué Loguear Predicciones

**Pregunta:** "¿Para qué loguear cada predicción si ya tengo logs de uvicorn?"

**Respuesta:** Los logs de uvicorn te dicen:
- Qué endpoint se llamó
- HTTP status code
- Cuánto tardó

Los logs de W&B te dicen:
- **Qué features** se usaron
- **Qué predicción** se hizo
- **Distribución de predicciones** (¿todas están en $200k-$500k? ¿hay outliers?)
- **Latencia promedio** por request
- **Error rate** (¿cuántos requests fallan?)

**Caso de uso real:** Stakeholder reporta "las predicciones están muy altas últimamente". Abres W&B dashboard:

```
prediction/mean: $450k (antes: $380k)
features/median_income: 9.2 (antes: 7.5)
```

**Conclusión:** No hay bug—simplemente los clientes están consultando casas en áreas más caras (`median_income` más alto). Sin W&B, estarías debuggeando código por horas.

#### La Implementación

```python
# api/app/core/wandb_logger.py

class WandBLogger:
    def __init__(self, project: str = "housing-mlops-api", enabled: bool = True):
        self.enabled = enabled and bool(os.getenv("WANDB_API_KEY"))

        if self.enabled:
            self._run = wandb.init(
                project=self.project,
                job_type="api-inference",  # Distinguir de training runs
                config={
                    "environment": os.getenv("ENVIRONMENT", "production"),
                    "model_version": os.getenv("MODEL_VERSION", "unknown")
                },
                reinit=True  # Permite múltiples init() en mismo proceso
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

        # Métricas agregadas
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

**Por qué `job_type="api-inference"`:**

En W&B dashboard, puedes filtrar por job type:
- `training`: Runs del pipeline de entrenamiento
- `sweep`: Runs del hyperparameter sweep
- `api-inference`: Predicciones en producción

**Por qué `reinit=True`:** Un proceso de uvicorn puede vivir días. `reinit=True` permite crear múltiples W&B runs dentro del mismo proceso (uno por startup/restart).

**Por qué sample first 100:** Loguear 10,000 features individuales por request sería demasiado overhead. Muestrear 100 da distribución representativa sin matar performance.

#### W&B Dashboard en Producción

```
# Métricas a monitorear:

prediction/count: Requests per minute (RPM)
  - Esperado: 100-500 RPM
  - Alerta: <10 RPM (¿está caído?) o >2000 RPM (¿DDoS?)

prediction/mean: Precio promedio predicho
  - Esperado: $300k-$450k (según mercado)
  - Alerta: >$1M (modelo roto) o <$50k (data drift)

performance/response_time_ms: Latencia
  - Esperado: 30-60ms
  - Alerta: >200ms (modelo lento o CPU throttling)

error/count: Errores por minuto
  - Esperado: 0-5 errores/min
  - Alerta: >50 errores/min (investigate immediately)
```

---

### 7. CORS y Security

```python
# api/app/main.py

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Frontend local (React/Streamlit)
        "http://localhost:8080",
        "https://app.company.com",   # Frontend en producción
    ],
    allow_credentials=False,  # No cookies (API es stateless)
    allow_methods=["GET", "POST"],  # Solo métodos necesarios
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,  # Cache preflight requests por 1 hora
)
```

**Por qué restricted origins:**

**Anti-pattern (permissive):**
```python
allow_origins=["*"]  # MALO: Cualquier sitio puede llamar tu API
```

**Problema:** Un sitio malicioso `evil.com` puede hacer requests a tu API desde el navegador del usuario, potencialmente:
- Consumir tu cuota de GCP (si no hay auth)
- Hacer predicciones spam
- DoS attack

**Pattern correcto (restrictive):**
```python
allow_origins=["https://app.company.com"]  # Solo tu frontend
```

**Para desarrollo local:** Agregar `http://localhost:3000` temporalmente, remover en producción.

**Por qué `allow_credentials=False`:** Este API es stateless—no usa cookies ni sesiones. `allow_credentials=True` sería innecesario y una superficie de ataque adicional.

---

### 8. El Flujo Completo de Una Request

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

**El viaje interno (< 50ms):**

```
1. FastAPI recibe request (1ms)
   ├─ CORS middleware valida origin
   └─ Router match: POST /api/v1/predict

2. Pydantic validation (2ms)
   ├─ Parse JSON → PredictionRequest object
   ├─ Validate types (longitude: float ✓)
   ├─ Validate ranges (longitude: -122.23, dentro de [-180, 180] ✓)
   └─ Custom validator (ocean_proximity: "NEAR BAY" → válido ✓)

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

4. FastAPI serializa response (2ms)
   └─ PredictionResponse → JSON

5. HTTP response enviado (1ms)

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

### 9. Lo Que Esta Arquitectura Logra

**Sin esta arquitectura (API naive):**
- Cargar modelo en cada request (5 segundos/request)
- Validación manual propensa a errores
- Sin observability (debugging es adivinar)
- Sin versionamiento de API (breaking changes rompen clientes)
- CORS abierto (vulnerability)

**Con esta arquitectura:**
- **Latencia:** <50ms por predicción (modelo cacheado)
- **Confiabilidad:** Pydantic garantiza requests válidos antes de llegar al modelo
- **Observability:** W&B dashboard muestra distribución de predicciones, latencia, errores
- **Maintainability:** Separation of concerns (core/routers/models)
- **Security:** CORS restrictivo, error handling robusto
- **Versionamiento:** `/api/v1/` permite evolucionar el API sin romper clientes

**El valor real:** Este API puede escalar de 10 requests/min a 10,000 requests/min sin cambios en el código—solo agregar más containers con load balancer. La arquitectura ya está lista.

---

<a name="model-strategies"></a>
## 11. Estrategias de Selección de Modelos y Parámetros

---

## Navegación

**[← Parte 1: Pipeline y Orquestación](/mlops/anatomia-pipeline-mlops-parte-1/)** | **[Parte 3: Producción y Best Practices →](/mlops/anatomia-pipeline-mlops-parte-3/)**

En la Parte 3 cubriremos:
- Estrategias de selección de modelos y parámetros
- Testing: Fixtures, mocking y coverage real
- Patrones de producción (Transform Pattern, Data Drift, Feature Stores)
- Checklist de Production Readiness

