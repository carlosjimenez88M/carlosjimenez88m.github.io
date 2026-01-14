---
title: "Anatomía de un Pipeline MLOps - Parte 1: Pipeline y Orquestación"
date: 2026-01-13
draft: false
tags: ["mlops", "machine-learning", "python", "gcp", "mlflow", "wandb"]
categories: ["MLOps", "Engineering"]
author: "Carlos Daniel Jiménez"
description: "Parte 1: Filosofía, arquitectura del proyecto y orquestación con Hydra + MLflow. Steps de preprocessing, feature engineering, hyperparameter tuning y model registry."
---

> **Serie MLOps Completo:** [Parte 1 (actual)](/mlops/anatomia-pipeline-mlops-parte-1/) | [Parte 2: Deployment →](/mlops/anatomia-pipeline-mlops-parte-2/) | [Parte 3: Producción →](/mlops/anatomia-pipeline-mlops-parte-3/)

# Anatomía de un Pipeline MLOps - Parte 1: Pipeline y Orquestación

# Anatomía de un Pipeline MLOps: De los Datos Crudos al Deployment en Producción

## Por Qué Este Post No Es Otro Tutorial de Scikit-Learn

La mayoría de los posts sobre MLOps te enseñan a entrenar un Random Forest en un notebook y te dicen "ahora ponlo en producción". Este post asume que ya sabes entrenar modelos. Lo que probablemente no sabes es cómo construir un sistema donde:

- Un commit a GitHub dispara un pipeline completo de 7 steps
- Cada decisión de preprocesamiento está respaldada por métricas cuantificables
- Los modelos se versionan con metadata rica, no con nombres de archivo tipo `model_final_v3_REAL.pkl`
- El deployment no requiere SSH a un servidor para copiar un pickle
- Rollback de una versión defectuosa toma 30 segundos, no 3 horas de panic debugging

Este post disecciona un pipeline real que implementa todo eso. No es teoría, es código que corre en producción. Basado en el capítulo 2 de "Hands-On Machine Learning" de Aurélien Géron, pero con la infraestructura que el libro no cubre.

**Repositorio completo:** [github](https://github.com/carlosjimenez88M/mlops-hand-on-ML-and-pytorch/tree/cap2-end_to_end/cap2-end_to_end)

---

## Tabla de Contenidos

1. [La Filosofía: Por Qué Ser Ordenado Es Más Importante Que Ser Inteligente](#filosofía)
2. [Estructura del Proyecto: Arquitectura Que Escala](#estructura)
3. [Orquestación con Hydra + MLflow](#orquestación)
4. [Step 02: Imputación Automatizada - Decisiones Respaldadas por Datos](#step-02)
5. [Step 03: Feature Engineering - KMeans Como Feature, No Solo Clustering](#step-03)
6. [Step 06: Hyperparameter Sweep - Optimización Bayesiana con W&B](#step-06)
7. [Step 07: Model Registry - Versionamiento en MLflow](#step-07)
8. [CI/CD con GitHub Actions: Automatización del Pipeline Completo](#github-actions)
9. [El Valor de MLOps: Por Qué Esto Importa](#mlops-value-proposition)
    - W&B vs MLflow: Por Qué Ambos, No Uno u Otro (#wandb-vs-mlflow)
10. [Docker y MLflow: Containerización del Ecosistema Completo](#docker-mlflow)
    - Pipeline Container con MLflow Tracking
    - API Container para Inference
    - Streamlit Container para Frontend
    - Docker Compose: Orquestación de los Tres Containers
    - Arquitectura del API: FastAPI en Producción (#api-architecture)
11. [Estrategias de Selección de Modelos y Parámetros](#model-strategies)
    - Model Selection: Comparación de 5 Algoritmos
    - Parameter Grids y GridSearch
    - Métricas de Evaluación: MAPE, SMAPE, wMAPE
12. [Testing: Fixtures, Mocking y Coverage Real](#testing)
13. [Patrones de Producción Que Nadie Te Cuenta](#production-patterns)
    - El Transform Pattern: El Truco del KMeans Sintético
    - Training/Serving Skew: El Asesino Silencioso
    - Data Drift: El Enemigo Que Este Proyecto (Aún) No Monitorea
    - Model Monitoring: Más Allá de Accuracy
    - The Cascade Pattern: Fallback Resilience
    - Feature Store Anti-Pattern: Cuándo NO Necesitas Uno
    - Production Readiness: Un Checklist Honesto
14. [Conclusiones: MLOps Como Disciplina de Ingeniería](#conclusiones)

---

<a name="filosofía"></a>
## 1. La Filosofía: Por Qué Ser Ordenado Es Más Importante Que Ser Inteligente

### El Problema Real del MLOps

Ser un MLOps engineer tiene dos cosas importantes en su quehacer:

**Primero, y lo que siento que es lo más importante: ser ordenado.** Suena redundante, pero cada cosa debe ir en su lugar. Un notebook con 50 celdas ejecutadas en orden aleatorio no es un pipeline—es una bomba de tiempo. Cuando ese modelo necesita reentrenarse a las 3 AM porque el data drift disparó una alerta, ¿quién se acuerda del orden correcto de las celdas?

**Segundo: lo que no se prueba, no deja de ser un mock o un prototipo.** Lejos de pensar en usar solamente patrones de diseño, el foco y lo que intentaré sembrar como idea central de este post es **la usabilidad de los productos y ver esto como software design.**

### El Mindset Correcto

Este proyecto trata Machine Learning como lo que realmente es: **software con componentes probabilísticos**. No es magia, es ingeniería. Y como ingeniería, necesita:

- **Versionamiento:** De datos, código, modelos y configuración
- **Testing:** Unit, integration y end-to-end
- **Observabilidad:** Logs, métricas y traces
- **Reproducibilidad:** Ejecutar hoy y en 6 meses debe dar el mismo resultado
- **Deployment:** Automatizado, no manual

### Referencia: Hands-On Machine Learning de Géron

Este post se basa en el **Capítulo 2 del libro de Géron**, un clásico que todos deberíamos leer. Pero el libro se enfoca en el modelo—cómo entrenar un buen predictor. Este post se enfoca en el **sistema alrededor del modelo**—cómo hacer que ese predictor llegue a producción de manera confiable.

**Lo que Géron enseña:** Imputación de datos, feature engineering, selección de modelos, evaluación.

**Lo que este post agrega:** GCS para almacenamiento, W&B para experimentación, MLflow para model registry, FastAPI para serving, Docker para deployment, GitHub Actions para CI/CD.

---

<a name="estructura"></a>
## 2. Estructura del Proyecto: Arquitectura Que Escala

### El Árbol Completo (200+ Archivos)

```
cap2-end_to_end/
├── main.py                                # Orquestador Hydra + MLflow
├── config.yaml                            # Single source of truth
├── pyproject.toml                         # Dependencias con UV
├── Makefile                               # CLI para operaciones comunes
├── Dockerfile                             # Pipeline containerizado
├── docker-compose.yaml                    # API + Streamlit + MLflow
├── pytest.ini                             # Configuración de tests
├── .env.example                           # Template de secrets
│
├── src/
│   ├── data/                              # Steps de procesamiento (01-04)
│   │   ├── 01_download_data/
│   │   │   ├── main.py                    # Download desde URL → GCS
│   │   │   ├── downloader.py              # Lógica de descarga
│   │   │   ├── models.py                  # Pydantic schemas
│   │   │   ├── MLproject                  # Entry point MLflow
│   │   │   └── conda.yaml                 # Dependencias aisladas
│   │   │
│   │   ├── 02_preprocessing_and_imputation/
│   │   │   ├── main.py
│   │   │   ├── preprocessor.py
│   │   │   ├── imputation_analyzer.py     # (crítico) Comparación de estrategias
│   │   │   └── utils.py
│   │   │
│   │   ├── 03_feature_engineering/
│   │   │   ├── main.py
│   │   │   ├── feature_engineer.py        # (crítico) KMeans clustering
│   │   │   └── utils.py                   # Optimización n_clusters
│   │   │
│   │   └── 04_segregation/
│   │       ├── main.py
│   │       ├── segregator.py              # Train/test split
│   │       └── models.py
│   │
│   ├── model/                             # Steps de modelado (05-07)
│   │   ├── 05_model_selection/
│   │   │   ├── main.py                    # Comparación de 5 algoritmos
│   │   │   ├── model_selector.py          # (crítico) GridSearch por modelo
│   │   │   └── utils.py
│   │   │
│   │   ├── 06_sweep/
│   │   │   ├── main.py                    # (crítico) W&B Bayesian optimization
│   │   │   ├── sweep_config.yaml          # Espacio de búsqueda
│   │   │   └── best_params.yaml           # Output (generado)
│   │   │
│   │   └── 07_registration/
│   │       ├── main.py                    # (crítico) Registro en MLflow
│   │       └── configs/
│   │           └── model_config.yaml      # Metadata (generado)
│   │
│   └── utils/
│       └── colored_logger.py              # Logging estructurado
│
├── api/                                   # FastAPI REST API
│   ├── app/
│   │   ├── main.py                        # FastAPI + lifespan
│   │   ├── core/
│   │   │   ├── config.py                  # Pydantic Settings
│   │   │   ├── model_loader.py            # Load desde MLflow/GCS/Local
│   │   │   └── wandb_logger.py            # Logging predicciones
│   │   ├── models/
│   │   │   └── schemas.py                 # Request/Response schemas
│   │   └── routers/
│   │       └── predict.py                 # POST /api/v1/predict
│   ├── Dockerfile                         # Imagen del API (port 8080)
│   └── requirements.txt
│
├── streamlit_app/                         # Frontend interactivo
│   ├── app.py                             # Aplicación Streamlit (450+ líneas)
│   ├── Dockerfile                         # Imagen Streamlit (port 8501)
│   └── requirements.txt
│
├── tests/                                 # Suite de tests
│   ├── conftest.py                        # Fixtures compartidas
│   ├── fixtures/
│   │   └── test_data_generator.py         # Datos sintéticos
│   ├── test_pipeline.py                   # Test de orquestación
│   ├── test_downloader.py
│   ├── test_preprocessor.py
│   ├── test_imputation_analyzer.py        # (crítico) Tests de imputación
│   ├── test_feature_engineering.py
│   ├── test_segregation.py
│   └── test_integration_simple.py         # End-to-end
│
└── docs/
    ├── API_ARCHITECTURE_POST.md
    ├── QUICKSTART_GUIDE.md
    └── TESTING_IMPROVEMENTS.md
```

**Los archivos marcados con (crítico) son los más críticos** para entender la arquitectura.

### Decisiones Arquitectónicas Fundamentales

#### 1. Separación `src/data` vs `src/model`

**Por qué:** Los steps de datos (01-04) producen artifacts **reutilizables**—preprocesamiento, features, splits. Los steps de modelo (05-07) los **consumen** pero pueden reentrenarse sin reejecutar todo upstream.

**Beneficio:** Si cambias hiperparámetros, reejecutas solo 06-07. Si cambias feature engineering, reejecutas 03-07. No re-descargas datos cada vez.

**Costo:** Más verbosidad, más archivos. Pero en pipelines reales con múltiples data scientists, el aislamiento vale oro.

#### 2. MLproject + conda.yaml por Step

Cada subdirectorio es un proyecto MLflow independiente:

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

**Ventajas:**
- Dependencias aisladas (step 03 usa scikit-learn 1.3, step 06 podría usar 1.4)
- Ejecución independiente: `mlflow run src/data/02_preprocessing`
- Tracking granular: cada step es un run separado

**Desventaja:** Overhead de archivos. Pero es el mismo overhead que tener microservicios—cada uno con su Dockerfile.

#### 3. `api/` Como Proyecto Separado

El API no está en `src/api/`. Es un proyecto hermano con su propio `requirements.txt`, Dockerfile y tests.

**Razón:** El API se deploya **independientemente** del pipeline. No necesita pandas completo, scikit-learn full o W&B client. Solo FastAPI, pydantic y el pickle del modelo.

**Resultado:** Imagen Docker de 200MB vs 1.5GB si incluyeras todo el pipeline.

#### 4. Tests en la Raíz

Los tests prueban el **sistema completo**, no módulos aislados. `test_integration_simple.py` corre el pipeline end-to-end. No encaja conceptualmente en `src/`.

#### 5. Ausencia de `notebooks/`

**Decisión deliberada.** Los notebooks son excelentes para exploración, terribles para producción. Este proyecto prioriza **reproducibilidad** sobre iteración rápida.

Si necesitas explorar, úsalos localmente pero **no los comitees**. Los notebooks en git son:
- Difíciles de revisar (diffs incomprensibles)
- Imposibles de testear
- Propensos a ejecutarse fuera de orden

---

<a name="orquestación"></a>
## 3. Orquestación con Hydra + MLflow

### Por Qué No Scripts Bash Simples

Ejecutar comandos Python secuenciales funciona para pipelines simples:

```bash
python src/data/01_download_data/main.py
python src/data/02_preprocessing/main.py
python src/data/03_feature_engineering/main.py
# ...
```

**Este enfoque falla cuando necesitas:**
- Ejecutar solo steps específicos (debugging)
- Cambiar parámetros sin editar código
- Versionar configuración junto al código
- Logs estructurados de qué corrió con qué params
- Rastrear dependencias entre steps

**Hydra + MLflow resuelve todos estos problemas.**

### El Orquestador: main.py

```python
"""
MLOps Pipeline Orchestrator
Ejecuta steps secuencialmente usando MLflow + Hydra
"""
import os
import sys
import mlflow
import hydra
from omegaconf import DictConfig
from pathlib import Path
import time

def validate_environment_variables() -> None:
    """Fail fast si faltan secrets críticos."""
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
    """Convierte execute_steps de config a lista."""
    steps = config['main']['execute_steps']
    if isinstance(steps, str):
        return [s.strip() for s in steps.split(',')]
    return list(steps)

def run_step(step_name: str, step_path: Path, entry_point: str, parameters: dict):
    """Ejecuta un step como MLflow project."""
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
    """Entry point principal del pipeline."""

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
  imputation_strategy: "auto"  # Comparará 4 estrategias

feature_engineering:
  gcs_input_path: "data/02-processed/housing_processed.parquet"
  gcs_output_path: "data/03-features/housing_features.parquet"
  n_clusters: 10
  optimize_hyperparams: true  # Busca mejor K

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
  sweep_count: 50  # 50 runs de Bayesian optimization
  metric_name: "mape"
  metric_goal: "minimize"

registration:
  registered_model_name: "housing_price_model"
  model_stage: "Staging"  # O "Production"
```

### Lo Que Este Código Hace Bien

**1. Fail Fast con Validación de Environment**

Antes de gastar CPU, verifica que todas las secrets existen. El mensaje de error incluye **instrucciones** de cómo conseguir cada valor.

```
ERROR: MISSING REQUIRED ENVIRONMENT VARIABLES
===============================================
  ERROR: WANDB_API_KEY: Weights & Biases API Key

Create .env file with:
  WANDB_API_KEY=your-key
```

Esto ahorra **frustración**—especialmente para nuevos colaboradores.

**2. Ejecución Selectiva Sin Comentar Código**

Cambias `config.yaml`:

```yaml
execute_steps: ["03_feature_engineering", "05_model_selection"]
```

Y solo esos steps corren. No editas Python, no comentas imports.

**3. Separación Entre Orchestration y Logic**

`main.py` no sabe cómo descargar datos o entrenar modelos. Solo sabe cómo **invocar** scripts que lo hacen. Cada step puede desarrollarse/testearse independientemente.

**4. Logging Estructurado con Visual Hierarchy**

Los separadores (`"="*70`) y emojis no son cosmética—en un pipeline que corre 2 horas, las secciones visuales permiten **escanear rápido** para encontrar qué step falló.

---

<a name="step-02"></a>
## 4. Step 02: Imputación Automatizada - Decisiones Respaldadas por Datos

### El Problema Real

California Housing tiene ~1% de `total_bedrooms` faltantes. Opciones obvias:

1. **Drop rows** → pierdes datos
2. **Fill con median** → asumes distribución sin verificar
3. **Fill con KNN** → asumes similitud en feature space
4. **Fill con IterativeImputer** → asumes relaciones modelables

**Pregunta:** ¿Cuál es mejor?

**Respuesta incorrecta:** "KNN siempre funciona"

**Respuesta correcta:** "Probé las 4, median tuvo RMSE de 0.8, KNN de 0.6, Iterative de 0.5. Uso Iterative porque minimiza error de reconstrucción. Aquí está el plot en W&B."

### imputation_analyzer.py: El Core

```python
"""
Imputation Analyzer - Compara estrategias automáticamente
Autor: Carlos Daniel Jiménez
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
    """Resultado de una estrategia de imputación."""
    method_name: str
    rmse: float
    imputed_values: np.ndarray
    imputer: object

class ImputationAnalyzer:
    """
    Analiza y compara estrategias de imputación.
    Selecciona automáticamente la mejor basándose en RMSE.
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
        Crea validation set masked para comparar estrategias.

        Strategy:
        1. Remove rows con target faltante (no podemos validar contra NaN)
        2. Split en train/val
        3. Maskear target en val set (simular missing values)
        4. Guardar ground truth

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

        # Maskear target en val
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
        """Evalúa SimpleImputer con strategy dada."""
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
        Evalúa KNNImputer con scaling.

        CRÍTICO: KNN requiere features escaladas o explota con overflow.
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
        """Evalúa IterativeImputer con RandomForest estimator."""
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
        """Compara todas las estrategias y selecciona la mejor."""
        train_set, val_set_missing, y_val_true = self.prepare_validation_set()

        # Evaluar todos
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

        # Seleccionar mejor
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
        """Aplica el mejor imputer al dataset completo."""
        if self.best_imputer is None:
            raise ValueError("Run compare_all_methods() first")

        df_out = df.copy()
        numeric_df = df_out.select_dtypes(include=[np.number])

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            # Check si es tuple (KNN con scaler)
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
        """Crea bar plot comparando RMSE de métodos."""
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

### Decisiones Técnicas Críticas

#### 1. La Métrica: RMSE de Reconstrucción

**¿Por qué RMSE y no MAE?**

MAE trata todos los errores igual. RMSE penaliza errores grandes más fuertemente.

Si un método imputa 100 bedrooms cuando la verdad es 3, eso es **problemático**. RMSE lo castiga más que MAE. En imputación, errores grandes distorsionan el dataset más que muchos errores pequeños.

#### 2. El Validation Set Masked

```python
train_set, val_set = train_test_split(housing_known, test_size=0.2)
val_set_missing = val_set.copy()
val_set_missing[self.target_column] = np.nan
y_val_true = val_set[self.target_column].copy()
```

Este **trick es crítico**. No puedes evaluar imputation strategies en los missing values reales—no sabes la verdad. Entonces:

1. Tomas filas donde el target NO falta
2. Splits en train/val
3. Artificialmente maskeas el target en val
4. Comparas qué tan bien cada imputer reconstruye los valores que conocías

Es **validación cruzada para preprocesamiento**, no solo para modelos.

#### 3. Por Qué KNN Necesita Scaling

```python
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_set)
```

KNN calcula distancias euclidianas entre observaciones. Si una feature está en rango [0, 1] y otra en [0, 10000], **la segunda domina completamente**.

StandardScaler normaliza todo a media 0, std 1. Ahora todas las features contribuyen equitativamente.

**IterativeImputer con RandomForest NO necesita scaling**—los árboles son invariantes a escala.

#### 4. El Imputer Como Tuple

```python
if isinstance(self.best_imputer, tuple):
    scaler, imputer = self.best_imputer
    # ... apply both
```

Si KNN ganó, necesitas guardar **tanto el scaler como el imputer**. En producción, cuando llegan datos nuevos:

1. Escalar con el mismo scaler fitted en training
2. Aplicar KNN imputer
3. Inverse transform para volver a escala original

Guardar solo el imputer sin el scaler **rompería todo**.

### Uso en el Pipeline

```python
# En main.py del Step 02
import wandb
import mlflow

analyzer = ImputationAnalyzer(df, target_column="total_bedrooms")
results = analyzer.compare_all_methods()

# Log a W&B
comparison_plot = analyzer.create_comparison_plot()
wandb.log({
    "imputation/comparison": wandb.Image(comparison_plot),
    "imputation/best_method": analyzer.best_method,
    "imputation/best_rmse": results[analyzer.best_method].rmse,
})

# Aplicar al dataset completo
housing_clean = analyzer.apply_best_imputer(housing_df)

# Guardar imputer
import joblib
joblib.dump(analyzer.best_imputer, "artifacts/imputer.pkl")
mlflow.log_artifact("artifacts/imputer.pkl")
```

### Lo Que Esto Logra

**Sin esto:** "Usé median porque es lo que hace todo el mundo."

**Con esto:** "Comparé 4 estrategias. IterativeImputer con RandomForest tuvo 15% menor RMSE que median. Aquí está el plot en W&B dashboard run `abc123`. El imputer está serializado en MLflow."

Ahora tienes **evidencia cuantificable** de por qué elegiste lo que elegiste. Seis meses después, cuando alguien pregunta, **los datos están ahí**.

---

<a name="step-03"></a>
## 5. Step 03: Feature Engineering - KMeans Como Feature, No Solo Clustering

### El Problema Real

California tiene patrones geográficos fuertes. Casas en San Francisco se comportan diferente que casas en el valle central. Pero latitude/longitude como features crudas no capturan esto bien—un modelo lineal no puede aprender "esta área es cara".

**Solución:** Clustering geográfico. Pero no para segmentar datos, sino para **crear una feature categórica**: `cluster_label`.

### ClusterSimilarity: Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Custom transformer para clustering geográfico.

    Design: Transformer de scikit-learn para integrarse en Pipeline.
    """

    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma  # Placeholder para RBF kernel (no usado actualmente)
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        """Fit KMeans en coordenadas geográficas."""
        self.kmeans_ = KMeans(
            self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        """Transforma coordenadas a cluster labels."""
        cluster_labels = self.kmeans_.predict(X)
        return np.expand_dims(cluster_labels, axis=1)

    def get_feature_names_out(self, names=None):
        """Retorna nombres de features para Pipeline."""
        return ["cluster_label"]
```

### El Pipeline de Preprocessing Completo

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def create_preprocessing_pipeline(n_clusters=10):
    """
    Crea pipeline que procesa:
    - Numéricas: impute + scale
    - Categóricas: impute + one-hot
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

### Optimización Automática de n_clusters

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

def optimize_n_clusters(
    df: pd.DataFrame,
    min_clusters=2,
    max_clusters=20
) -> Tuple[int, Dict]:
    """
    Busca el mejor K para KMeans usando silhouette score.

    Métricas:
    - Silhouette score (0 a 1): Separación de clusters. Maximizar.
    - Davies-Bouldin index: Dispersión interna vs separación. Minimizar.
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

    # Seleccionar K con mejor silhouette
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

### Visualización: Elbow Method + Silhouette

```python
def create_optimization_plots(metrics: Dict) -> Dict[str, plt.Figure]:
    """Crea plots de optimización de K."""

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

### Uso en el Pipeline

```python
# En main.py del Step 03
import wandb
import mlflow
import joblib

# Descargar datos desde GCS
df = download_from_gcs(bucket, "data/02-processed/housing_processed.parquet")

# Optimizar K
optimal_k, metrics = optimize_n_clusters(df, min_clusters=5, max_clusters=15)

print(f"Optimal K: {optimal_k}")
print(f"   Silhouette: {metrics['best_silhouette']:.4f}")

# Crear plots
plots = create_optimization_plots(metrics)

# Log a W&B
wandb.log({
    "optimization/optimal_k": optimal_k,
    "optimization/silhouette": metrics["best_silhouette"],
    "optimization/elbow_plot": wandb.Image(plots["elbow_method"]),
    "optimization/silhouette_plot": wandb.Image(plots["silhouette_scores"]),
})

# Crear pipeline con K óptimo
preprocessing_pipeline = create_preprocessing_pipeline(n_clusters=optimal_k)

# Fit pipeline
target_column = "median_house_value"
y = df[target_column]
X = df.drop(columns=[target_column])

preprocessing_pipeline.fit(X, y)

# Transform data
X_transformed = preprocessing_pipeline.transform(X)

# Reconstruir DataFrame con target
df_transformed = pd.DataFrame(
    X_transformed,
    columns=preprocessing_pipeline.get_feature_names_out()
)
df_transformed[target_column] = y.values

# Upload a GCS
upload_to_gcs(df_transformed, bucket, "data/03-features/housing_features.parquet")

# Guardar pipeline
joblib.dump(preprocessing_pipeline, "artifacts/preprocessing_pipeline.pkl")
mlflow.log_artifact("artifacts/preprocessing_pipeline.pkl")
```

### Decisiones Técnicas Críticas

#### 1. Por Qué Silhouette Score

**Silhouette score** (rango 0 a 1) mide qué tan bien separados están los clusters:

- **1.0:** Clusters perfectamente separados
- **0.5:** Overlap moderado
- **0.0:** Clusters aleatorios

Es **interpretable** y generalmente correlaciona bien con calidad visual de clusters.

**Davies-Bouldin index:** También lo calculamos pero no lo usamos para decisión—es más sensible a outliers.

#### 2. La Crítica Obvia

Este código optimiza `n_clusters` basándose en **métricas de clustering**, no en **performance del modelo final**.

Un approach más riguroso sería:

```python
for k in range(5, 15):
    pipeline = create_preprocessing_pipeline(n_clusters=k)
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)

    model = RandomForestRegressor()
    model.fit(X_train_transformed, y_train)

    mape = calculate_mape(model, X_test_transformed, y_test)
    # Seleccionar K con mejor MAPE
```

Esto tomaría **10x más tiempo** pero sería más riguroso.

**Trade-off:** Este pipeline prioriza velocidad sobre rigor absoluto. Para California Housing, silhouette score es suficientemente bueno. Para datasets más complejos, considera el approach de cross-validation completo.

#### 3. handle_unknown="ignore" en OneHotEncoder

```python
OneHotEncoder(handle_unknown="ignore")
```

**Crítico para producción.** Si en training tienes categorías `["<1H OCEAN", "INLAND", "NEAR BAY"]` pero en producción llega `"ISLAND"` (que no viste), el encoder:

- **Sin `handle_unknown`:** Explota con ValueError
- **Con `handle_unknown="ignore"`:** Genera vector de ceros para esa observación

Pierdes información de esa observación, pero el **API no devuelve HTTP 500**.

#### 4. Por Qué Guardar el Pipeline, No Solo el Modelo

```python
joblib.dump(preprocessing_pipeline, "artifacts/preprocessing_pipeline.pkl")
```

En producción, necesitas:

1. Cargar el pipeline
2. Transform datos nuevos
3. Predecir con el modelo

Si solo guardas el modelo, no sabes:
- Qué features espera
- En qué orden
- Qué transformaciones aplicar

El pipeline **encapsula todo eso**.

### Lo Que Esto Logra

**Sin esto:** "Usé KMeans con K=10 porque leí que 10 clusters es bueno."

**Con esto:** "Probé K de 5 a 15. K=8 maximizó silhouette score (0.64). Aquí están los plots de elbow method y silhouette. El pipeline con K=8 está serializado en MLflow."

**Evidencia cuantificable + artifact reproducible.**

---

<a name="step-06"></a>
## 6. Step 06: Hyperparameter Sweep - Optimización Bayesiana con W&B

### El Problema de Model Selection vs Hyperparameter Tuning

La mayoría de los proyectos de ML cometen este error: entrenan un Random Forest en un notebook, ajustan algunos hiperparámetros hasta que R² se ve "bien" y declaran victoria. Tres meses después, cuando alguien pregunta "¿por qué Random Forest y no XGBoost?", la respuesta es silencio incómodo.

**Este pipeline separa dos fases:**

1. **Model Selection (Step 05):** Compara algoritmos con GridSearch rápido (5-10 combos por modelo)
2. **Hyperparameter Sweep (Step 06):** Optimiza el ganador con Bayesian search exhaustivo (50+ runs)

**Razón:** No tienes tiempo ni cómputo para hacer sweep exhaustivo de 5 algoritmos. Primero decides **estrategia** (qué algoritmo), luego **tácticas** (qué hiperparámetros).

### sweep_config.yaml: El Espacio de Búsqueda

```yaml
# =================================================================
# W&B Sweep Configuration for Random Forest
# Autor: Carlos Daniel Jiménez
# =================================================================

program: main.py
method: bayes  # Bayesian optimization, no random, no grid

metric:
  name: wmape  # Weighted MAPE (menos sesgado que MAPE)
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

# Early stopping: elimina runs pobres temprano
early_terminate:
  type: hyperband
  min_iter: 10   # Mínimo 10 runs antes de terminar
  eta: 3         # Elimina 1/3 de runs pobres
  s: 2

name: housing-rf-sweep-improved
description: "Optimize Random Forest with wmape + feature tracking"
```

### main.py del Step 06: El Sweep Real

```python
"""
W&B Sweep for Random Forest Hyperparameter Optimization.
Autor: Carlos Daniel Jiménez
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

# Module-level data cache (cargado una vez, reusado en todos los runs)
_data_cache = {
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "feature_names": None
}

def train():
    """
    Training function llamada por W&B Sweep agent.
    Ejecutada para cada combinación de hiperparámetros.

    Usa module-level cache para evitar recargar datos en cada run.
    """
    run = wandb.init()
    config = wandb.config

    logger.info("="*70)
    logger.info(f"SWEEP RUN: {run.name}")
    logger.info("="*70)

    try:
        # Preparar parámetros
        params = {
            'n_estimators': int(config.n_estimators),
            'max_depth': int(config.max_depth) if config.max_depth else None,
            'min_samples_split': int(config.min_samples_split),
            'min_samples_leaf': int(config.min_samples_leaf),
            'max_features': config.max_features,
            'random_state': 42
        }

        # Train model usando cached data
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

        # Log todo a W&B
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
    """Main function para inicializar y ejecutar el sweep."""
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

    # Cargar datos UNA VEZ en module-level cache
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

    # Guardar best params
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

    # Guardar a YAML
    best_params_path = Path(__file__).parent / "best_params.yaml"
    with open(best_params_path, 'w') as f:
        yaml.dump(best_params, f)

    logger.info(f"\nBest params saved to: {best_params_path}")
    logger.info(f"   MAPE: {best_params['metrics']['mape']:.2f}%")

if __name__ == "__main__":
    main()
```

### Decisiones Técnicas Críticas

#### 1. Bayesian Optimization, No Random Search

```yaml
method: bayes  # No random, no grid
```

**Random search:** Prueba combinaciones aleatorias. No aprende de runs anteriores.

**Grid search:** Prueba todas las combinaciones. Exhaustivo pero **carísimo** (5 × 4 × 3 × 3 × 2 = 360 combos).

**Bayesian optimization:** Construye un modelo probabilístico de la función que optimizas (MAPE en función de hiperparámetros) y usa ese modelo para decidir qué probar siguiente.

Si detecta que `max_depth=None` consistentemente da mejor MAPE, **explora más en esa región** del espacio.

**50 runs es <15% del espacio total**, pero capturan el 80% del beneficio posible.

#### 2. wMAPE, No MAPE

```yaml
metric:
  name: wmape  # Weighted MAPE
```

**MAPE estándar:** Penaliza errores en casas baratas más que en casas caras.

Si una casa vale $10,000 y predices $12,000, error = 20%.
Si una casa vale $500,000 y predices $510,000, error = 2%.

Ambos errores son **$10,000**, pero MAPE los ve radicalmente diferentes.

**wMAPE (Weighted MAPE):** Pondera por el valor real. Menos sesgado hacia valores bajos.

**Por qué funciona aquí:** California Housing no tiene casas de $0. Rango está entre $15k y $500k—razonablemente acotado.

#### 3. Variables Globales Para Data Cache

```python
_data_cache = {
    "X_train": None,
    "X_test": None,
    # ...
}
```

Las variables globales son generalmente código sucio. **Aquí son la decisión correcta.**

Cada run del sweep necesita los mismos datos. Sin cache, cargarías desde GCS **50 veces**. Con California Housing (20k filas), eso son segundos desperdiciados. Con datasets más grandes, son **minutos u horas**.

**Alternativa "limpia":** Pasar datos como argumento a cada función. Pero W&B Sweeps tiene interfaz fija—la función que pasas a `wandb.agent()` no puede recibir argumentos adicionales.

Las variables globales aquí tienen **scope limitado**—solo existen durante el proceso del sweep.

#### 4. Early Stopping con Hyperband

```yaml
early_terminate:
  type: hyperband
  min_iter: 10
  eta: 3
```

**Hyperband** elimina runs pobres temprano. Si después de 10 runs un set de hiperparámetros muestra MAPE de 25% mientras otros están en 8%, Hyperband lo **detiene**.

**eta=3:** Elimina el peor tercio de runs en cada iteración.

**Beneficio:** Ahorras cómputo en hiperparámetros obviamente malos.

#### 5. Feature Importances Loggeadas

```python
feature_importances = log_feature_importances(model, feature_names)
wandb.log({
    **{f"feature_importance_{k}": v
       for k, v in list(feature_importances.items())[:10]}
})
```

Random Forest calcula feature importances **gratis**. Sería valioso loggearlo para entender qué features dominan el modelo.

En W&B dashboard, puedes comparar runs y ver "en el mejor run, `median_income` tuvo importance de 0.45".

### El Output Crítico: best_params.yaml

```yaml
hyperparameters:
  n_estimators: 200
  max_depth: 20
  min_samples_split: 2
  min_samples_leaf: 1
  max_features: sqrt

metrics:
  mape: 7.82
  wmape: 7.65
  r2: 0.87

sweep_id: abc123xyz
best_run_id: run_456
```

Los hiperparámetros óptimos se guardan en **YAML**, not pickle. Razón:

**YAML es legible y git-friendly.** Si en el próximo retraining cambias de `n_estimators=200` a `n_estimators=300`, un `git diff` lo muestra claramente.

Con pickle, es un **blob binario opaco**.

### Lo Que Esto Logra

**Sin esto:** "Usé `n_estimators=100` porque es el default de scikit-learn."

**Con esto:** "Corrí sweep bayesiano de 50 runs. Optimal config: `n_estimators=200, max_depth=20`. MAPE mejoró de 8.5% a 7.8%. Aquí está el sweep en W&B: `wandb.ai/project/sweeps/abc123`."

**Evidencia cuantificable** de por qué elegiste cada hiperparámetro.

---

<a name="step-07"></a>
## 7. Step 07: Model Registry - Versionamiento en MLflow

### Por Qué No Basta con Guardar el Pickle

La tentación es:

```python
import joblib
joblib.dump(model, "best_model.pkl")
```

Esto funciona hasta que necesitas responder:

- ¿Qué hiperparámetros usó?
- ¿Con qué datos se entrenó?
- ¿Qué métricas logró?
- ¿Cómo rollback a la versión anterior?

**MLflow Model Registry** resuelve esto.

### register_model_to_mlflow(): El Core

```python
"""
Registro de modelo en MLflow Model Registry.
Autor: Carlos Daniel Jiménez
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
    Registra modelo en MLflow Model Registry con metadata rica.

    Args:
        model: Trained sklearn model
        model_name: Nombre para registered model
        model_stage: Stage (Staging/Production)
        params: Hyperparameters
        metrics: Métricas de evaluación
        feature_columns: Lista de features
        target_column: Target column name
        gcs_train_path: Path a training data
        gcs_test_path: Path a test data

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

### Decisiones Técnicas Críticas

#### 1. Artifact vs Registered Model

**Artifact:** Pickle guardado en un run específico. Para usarlo, necesitas el `run_id`.

```python
mlflow.sklearn.log_model(model, "model")  # Solo artifact
# Uso: mlflow.sklearn.load_model(f"runs://{run_id}/model")
```

**Registered Model:** Versionado con nombre semántico, stages y metadata.

```python
client.create_model_version(name="housing_price_model", source=model_uri)
# Uso: mlflow.pyfunc.load_model("models:/housing_price_model/Production")
```

En producción, tu API carga `models:/housing_price_model/Production`, **no `runs:/abc123/model`**.

Cuando registras una nueva versión, la transicionas a Production y el deployment **automáticamente** toma la nueva versión.

#### 2. Metadata Rica en Markdown

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

Esto guarda **markdown en la descripción** del modelo. Cuando abres MLflow UI y navegas a `housing_price_model v3`, ves:

- Qué hiperparámetros usó
- Qué métricas logró
- De dónde vinieron los datos

**Por qué es oro:** Seis meses después, cuando alguien pregunta "¿por qué el modelo v3 tiene mejor MAPE que v2?", abres MLflow y **la respuesta está ahí**.

No necesitas buscar en logs ni preguntar a quien lo entrenó.

#### 3. Tags Para Búsqueda

```python
tags = {
    "algorithm": "RandomForest",
    "mape": f"{metrics['mape']:.2f}",
    "r2": f"{metrics['r2']:.4f}",
}

for key, value in tags.items():
    client.set_model_version_tag(model_name, model_version.version, key, value)
```

En MLflow puedes **filtrar modelos por tags**. "Muéstrame todos los modelos con MAPE < 8%" es una query que funciona si taggeaste consistentemente.

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

Este YAML se loggea a MLflow **Y** se guarda en el repo (en `configs/model_config.yaml`).

**Por qué YAML y no solo MLflow:** Tu FastAPI app necesita leer configuración al iniciar. Puede hacer `mlflow.load_model()` para el pickle, pero necesita saber los **feature names** para validación de input.

El YAML es esa **single source of truth**.

#### 5. Versionado en Git

Cuando commiteas `model_config.yaml`, el diff muestra:

```diff
- version: 2
+ version: 3
- mape: 8.5
+ mape: 7.8
- n_estimators: 100
+ n_estimators: 200
```

Es **auditable**. Sabes exactamente qué cambió entre versiones.

### El Flujo Completo: Sweep → Registration → Production

```bash
# 1. Model Selection (Step 05)
python src/model/05_model_selection/main.py
# Output: "Best: RandomForestRegressor (MAPE: 8.2%)"

# 2. Hyperparameter Sweep (Step 06)
python src/model/06_sweep/main.py --sweep_count=50
# Output: best_params.yaml con hiperparámetros óptimos

# 3. Model Registration (Step 07)
python src/model/07_registration/main.py --params_file=best_params.yaml
# Output: Modelo registrado en MLflow Registry

# 4. Transition to Production (manual)
mlflow models transition \
  --name housing_price_model \
  --version 3 \
  --stage Production
```

### Lo Que Este Approach Soluciona

**Sin Model Registry:**
- Pickles en carpetas: `model_v3_final_FINAL_2.pkl`
- No sabes qué hiperparámetros usa cada uno
- Rollback = buscar el pickle correcto en GCS

**Con Model Registry:**
- Modelos con versiones semánticas: v1, v2, v3
- Metadata embebida: params, metrics, data sources
- Rollback = `transition v3 to Archived` + `transition v2 to Production`

---

<a name="github-actions"></a>
## 8. CI/CD con GitHub Actions: Automatización del Pipeline Completo

---

## Navegación

**[← Inicio](/mlops/)** | **[Parte 2: Deployment e Infraestructura →](/mlops/anatomia-pipeline-mlops-parte-2/)**

En la Parte 2 cubriremos:
- CI/CD con GitHub Actions
- W&B vs MLflow: Estrategias complementarias
- Containerización completa con Docker
- Arquitectura FastAPI en producción

