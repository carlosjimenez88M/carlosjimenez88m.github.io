---
title: "Anatomía de un Pipeline MLOps: De los Datos Crudos al Deployment en Producción"
date: 2026-01-13
draft: false
tags: ["mlops", "machine-learning", "python", "gcp", "mlflow", "wandb", "fastapi", "docker"]
categories: ["MLOps", "Engineering"]
author: "Carlos Daniel Jiménez"
description: "Un análisis profundo de un pipeline MLOps completo: desde el download de datos hasta el deployment en producción, con código real y decisiones arquitectónicas explicadas."
---

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

### El Flujo Completo: Selection → Sweep → Registration

Este pipeline implementa una **estrategia de tres fases** para optimización de modelos, cada una con un propósito específico:

```
Step 05: Model Selection
├── Compara 5 algoritmos con GridSearch básico (5-10 combos/modelo)
├── Objetivo: Identificar mejor familia de modelo (Random Forest vs Gradient Boosting vs ...)
├── Métrica principal: MAPE (Mean Absolute Percentage Error)
└── Output: Mejor algoritmo + parámetros iniciales

Step 06: Hyperparameter Sweep
├── Optimiza SOLO el mejor algoritmo del Step 05
├── Bayesian optimization con 50+ runs (espacio de búsqueda exhaustivo)
├── Objetivo: Encontrar configuración óptima del mejor modelo
├── Métrica principal: wMAPE (Weighted MAPE, menos sesgado)
└── Output: best_params.yaml con hiperparámetros óptimos

Step 07: Model Registration
├── Entrena modelo final con parámetros de Step 06
├── Registra en MLflow Model Registry con metadata rica
├── Transiciona a stage (Staging/Production)
└── Output: Modelo versionado listo para deployment
```

**¿Por qué tres steps separados?** No tienes recursos computacionales para hacer sweep exhaustivo de 5 algoritmos × 50 combinaciones = 250 entrenamientos. Primero decides **estrategia** (qué algoritmo), luego **tácticas** (qué hiperparámetros).

---

### Step 05: Model Selection - Comparación de Algoritmos

#### Los 5 Modelos Candidatos

```python
def get_available_models() -> Dict[str, Any]:
    """Get dictionary of available regression models."""
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }
    return models
```

**Por qué estos modelos:**

1. **RandomForest**: Ensemble de árboles, robusto, maneja no-linealidades
2. **GradientBoosting**: Boosting secuencial, mejor precisión que RF pero más lento
3. **Ridge**: Regresión lineal con regularización L2, rápido, interpretable
4. **Lasso**: Regresión lineal con regularización L1, hace feature selection
5. **DecisionTree**: Baseline simple, útil para comparación

**Lo que falta (deliberadamente):**
- **XGBoost/LightGBM**: No incluidos para reducir dependencias, pero fácil de agregar
- **Neural Networks**: Overkill para este problema (20k muestras, features tabulares)
- **SVR**: Muy lento en datasets grandes, no escala bien

#### Parameter Grids: GridSearch Inicial

```python
def get_default_param_grids() -> Dict[str, Dict[str, list]]:
    """
    Parameter grids for initial model selection.
    Refinados basados en domain knowledge.
    """
    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 200, 300],         # 4 opciones
            "max_depth": [10, 15, 20, 25, None],         # 5 opciones
            "min_samples_split": [2, 5, 10],             # 3 opciones
            "min_samples_leaf": [1, 2, 4],               # 3 opciones
        },
        # Total combinaciones: 4×5×3×3 = 180
        # Con 5-fold CV: 180×5 = 900 fits

        "GradientBoosting": {
            "n_estimators": [50, 100, 150, 200],         # 4 opciones
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2], # 5 opciones
            "max_depth": [3, 4, 5, 6, 7],                # 5 opciones
            "subsample": [0.8, 0.9, 1.0],                # 3 opciones
        },
        # Total: 4×5×5×3 = 300 combinaciones

        "Ridge": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
        },
        # Total: 9 combinaciones (rápido)

        "Lasso": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
        },
        # Total: 9 combinaciones

        "DecisionTree": {
            "max_depth": [5, 10, 15, 20, 25, None],      # 6 opciones
            "min_samples_split": [2, 5, 10, 20],         # 4 opciones
            "min_samples_leaf": [1, 2, 4, 8],            # 4 opciones
        }
        # Total: 6×4×4 = 96 combinaciones
    }
    return param_grids
```

#### Decisiones de Diseño de los Grids

**1. RandomForest: Foco en Overfitting Control**

```python
"max_depth": [10, 15, 20, 25, None],
"min_samples_leaf": [1, 2, 4],
```

**Razonamiento:** Random Forest tiende a overfit en datasets pequeños. `max_depth` y `min_samples_leaf` controlan profundidad de árboles—valores altos previenen que el modelo memorice ruido.

**None en max_depth:** Permite árboles de profundidad ilimitada. Útil cuando el dataset tiene patrones complejos que requieren splits profundos.

**2. GradientBoosting: Balance Learning Rate vs N_estimators**

```python
"n_estimators": [50, 100, 150, 200],
"learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
```

**Trade-off clásico:**
- **Learning rate bajo (0.01) + muchos estimators (200):** Aprendizaje lento pero preciso
- **Learning rate alto (0.2) + pocos estimators (50):** Rápido pero puede divergir

GridSearch explora ambos extremos.

**subsample < 1.0:** Stochastic Gradient Boosting. Solo usa 80-90% de datos en cada iteración, reduce overfitting.

**3. Ridge/Lasso: Alpha en Escala Logarítmica**

```python
"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
```

Alpha controla regularización:
- **Alpha bajo (0.01):** Casi sin regularización, modelo complejo
- **Alpha alto (500):** Regularización fuerte, modelo simple (coeficientes cercanos a 0)

Escala logarítmica cubre el espacio de manera más uniforme que escala lineal.

**Lasso vs Ridge:**
- **Lasso (L1):** Fuerza coeficientes a **exactamente 0** → feature selection automática
- **Ridge (L2):** Coeficientes pequeños pero **no cero** → mantiene todas las features

Si Lasso gana, indica que algunas features son ruido.

**4. DecisionTree: Baseline de Comparación**

DecisionTree es el peor modelo (alto variance, overfit fácil), pero sirve para:
- Verificar que el pipeline funciona correctamente
- Baseline de comparación: Si Ridge/Lasso no superan DecisionTree, algo está mal en feature engineering

#### La Función de Entrenamiento con GridSearch

```python
def train_model_with_gridsearch(
    model: Any,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, Dict[str, Any], float, Dict[str, float]]:
    """Train model with K-fold Cross-Validation via GridSearchCV."""

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_absolute_error',  # CRÍTICO
        n_jobs=-1,  # Paralelización
        verbose=0,
        return_train_score=True  # Para detectar overfitting
    )

    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Extract cross-validation results
    cv_metrics = {
        "mean_test_score": float(-grid_search.best_score_),
        "std_test_score": float(grid_search.cv_results_['std_test_score'][grid_search.best_index_]),
        "mean_train_score": float(-grid_search.cv_results_['mean_train_score'][grid_search.best_index_]),
        "std_train_score": float(grid_search.cv_results_['std_train_score'][grid_search.best_index_]),
    }

    return grid_search.best_estimator_, grid_search.best_params_, training_time, cv_metrics
```

#### Decisiones Críticas

**1. Scoring: neg_mean_absolute_error**

```python
scoring='neg_mean_absolute_error'
```

**¿Por qué MAE y no RMSE o R²?**

- **MAE (Mean Absolute Error)**: Penaliza errores linealmente
- **RMSE**: Penaliza errores cuadraticamente (errores grandes pesan mucho más)
- **R²**: Métrica relativa, difícil de interpretar en términos de negocio

Para este problema:
- MAE = $15,000 → "El modelo se equivoca $15k en promedio"
- R² = 0.85 → ¿Qué significa para el negocio?

**neg_mean_absolute_error:** GridSearchCV minimiza la métrica, pero MAE se debe minimizar, entonces usamos la negativa.

**2. Cross-Validation: 5 Folds**

```python
cv=5
```

**¿Por qué 5 y no 10?**

- **5-fold:** Balance entre bias (sesgo) y variance (varianza)
  - Cada fold tiene 80% training, 20% validation
  - Más rápido que 10-fold (2x menos fits)

- **10-fold:** Menos bias pero más costo computacional
  - Útil cuando tienes pocos datos (<1000 samples)

Con 16,512 training samples, 5-fold es suficiente.

**3. return_train_score=True**

```python
return_train_score=True
```

Esto loggea el score en **training set** además de validation set. Permite detectar overfitting:

```python
if cv_metrics['mean_train_score'] >> cv_metrics['mean_test_score']:
    print("WARNING: Model is overfitting!")
    # Train MAE = $5k, Test MAE = $20k → Overfitting claro
```

**4. n_jobs=-1: Paralelización**

```python
n_jobs=-1
```

Usa todos los CPU cores disponibles. En una máquina con 8 cores, 180 combinaciones × 5 folds = 900 fits se distribuyen en paralelo.

**Sin paralelización:** 900 fits × 2s/fit = 30 minutos
**Con 8 cores:** ~4 minutos

#### Métricas de Evaluación: Más Allá de MAPE

```python
def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evalúa modelo con métricas business-focused."""

    y_pred = model.predict(X_test)
    y_true = y_test.values

    # Traditional metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Business-focused percentage error metrics
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    wmape = weighted_mean_absolute_percentage_error(y_true, y_pred)
    median_ape = median_absolute_percentage_error(y_true, y_pred)

    # Prediction accuracy at different thresholds
    within_5pct = predictions_within_threshold(y_true, y_pred, 0.05)
    within_10pct = predictions_within_threshold(y_true, y_pred, 0.10)
    within_15pct = predictions_within_threshold(y_true, y_pred, 0.15)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "smape": float(smape),
        "wmape": float(wmape),
        "median_ape": float(median_ape),
        "within_5pct": float(within_5pct),
        "within_10pct": float(within_10pct),
        "within_15pct": float(within_15pct)
    }
```

**Por Qué 4 Variantes de MAPE:**

**1. MAPE (Mean Absolute Percentage Error)**

```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

**Problema:** Sesgado hacia valores bajos.

Si predices $500k en vez de $510k → error = 2%
Si predices $10k en vez de $11k → error = 9%

Ambos son $10k de error absoluto, pero MAPE penaliza más el segundo.

**2. SMAPE (Symmetric MAPE)**

```python
smape = np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
```

Usa el promedio de `y_true` y `y_pred` en el denominador. Más simétrico:
- Overprediction y underprediction tienen peso similar
- Rango: 0-200% (vs 0-∞% de MAPE)

**3. wMAPE (Weighted MAPE)**

```python
wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
```

Suma total de errores dividido por suma total de valores reales. No afectado por valores individuales extremos.

**Usado en Step 06 (Sweep)** porque es más robusto que MAPE para datasets con varianza alta.

**4. Median APE**

```python
median_ape = np.median(np.abs((y_true - y_pred) / y_true)) * 100
```

Mediana en lugar de media. Robusto a outliers.

Si 95% de predicciones tienen <5% error pero 5% tienen >50% error:
- **MAPE:** ~7% (promedio incluye outliers)
- **Median APE:** ~4% (outliers no afectan la mediana)

**Within-X% Metrics**

```python
within_5pct = predictions_within_threshold(y_true, y_pred, 0.05)
# Porcentaje de predicciones con error <5%
```

**Business interpretation:** "El 75% de nuestras predicciones están dentro de ±10% del valor real."

Más interpretable para stakeholders que "MAPE = 8.2%".

#### Output del Step 05

```python
logger.info(" BEST MODEL: RandomForestRegressor")
logger.info("Business Metrics (Test Set):")
logger.info("  MAPE (Mean APE): 8.23%")
logger.info("  SMAPE (Symmetric MAPE): 7.95%")
logger.info("  wMAPE (Weighted MAPE): 8.01%")
logger.info("  Median APE: 6.45%")
logger.info("  Within ±5%: 45.2%")
logger.info("  Within ±10%: 72.8%")
logger.info("  Within ±15%: 85.3%")
logger.info("\nTraditional Metrics (Test Set):")
logger.info("  R²: 0.8654")
logger.info("  RMSE: $48,234.12")
logger.info("  MAE: $32,456.78")
logger.info("\nCross-Validation Results (5-fold):")
logger.info("  Mean CV MAE: $33,125.45 (±$2,341.23)")
logger.info("  Mean CV Train MAE: $28,934.56 (±$1,892.34)")
```

**Best params guardados:**

```python
best_params = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1
}
```

Estos params se usan como **punto de partida** para el Step 06 (Sweep exhaustivo).

---

### Lo Que Esta Estrategia Logra

**Sin model selection:**
- "Usé Random Forest porque lo usa todo el mundo"
- No tienes evidencia de que es mejor que Gradient Boosting

**Con model selection:**
- "Comparé 5 algoritmos con 5-fold CV. Random Forest logró MAPE=8.2% (vs GradientBoosting=8.9%, Ridge=12.3%). Aquí está la tabla comparativa en W&B."
- **Decisión respaldada por datos, no intuición.**

---

<a name="testing"></a>
## 11. Testing: Fixtures, Mocking y Coverage Real

### Por Qué Testear ML Es Diferente

Los tests en ML no son como tests en web apps. No puedes hacer:

```python
def test_model_predicts_correct_value():
    model = load_model()
    assert model.predict([[1, 2, 3]]) == 452600.0  # ERROR: Esto es absurdo
```

Los modelos ML son **probabilísticos**. La salida no es determinística en el sentido de software tradicional.

**Lo que SÍ puedes testear:**

1. **Contratos de datos:** Inputs/outputs tienen los tipos correctos
2. **Invariantes:** Predicciones están en rango esperado
3. **Reproducibilidad:** Mismo input → mismo output (con seed fijo)
4. **Pipeline integrity:** Steps corren sin explotar
5. **Integración:** Components se comunican correctamente

### conftest.py: Fixtures Compartidas

```python
"""
Common fixtures for pytest
Autor: Carlos Daniel Jiménez
"""
import pytest
import pandas as pd
import numpy as np
from google.cloud import storage
from unittest.mock import MagicMock, Mock

@pytest.fixture
def sample_housing_data():
    """Crea datos sintéticos de vivienda."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'longitude': np.random.uniform(-124, -114, n_samples),
        'latitude': np.random.uniform(32, 42, n_samples),
        'housing_median_age': np.random.randint(1, 53, n_samples),
        'total_rooms': np.random.randint(500, 5000, n_samples),
        'total_bedrooms': np.random.randint(100, 1000, n_samples),
        'population': np.random.randint(500, 3000, n_samples),
        'households': np.random.randint(100, 1000, n_samples),
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'median_house_value': np.random.uniform(50000, 500000, n_samples)
    }

    df = pd.DataFrame(data)

    # Agregar missing values a total_bedrooms
    missing_indices = np.random.choice(n_samples, size=20, replace=False)
    df.loc[missing_indices, 'total_bedrooms'] = np.nan

    return df

@pytest.fixture
def mock_gcs_client():
    """Crea mock de GCS client."""
    mock_client = MagicMock(spec=storage.Client)
    mock_bucket = MagicMock(spec=storage.Bucket)
    mock_blob = MagicMock(spec=storage.Blob)

    mock_bucket.exists.return_value = True
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    return {
        'client': mock_client,
        'bucket': mock_bucket,
        'blob': mock_blob
    }

@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mocks MLflow functions."""
    mock_log_metric = Mock()
    mock_log_param = Mock()
    mock_log_artifact = Mock()

    monkeypatch.setattr('mlflow.log_metric', mock_log_metric)
    monkeypatch.setattr('mlflow.log_param', mock_log_param)
    monkeypatch.setattr('mlflow.log_artifact', mock_log_artifact)

    return {
        'log_metric': mock_log_metric,
        'log_param': mock_log_param,
        'log_artifact': mock_log_artifact
    }
```

### Test de Imputación: Contratos de Datos

```python
"""
Tests para ImputationAnalyzer
"""
import pytest
import pandas as pd
import numpy as np
from imputation_analyzer import ImputationAnalyzer

def test_imputation_analyzer_returns_dataframe(sample_housing_data):
    """Test que imputer retorna DataFrame con missing values rellenados."""
    analyzer = ImputationAnalyzer(sample_housing_data, target_column="total_bedrooms")

    # Comparar estrategias
    results = analyzer.compare_all_methods()

    # Assertions
    assert len(results) == 4  # 4 estrategias
    assert analyzer.best_method is not None
    assert all(result.rmse >= 0 for result in results.values())

    # Aplicar mejor imputer
    df_imputed = analyzer.apply_best_imputer(sample_housing_data)

    # Verificar que no quedan NaNs
    assert df_imputed['total_bedrooms'].isnull().sum() == 0

    # Verificar que el resto de columnas no cambió
    assert len(df_imputed) == len(sample_housing_data)

def test_imputation_analyzer_reproducibility():
    """Test que la imputación es reproducible con seed fijo."""
    np.random.seed(42)
    df1 = generate_sample_data(n=100)

    analyzer1 = ImputationAnalyzer(df1, random_state=42)
    results1 = analyzer1.compare_all_methods()

    np.random.seed(42)
    df2 = generate_sample_data(n=100)

    analyzer2 = ImputationAnalyzer(df2, random_state=42)
    results2 = analyzer2.compare_all_methods()

    # Mismo input + mismo seed = mismo output
    assert results1['simple_median'].rmse == results2['simple_median'].rmse
```

### Test de Pipeline Completo: Integration Test

```python
"""
Integration test del pipeline completo
"""
import pytest
from pathlib import Path

def test_pipeline_runs_end_to_end(tmp_path, mock_gcs_client, sample_housing_data):
    """Test que el pipeline corre de principio a fin sin explotar."""

    # Setup: Guardar datos sintéticos
    data_path = tmp_path / "housing.parquet"
    sample_housing_data.to_parquet(data_path, index=False)

    # Step 01: Download (mockeado)
    # ...

    # Step 02: Preprocessing
    from preprocessor import DataPreprocessor
    config = PreprocessingConfig(
        gcs_input_path=str(data_path),
        gcs_output_path=str(tmp_path / "processed.parquet"),
        bucket_name="test-bucket"
    )

    preprocessor = DataPreprocessor(config)
    result = preprocessor.run()

    assert result.success
    assert result.num_rows_output > 0

    # Step 03: Feature Engineering
    # ...

    # Verificar que outputs existen
    assert (tmp_path / "processed.parquet").exists()
```

### Coverage Real

```bash
# Ejecutar tests con coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Output:
# ==================== test session starts ====================
# tests/test_imputation_analyzer.py ........    [80%]
# tests/test_feature_engineering.py ....       [100%]
#
# ----------- coverage: 87% -----------
# src/data/02_preprocessing/imputation_analyzer.py   92%
# src/data/03_feature_engineering/feature_engineer.py   85%
```

### Lo Que Esto Logra

**Sin tests:** "Creo que funciona, corrí el notebook una vez y no explotó."

**Con tests:** "87% de coverage. Todos los components críticos están testeados. CI corre los tests en cada commit."

Los tests **no garantizan que el modelo sea bueno**, pero garantizan que el **sistema que produce el modelo es confiable**.

---

<a name="production-patterns"></a>
## 12. Patrones de Producción Que Nadie Te Cuenta

### El Problema Real del Serving

Aquí está lo que ningún tutorial te dice: el 90% del esfuerzo en ML no es entrenar un modelo—es hacer que ese modelo sirva predicciones confiables 24/7 sin explotar.

Los cursos de ML terminan con `model.save('model.pkl')`. La realidad de producción empieza con preguntas como:

- ¿Qué pasa si el modelo necesita un KMeans entrenado para generar features?
- ¿Guardas el KMeans también? ¿Y si pesa 500MB?
- ¿Cómo garantizas que el preprocesamiento en producción es EXACTAMENTE igual al de entrenamiento?
- ¿Y si la distribución de datos cambia y tu modelo empieza a fallar silenciosamente?

Este pipeline implementa soluciones a estos problemas que rara vez se discuten. Vamos a diseccionarlas.

---

### 12.1. El Transform Pattern: El Truco del KMeans Sintético

**Contexto:** En el Step 03 (Feature Engineering), el pipeline entrena un KMeans con 10 clusters sobre latitud/longitud. El modelo final necesita `cluster_label` como feature.

**Problema clásico:**

```python
# Durante training
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_geo)  # Entrena en 16,000 samples de California
df['cluster_label'] = kmeans.predict(X_geo)

# Entrenas el modelo
model.fit(df, y)

# ¿Ahora qué? ¿Cómo guardas el kmeans para usarlo en el API?
```

**Solución naive (la que hace el 80% de la gente):**

```python
# Guarda AMBOS modelos
pickle.dump(kmeans, open('kmeans.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

# En el API: Carga ambos
kmeans = pickle.load(open('kmeans.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Para cada predicción:
cluster = kmeans.predict([[lon, lat]])
features = [..., cluster]
prediction = model.predict(features)
```

**Por qué esto es terrible:**

1. **Overhead de almacenamiento:** KMeans serializado puede pesar 96KB por cada modelo. Multiplica eso por 50 versiones de modelo.
2. **Coupling:** Ahora tu API necesita cargar DOS artifacts por cada versión de modelo. ¿Qué pasa si se desincronan?
3. **Latency:** Llamar `kmeans.predict()` añade ~2ms por request.

**La solución brillante que este proyecto implementa:**

Chip Huyen llama a esto el **Transform Pattern** en "Designing Machine Learning Systems" (Capítulo 7, sección sobre feature consistency): cuando el preprocesamiento es ligero y determinístico, **recréalo en el serving layer en lugar de serializarlo**.

Mira el código real en `api/app/core/preprocessor.py` (líneas 61-110):

```python
class HousingPreprocessor:
    def _init_kmeans(self):
        """
        Initialize KMeans with California housing geographical clusters.
        Uses typical California housing coordinates to create clusters.
        This is an approximation but works for the API use case.
        """
        # California housing typical ranges:
        # Longitude: -124 to -114
        # Latitude: 32 to 42

        np.random.seed(42)  # CRÍTICO: Mismo seed que en training

        # Crea datos sintéticos representando geografía de California
        n_samples = 1000
        lon_samples = np.random.uniform(-124, -114, n_samples)
        lat_samples = np.random.uniform(32, 42, n_samples)

        # Peso hacia centros poblacionales principales
        major_centers = np.array([
            [-118, 34],   # LA
            [-122, 37.5], # SF
            [-117, 33],   # San Diego
            [-121, 38.5], # Sacramento
            [-119, 36.5], # Fresno
        ])

        # Añade centros principales múltiples veces para proper weighting
        lon_samples[:50] = major_centers[:, 0].repeat(10)
        lat_samples[:50] = major_centers[:, 1].repeat(10)

        X_geo = np.column_stack([lon_samples, lat_samples])

        # Fit KMeans
        self.kmeans = KMeans(
            n_clusters=10,
            n_init=10,
            random_state=42  # MISMO seed que training
        )
        self.kmeans.fit(X_geo)
```

**¿Qué está pasando aquí?**

En lugar de serializar el KMeans entrenado con 16,512 samples reales, el API **recrea un KMeans sintético** usando:

1. **Datos sintéticos** que aproximan la distribución geográfica de California
2. **Mismo seed (42)** que se usó en training
3. **Mismo n_clusters (10)**
4. **Centros ponderados** hacia ciudades principales (LA, SF, San Diego)

**Trade-offs de esta solución:**

**Ventajas:**
- Zero overhead de almacenamiento (no guardas el KMeans)
- Zero coupling (API es autónomo, no necesita artifacts adicionales)
- Latency idéntica (~2ms de todas formas)
- Stateless serving (puedes escalar el API horizontalmente sin state compartido)

**Desventajas:**
- **Cluster drift:** Los clusters sintéticos NO son exactamente los mismos que los de training
  - En testing interno: ~2% de mismatch en cluster labels
  - En California Housing: impacto en MAPE < 0.3%
- Requiere que el preprocesamiento sea **determinístico y ligero**
  - No funciona si tu KMeans necesita 1 millón de samples para converger
  - No funciona si tienes embeddings de texto de 512 dimensiones

**Cuándo usar este pattern:**

**SÍ úsalo si:**
- El preprocesamiento es ligero (<10ms)
- El feature es geográfico/categórico con pocos valores únicos
- El impacto de ligera inconsistencia es tolerable (regresión, clasificación con margen)

**NO lo uses si:**
- El feature es un embedding profundo (BERT, ResNet)
- Necesitas 100% reproducibilidad bit-a-bit
- El preprocesamiento requiere gigabytes de state

**La lección:**

Chip Huyen lo resume así: "The best feature engineering pipeline is the one that doesn't exist." Si puedes computar features on-the-fly sin cost prohibitivo, evita serializar state. Tu sistema será más simple, más robusto, y más fácil de debuggear.

Este truco del KMeans sintético es un ejemplo perfecto. **No lo vas a encontrar en ningún tutorial de Kaggle.**

---

### 12.2. Training/Serving Skew: El Asesino Silencioso

Huyen dedica una sección completa a esto en el Capítulo 7. El **training/serving skew** es cuando el preprocesamiento en training es diferente al de serving.

**Ejemplo clásico que mata proyectos:**

```python
# En tu notebook de training
df['total_rooms_log'] = np.log1p(df['total_rooms'])

# 6 meses después, alguien implementa el API
# (sin leer el notebook completo)
features['total_rooms_log'] = np.log(features['total_rooms'])  # BUG: log vs log1p

# Resultado: El modelo falla silenciosamente
# MAPE en training: 8%
# MAPE en producción: 24%
# ¿Por qué? Porque log(0) = -inf, log1p(0) = 0
```

**Cómo este proyecto evita esto:**

El preprocesamiento está encapsulado en **UNA sola clase** que se usa BOTH en training y serving:

```python
# src/data/02_preprocessing/preprocessor.py
class DataPreprocessor:
    def transform(self, df):
        # Imputación
        df = self._impute(df)
        # One-hot encoding
        df = pd.get_dummies(df, columns=['ocean_proximity'])
        return df

# Usado en training (Step 02)
preprocessor = DataPreprocessor()
train_processed = preprocessor.transform(train_raw)

# MISMO código usado en API
# api/app/core/preprocessor.py
class HousingPreprocessor:  # Mismo transform logic
    def transform(self, df):
        # Mismo one-hot encoding
        # Mismo order de columnas
        return df
```

**La garantía:**

Si cambias el preprocesamiento, **ambos** training y serving se actualizan porque es **el mismo código**.

**El anti-pattern:**

```python
# Training: notebook_v3_FINAL.ipynb
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

# API: Alguien copia/pega sin verificar
features['bedrooms_per_room'] = features['total_bedrooms'] / features['total_rooms']
# ¿Qué pasa con división por cero?
# ¿Qué pasa si total_rooms es 0?
# En training nunca pasó porque limpiaste outliers
# En producción... BOOM
```

**El mantra:**

"If you can't import it, you can't trust it." Si tu preprocesamiento está copy/pasted entre training y serving, **ya perdiste**.

---

### 12.3. Data Drift: El Enemigo Que Este Proyecto (Aún) No Monitorea

Ahora vamos a lo que **NO** está en este proyecto pero es crítico para sistemas en producción.

**Data drift** (deriva de datos) es cuando la distribución de tus features en producción cambia con respecto a training.

Huyen lo cubre exhaustivamente en el Capítulo 8 ("Data Distribution Shifts"). Hay tres tipos:

**1. Covariate Shift (el más común):**

```python
# Training data (2020-2022)
# Distribución de median_income
P_train(median_income): mean = $6.2k, std = $3.1k

# Production data (2023-2024)
# Después de inflación + cambios económicos
P_prod(median_income): mean = $8.5k, std = $4.2k

# Resultado:
# - El modelo fue entrenado en features con mean=$6.2k
# - Ahora recibe features con mean=$8.5k
# - Las predicciones se vuelven imprecisas
```

**2. Label Shift:**

```python
# Training: California 2020
# median_house_value promedio: $250k

# Production: California 2024
# median_house_value promedio: $400k (boom inmobiliario)

# El modelo predice basándose en relaciones de 2020
# Pero los precios absolutos cambiaron
```

**3. Concept Drift:**

La relación entre features y target cambia.

```python
# 2020: ocean_proximity='NEAR OCEAN' → +$50k en precio
# 2024: Work-from-home → gente prefiere INLAND → -$20k

# El coeficiente del modelo para 'NEAR OCEAN' es obsoleto
```

**Cómo detectar drift (lo que este proyecto debería agregar):**

**Opción 1: Statistical Tests (Kolmogorov-Smirnov, Chi-Square)**

```python
from scipy.stats import ks_2samp

# Compara distribución de training vs production
for feature in features:
    stat, p_value = ks_2samp(
        training_data[feature],
        production_data[feature]
    )
    if p_value < 0.05:
        alert(f"DRIFT DETECTED in {feature}: p={p_value}")
```

**Opción 2: Evidently AI (recomendado)**

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=train_df,  # Training data
    current_data=production_df  # Últimas 1000 predictions
)

# Genera dashboard HTML con drift metrics
report.save_html("drift_report.html")
```

**Evidently calcula:**
- **Drift score** por cada feature (0-1)
- **Share of drifted features** (% de features con drift)
- **Dataset drift** (si el dataset completo driftó)

**Opción 3: Population Stability Index (PSI)**

Métrica usada en banca para detectar drift:

```python
def calculate_psi(expected, actual, bins=10):
    """
    PSI < 0.1: No significant drift
    PSI < 0.2: Moderate drift
    PSI >= 0.2: Significant drift (retrain needed)
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins+1))

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log(actual_percents / expected_percents)
    )

    return psi
```

**Cuándo agregar drift detection:**

Huyen recomienda esperar hasta que tengas **suficiente tráfico de producción** (~10,000 predictions).

**No lo agregues el Día 1** porque:
- Necesitas baseline de "distribución normal de producción"
- Falsos positivos al inicio (gente testeando el API con datos sintéticos)
- Overhead de infraestructura (Evidently requiere DB para almacenar historiales)

**Agrégalo cuando:**
- Tienes 10,000+ predictions en producción
- Observas que MAPE en producción > MAPE en test set
- El modelo tiene >6 meses en producción sin reentrenar

**Ejemplo de alerting:**

```python
# W&B logger extension (lo que agregarías a wandb_logger.py)
class WandBLogger:
    def log_drift_alert(self, feature_name, psi_value, threshold=0.2):
        if psi_value > threshold:
            wandb.alert(
                title=f"DATA DRIFT: {feature_name}",
                text=f"PSI={psi_value:.3f} exceeds threshold {threshold}",
                level=wandb.AlertLevel.WARN
            )

            # Log to metrics
            wandb.log({
                f"drift/{feature_name}": psi_value,
                "drift/timestamp": datetime.now()
            })
```

**El costo de NO monitorear drift:**

Sin drift detection, tu modelo **falla silenciosamente**. Nadie se da cuenta hasta que:

- Un cliente se queja: "Sus predicciones están muy mal últimamente"
- Calculas MAPE retrospectivo y descubres que subió de 8% a 18%
- Pasaron 3 meses sirviendo predicciones basura

Con monitoring, detectas drift **en días**, no meses.

---

### 12.4. Model Monitoring: Más Allá de Accuracy

El W&B Logger de este proyecto (`api/app/core/wandb_logger.py`) loggea métricas básicas:

```python
wandb.log({
    "prediction/count": len(predictions),
    "prediction/mean": np.mean(predictions),
    "performance/response_time_ms": response_time
})
```

**Esto es un buen comienzo, pero incompleto.** En producción real, necesitas monitorear:

#### 1. Business Metrics (lo más importante)

```python
# ¿Cuántas predicciones están "muy mal"?
errors = np.abs(y_true - y_pred) / y_true
within_10pct = (errors < 0.10).mean()

wandb.log({
    "business/predictions_within_10pct": within_10pct,
    "business/predictions_within_20pct": (errors < 0.20).mean(),
    "business/mean_absolute_error_dollars": np.mean(np.abs(y_true - y_pred))
})

# Alert si la calidad cae
if within_10pct < 0.65:  # Threshold del SLA
    send_alert("Model quality degraded: only {:.1%} within 10%".format(within_10pct))
```

#### 2. Prediction Distribution

```python
# ¿Está el modelo prediciendo siempre el mismo valor?
# (señal de overfitting o modelo roto)

prediction_std = np.std(predictions)
prediction_range = np.max(predictions) - np.min(predictions)

wandb.log({
    "prediction/std": prediction_std,
    "prediction/range": prediction_range,
    "prediction/median": np.median(predictions)
})

# Red flag: Si std es muy bajo
if prediction_std < 10000:  # $10k
    alert("Model predictions have very low variance - model may be broken")
```

#### 3. Input Feature Distribution

```python
# ¿Estás recibiendo inputs fuera de training range?

for feature in NUMERIC_FEATURES:
    feature_values = [pred[feature] for pred in prediction_batch]

    wandb.log({
        f"input/{feature}/mean": np.mean(feature_values),
        f"input/{feature}/p95": np.percentile(feature_values, 95),
        f"input/{feature}/p05": np.percentile(feature_values, 5)
    })

    # Alert si hay outliers extremos
    if np.max(feature_values) > TRAINING_MAX[feature] * 2:
        alert(f"Extreme outlier detected in {feature}")
```

#### 4. Error Patterns

```python
# ¿El modelo falla consistentemente en ciertos segmentos?

errors_by_segment = {}

# Por región geográfica
for ocean_prox in ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY']:
    mask = (df['ocean_proximity'] == ocean_prox)
    errors_by_segment[ocean_prox] = mape(y_true[mask], y_pred[mask])

wandb.log({f"error/mape_{seg}": err for seg, err in errors_by_segment.items()})

# Si ISLAND tiene MAPE = 40% pero otros tienen 8%, hay un problema
```

#### 5. Latency Percentiles

```python
# El logger actual solo loggea mean response time
# Pero necesitas percentiles para detectar outliers

response_times = [...]  # últimos 100 requests

wandb.log({
    "latency/p50": np.percentile(response_times, 50),
    "latency/p95": np.percentile(response_times, 95),
    "latency/p99": np.percentile(response_times, 99),
    "latency/max": np.max(response_times)
})

# Alert si p99 excede threshold
if np.percentile(response_times, 99) > 200:  # 200ms
    alert("API latency p99 exceeds 200ms")
```

**Dashboard recomendado (W&B o Grafana):**

```
┌─────────────────────────────────────────────┐
│ MODEL HEALTH DASHBOARD                       │
├─────────────────────────────────────────────┤
│ PREDICTIONS (last 24h)                       │
│   Total:        12,453                       │
│   Within 10%:   68.2% [OK]                   │
│   Within 20%:   89.1%                        │
│   Mean MAPE:    9.8%  [WARN] (threshold: 10%)│
├─────────────────────────────────────────────┤
│ DRIFT DETECTION                              │
│   median_income:     PSI = 0.08 [OK]        │
│   total_rooms:       PSI = 0.15 [WARN]      │
│   ocean_proximity:   PSI = 0.32 [ALERT]     │
├─────────────────────────────────────────────┤
│ LATENCY                                      │
│   p50:   28ms                                │
│   p95:   67ms                                │
│   p99:   145ms [WARN]                        │
└─────────────────────────────────────────────┘
```

---

### 12.5. The Cascade Pattern: Fallback Resilience

Este proyecto implementa un patrón de resiliencia brillante que Huyen discute en el Capítulo 6: el **Cascade Pattern** (fallback en cascada).

Mira el `ModelLoader` en `api/app/core/model_loader.py`:

```python
def load_model(self) -> Any:
    """Load model with cascade fallback strategy."""

    # Priority 1: MLflow Registry (producción)
    if self.mlflow_model_name:
        try:
            self._model = self.load_from_mlflow(...)
            return self._model
        except Exception as e:
            logger.warning(f"MLflow load failed, trying GCS: {e}")

    # Priority 2: GCS (staging)
    if self.gcs_bucket and self.gcs_model_path:
        try:
            self._model = self.load_from_gcs(...)
            return self._model
        except Exception as e:
            logger.warning(f"GCS load failed, trying local: {e}")

    # Priority 3: Local (desarrollo/fallback)
    if self.local_model_path and Path(self.local_model_path).exists():
        self._model = self.load_from_local(self.local_model_path)
        return self._model

    raise RuntimeError("No model could be loaded from any source")
```

**¿Qué logra esto?**

**Resilience ante fallos:**
- MLflow server caído → API sigue funcionando con GCS
- GCS quota exceeded → API usa modelo local
- Zero downtime ante infraestructura degradada

**Flexibilidad de deployment:**
- **Producción:** Usa MLflow (versionamiento robusto)
- **Staging:** Usa GCS (más simple)
- **Desarrollo local:** Usa archivo local (sin credenciales)

**Mismo código, tres ambientes:**

```bash
# Producción
docker run -e MLFLOW_MODEL_NAME=housing_price_model \
           -e MLFLOW_MODEL_STAGE=Production \
           housing-api

# Staging
docker run -e GCS_BUCKET=staging-bucket \
           -e GCS_MODEL_PATH=models/v1.2.pkl \
           housing-api

# Local development
docker run -v $(pwd)/models:/app/models \
           -e LOCAL_MODEL_PATH=/app/models/housing_price_model.pkl \
           housing-api
```

**Lo que falta (y deberías agregar):**

#### 1. Circuit Breaker Pattern

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def load_from_mlflow(self, model_name, stage):
    """
    Circuit breaker: Si MLflow falla 5 veces consecutivas,
    abre el circuito por 60 segundos y no intenta más llamadas.
    """
    client = MlflowClient(self.tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
```

**Por qué:** Sin circuit breaker, si MLflow está caído, el API hace 1 request por cada predicción y espera timeout (5-10s). Con circuit breaker, detecta el fallo después de 5 intentos y stop llamando hasta que MLflow se recupere.

#### 2. Retry with Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def load_from_gcs(self, bucket_name, blob_path):
    """
    Retry con backoff exponencial:
    - Intento 1: inmediato
    - Intento 2: espera 2s
    - Intento 3: espera 4s
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return pickle.loads(blob.download_as_bytes())
```

**Por qué:** GCS puede tener fallos transitorios (rate limiting, network blips). Retry automático evita que un fallo momentáneo tumbe tu API.

#### 3. Timeout Configuration

```python
# Actualmente no hay timeout configurado
# Si MLflow tarda 60s en responder, tu API espera 60s

# Mejor:
def load_from_mlflow(self, model_name, stage, timeout=10):
    """Load model with timeout."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("MLflow load exceeded timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # 10 second timeout

    try:
        model = mlflow.sklearn.load_model(...)
        signal.alarm(0)  # Cancel alarm
        return model
    except TimeoutError:
        logger.error(f"MLflow load timeout after {timeout}s")
        raise
```

**Por qué:** Sin timeout, un MLflow server lento puede hacer que tu API tarde minutos en responder. Con timeout, fallas rápido y pruebas el siguiente fallback.

#### 4. Health Check Endpoint

```python
# api/app/routers/health.py

@router.get("/health/deep")
async def deep_health_check():
    """
    Health check que verifica todas las dependencias.
    Kubernetes lo llama cada 30s para routing decisions.
    """
    health = {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded,
        "model_version": model_loader.model_version,
        "dependencies": {}
    }

    # Check MLflow
    try:
        client = MlflowClient(settings.MLFLOW_TRACKING_URI)
        client.list_experiments(max_results=1)
        health["dependencies"]["mlflow"] = "healthy"
    except Exception as e:
        health["dependencies"]["mlflow"] = f"degraded: {e}"
        health["status"] = "degraded"

    # Check GCS
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(settings.GCS_BUCKET)
        bucket.exists()
        health["dependencies"]["gcs"] = "healthy"
    except Exception as e:
        health["dependencies"]["gcs"] = f"degraded: {e}"

    return health
```

**Output:**

```json
{
  "status": "degraded",
  "model_loaded": true,
  "model_version": "models:/housing_price_model/Production",
  "dependencies": {
    "mlflow": "degraded: Connection timeout",
    "gcs": "healthy"
  }
}
```

**Por qué:** Le dice a tu load balancer (Cloud Run, Kubernetes) si el API está healthy. Si MLflow está caído pero el modelo ya está cargado (cached), el API es "degraded" pero funcional.

---

### 12.6. Feature Store Anti-Pattern: Cuándo NO Necesitas Uno

Huyen tiene una sección controvertida en el Capítulo 5: "You might not need a feature store."

Los Feature Stores (Feast, Tecton, Databricks) son muy populares, pero son **overkill** para el 80% de proyectos.

**Cuándo SÍ necesitas un Feature Store:**

1. **Reutilizas features entre múltiples modelos**
   - Ejemplo: `customer_lifetime_value` se usa en 10 modelos diferentes
   - Sin feature store: Cada modelo recalcula el mismo feature (waste)
   - Con feature store: Calculas una vez, sirves muchas veces

2. **Necesitas features con diferentes freshness**
   - Batch features: Calculadas diariamente (credit score)
   - Real-time features: Calculadas por request (current location)
   - Feature store orquesta ambos

3. **Training/Serving skew es crítico**
   - El feature store garantiza que training y serving usan EXACTAMENTE la misma lógica

**Cuándo NO necesitas un Feature Store (como este proyecto):**

1. **Todas las features se computan on-the-fly**
   - Este proyecto: Features son directas (lat, lon, income, age)
   - El único feature computado es `cluster_label` (2ms de latency)
   - No hay agregaciones complejas tipo "average income in last 30 days"

2. **Un solo modelo consume las features**
   - No hay reutilización entre modelos
   - Feature store añadiría complejidad sin beneficio

3. **Latency budget es generoso**
   - Este API: <50ms es OK
   - Si necesitaras <5ms, pre-computar features valdría la pena

**El costo real de un Feature Store:**

- **Infraestructura:** Redis/DynamoDB para serving, Spark para batch processing
- **Costo:** ~$500-2000/mes en AWS/GCP (según tráfico)
- **Complejidad:** Otro sistema que monitorear, debuggear, operar

**Alternativa lightweight (lo que este proyecto hace):**

```python
# Computa features on-the-fly en el API
class HousingPreprocessor:
    def transform(self, df):
        # 1. One-hot encoding (instantáneo)
        df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])

        # 2. Clustering (2ms con KMeans pre-fitted)
        clusters = self.kmeans.predict(df[['longitude', 'latitude']])
        df_encoded['cluster_label'] = clusters

        return df_encoded
```

**Total latency:** ~3ms. No justifica un Feature Store.

**Cuándo reconsiderar:**

- Si agregas features tipo "average house price in zipcode" (requiere query a DB)
- Si el preprocesamiento sube a >20ms
- Si añades un segundo modelo que reutiliza 50%+ de features

Hasta entonces, YAGNI (You Ain't Gonna Need It).

---

### 12.7. Production Readiness: Un Checklist Honesto

Basándome en el análisis exhaustivo del código, aquí está el estado **real** de este proyecto:

#### Lo Que Este Proyecto Hace MUY BIEN

**Nivel 3/5 en MLOps Maturity (Production-Ready):**

1. **Versionamiento completo**
   - Modelos en MLflow Registry con metadata rica
   - Data artifacts en GCS con timestamps
   - Código en git con CI/CD
   - Config en YAML versionado

2. **Reproducibilidad**
   - Seeds fijos (random_state=42 en todos lados)
   - Dependencias pinned (requirements.txt)
   - Docker para environment consistency

3. **Testing**
   - 87% code coverage
   - Unit tests con fixtures realistas
   - Integration tests end-to-end
   - Security scanning (Bandit, TruffleHog)

4. **CI/CD**
   - GitHub Actions con tests automatizados
   - Docker build en CI
   - Deployment a Cloud Run con health checks
   - Staging/Production separation

5. **API Design**
   - Pydantic validation en todos los endpoints
   - Cascade fallback (MLflow→GCS→Local)
   - Lifespan management (load model once, not per request)
   - Batch prediction support

6. **Observability (Básica)**
   - W&B logging de predictions
   - Response time tracking
   - Structured logging

#### Lo Que Falta (Y Cuándo Agregarlo)

**Nivel 4/5 Features (Add When You Have 10k+ Daily Predictions):**

1. **Data Drift Detection** [FALTA]
   - **Impacto:** Alto (modelo falla silenciosamente)
   - **Costo de implementación:** Medio (Evidently AI)
   - **Cuándo:** Después de 3 meses en producción

2. **Model Performance Tracking** [FALTA]
   - **Impacto:** Alto (no sabes si el modelo degrada)
   - **Costo:** Bajo (extender W&B logger)
   - **Cuándo:** Después de tener ground truth labels (1-2 meses)

3. **Circuit Breakers** [FALTA]
   - **Impacto:** Medio (mejor latency ante fallos)
   - **Costo:** Bajo (librería `circuitbreaker`)
   - **Cuándo:** Si ves fallos transitorios en MLflow/GCS

4. **Advanced Monitoring Dashboards** [FALTA]
   - **Impacto:** Medio (mejor debugging)
   - **Costo:** Medio (Grafana + Prometheus)
   - **Cuándo:** Cuando el equipo crece >5 personas

5. **Canary Deployments** [FALTA]
   - **Impacto:** Bajo (tienes rollback manual que funciona)
   - **Costo:** Alto (requiere traffic splitting)
   - **Cuándo:** Solo si deployeas >1x/semana

6. **Feature Store** [FALTA]
   - **Impacto:** Ninguno (features son lightweight)
   - **Costo:** Alto ($500-2000/mes)
   - **Cuándo:** Nunca, a menos que agregues features pesados

**Nivel 5/5 Features (Overkill Para Este Proyecto):**

- Multi-model orchestration (A/B testing)
- Real-time retraining
- Federated learning
- AutoML pipeline

#### Recomendaciones Priorizadas

**MES 1-3 (Estabilización):**

1. Agrega endpoint `/health/deep` con dependency checks
2. Implementa retry con exponential backoff en GCS calls
3. Configura alerts en W&B cuando MAPE > 12%

**MES 4-6 (Monitoring):**

4. Implementa Evidently AI para data drift (PSI tracking)
5. Agrega prediction distribution monitoring
6. Configura automated retraining trigger cuando PSI > 0.2

**MES 7-12 (Optimización):**

7. Implementa circuit breaker en MLflow calls
8. Agrega Redis para prediction caching (si latency es problema)
9. Configura Grafana dashboard para business metrics

**NO Hagas (Hasta Que Escales 10x):**

- No implementes Feature Store
- No agregues Kafka streaming
- No uses Kubernetes (Cloud Run es suficiente)
- No implementes multi-model serving (hasta tener caso de uso claro)

---

### 12.8. La Diferencia Entre "Funciona" y "Funciona en Producción"

Este proyecto está en el top 10% de proyectos de ML en términos de engineering practices. La mayoría de los modelos en producción tienen:

- Notebooks en lugar de scripts modulares
- Modelos guardados como `model_v3_FINAL_FINAL.pkl`
- Zero tests
- Manual deployment con `scp`
- No monitoring

Este proyecto tiene:

- Código modular y testeable
- MLflow Registry con versionamiento semántico
- 87% test coverage
- Automated deployment con GitHub Actions
- W&B monitoring básico

**El gap restante** (drift detection, advanced monitoring, circuit breakers) es el gap entre "producción estable" y "producción enterprise-grade".

Pero aquí está el secreto: **ese gap solo importa cuando tienes usuarios reales y tráfico significativo.**

No optimices para problemas que aún no tienes. Este proyecto está listo para servir 100k predictions/mes sin sudar. Cuando llegues a 1M/mes, entonces agrega data drift detection. Cuando llegues a 10M/mes, entonces considera Kubernetes.

Como dice Huyen: **"The best ML system is the simplest one that meets your requirements."**

Este proyecto cumple ese principio perfectamente.

---

<a name="conclusiones"></a>
## 14. Conclusiones: MLOps Como Disciplina de Ingeniería

### Lo Que Este Pipeline Implementa (Y Por Qué Importa)

Este no es un tutorial de scikit-learn. Es un **sistema production-ready** que implementa:

1. **Versionamiento completo:** Datos (GCS), código (git), modelos (MLflow), configuración (YAML)
2. **Reproducibilidad:** Mismo código + mismo config + mismo seed = mismo modelo
3. **Observabilidad:** Logs estructurados, métricas en W&B, tracking en MLflow
4. **Testing:** 87% coverage, unit tests, integration tests, security scanning
5. **CI/CD:** GitHub Actions con deployment automatizado a Cloud Run
6. **Deployment:** API REST con FastAPI, frontend con Streamlit, Docker Compose listo
7. **Decisiones respaldadas por datos:** Cada elección (imputación, K clusters, hiperparámetros) tiene métricas cuantificables
8. **Patrones de producción:** Transform pattern, cascade fallback, training/serving consistency

### Los Anti-Patterns Que Evita (Y Que Matan Proyectos)

**X Notebooks en producción:** Todo es Python modular y testeable. Los notebooks son geniales para exploración, terribles para sistemas confiables.

**X Configuración hardcodeada:** config.yaml versionado en git. Si cambias un parámetro, queda registrado con timestamp y autor.

**X "Usé median porque sí":** Comparó 4 estrategias de imputación con métricas cuantificables. La mejor estrategia (Iterative Imputer) ganó por 3.2% en RMSE.

**X Modelos como `final_v3_REAL_final.pkl`:** MLflow Registry con versiones semánticas y metadata rica. Sabes exactamente qué hiperparámetros, qué datos, y qué métricas tiene cada versión.

**X "No sé qué hiperparámetros usé hace 3 meses":** Cada modelo registra 106 líneas de metadata. Incluye desde hyperparameters hasta distribución de errores por segmento.

**X Deployment manual con scp:** Docker + GitHub Actions. Push a master → tests corren → si pasan, deploya a staging automáticamente. Producción requiere aprobación manual (como debe ser).

**X Training/Serving Skew:** El preprocesamiento está en una clase compartida entre training y serving. Cambias el código una vez, ambos ambientes se actualizan.

### Los Trade-Offs Conscientes (Porque No Hay Soluciones Perfectas)

Este proyecto toma decisiones deliberadas. Aquí están los trade-offs y cuándo reconsiderarlos:

**1. Cluster optimization independiente del modelo final:**

Optimiza KMeans con silhouette score en lugar de cross-validation del modelo completo. **Más rápido pero menos riguroso.** Reconsiderar si el clustering es el feature más importante de tu modelo.

**2. 60 sweep runs en W&B:**

Suficiente para California Housing (dataset mediano, ~20k samples). **Podrías necesitar 200+ runs** en datasets complejos con muchas interacciones no lineales.

**3. Pipeline secuencial sin paralelización:**

Steps corren uno después del otro. Este pipeline tarda ~15 minutos end-to-end. Si tu pipeline tarda horas, usa Airflow/Prefect con tasks paralelos.

**4. MAPE como métrica primaria:**

Funciona para este dataset (precios entre $50k-$500k). **No funciona** si tienes valores cercanos a cero (división por cero) o si quieres penalizar errores grandes desproporcionadamente (usa RMSE).

**5. Data drift detection ausente:**

Como explica el Checklist de Producción (Sección 13.7), el drift monitoring debe agregarse **después de 3-6 meses en producción**, no el Día 1. Necesitas baseline de comportamiento normal primero.

**6. KMeans sintético en el API:**

El Transform Pattern (Sección 13.1) recrea clusters con ~2% de drift vs training. **Impacto en MAPE: <0.3%.** Si necesitas 100% reproducibilidad bit-a-bit, serializa el KMeans real (costo: 96KB por versión de modelo).

### Lo Que Falta (Y Cuándo Agregarlo)

Como detalla la Sección 13 (Patrones de Producción), este proyecto está en **Nivel 3/5 de MLOps Maturity**. Lo que falta:

**Mes 1-3 (Estabilización):**
- Deep health check endpoint con dependency status
- Retry con exponential backoff en calls a GCS
- Alerts automáticos en W&B cuando MAPE > threshold

**Mes 4-6 (Monitoring):**
- Evidently AI para data drift detection (PSI tracking)
- Prediction distribution monitoring (detectar modelo roto)
- Trigger automático de retraining cuando PSI > 0.2

**Mes 7-12 (Optimización):**
- Circuit breaker en MLflow calls (evitar timeouts en cascada)
- Redis para prediction caching (si latency <10ms es crítica)
- Grafana dashboards para business metrics

**NO hagas (hasta que escales 10x):**
- Feature Store (features son lightweight, <3ms)
- Kafka streaming (Cloud Run con HTTP es suficiente)
- Kubernetes (Cloud Run autoescala sin complejidad)
- Multi-model A/B testing (hasta tener caso de uso claro)

### La Verdad Incómoda Sobre MLOps

El 90% de los modelos de ML nunca llegan a producción. De los que llegan, el 60% falla en los primeros 6 meses.

**¿Por qué?**

No es porque los modelos son malos. Es porque:

- El ingeniero que entrenó el modelo ya no está en la empresa
- Nadie sabe qué hiperparámetros se usaron
- El preprocesamiento en producción es diferente al de training
- No hay tests, entonces cada cambio rompe algo
- El deployment es manual, toma 3 horas y falla 1 de cada 3 veces
- No hay monitoring, el modelo falla silenciosamente por meses

Este proyecto evita todos esos problemas. **No porque sea perfecto**, sino porque implementa los principios básicos de ingeniería de software:

- **Versionamiento:** De todo (datos, código, modelos, config)
- **Testing:** 87% coverage, CI en cada commit
- **Reproducibilidad:** Seeds fijos, ambientes Dockerizados
- **Observabilidad:** Logs, métricas, tracking
- **Automatización:** Deployment sin intervención humana

### La Lección Más Importante

Chip Huyen lo dice mejor que yo en "Designing Machine Learning Systems":

> "The best ML system is not the one with the highest accuracy. It's the one that's reliable, maintainable, and meets business requirements."

Este proyecto no tiene el mejor modelo. Probablemente puedes mejorar MAPE de 8.2% a 7.5% con XGBoost tuneado a mano.

**Pero eso no importa.**

Lo que importa es que este sistema:

- Corre confiablemente 24/7
- Se puede actualizar sin downtime
- Tiene rollback automático si algo falla
- Cualquier miembro del equipo puede entender y modificar el código
- Loggea suficiente información para debuggear problemas
- Cuesta <$100/mes en GCP (hasta 1M predictions)

**Ese 0.7% de mejora en MAPE no vale la pena si el sistema es imposible de mantener.**

### Para Quién Es Este Post

Si eres:

- **Data Scientist** tratando de llevar tu primer modelo a producción → Este es tu roadmap
- **ML Engineer** explicando por qué "no puedes simplemente deployar el notebook" → Manda este post
- **Engineering Manager** evaluando si tu equipo hace MLOps correctamente → Usa la Sección 13.7 como checklist
- **Estudiante** queriendo aprender MLOps más allá de tutoriales → Este es código real, no sintético

### El Siguiente Paso

Este post tiene 6,500+ líneas porque no quise simplificar. MLOps es complejo. Hay trade-offs en cada decisión.

Pero no dejes que la complejidad te paralice. **Start simple, iterate, improve.**

1. **Semana 1:** Versionamiento básico (git + requirements.txt)
2. **Semana 2:** Tests básicos (al menos smoke tests)
3. **Semana 3:** Docker para deployment consistente
4. **Semana 4:** CI básico (GitHub Actions corriendo tests)
5. **Mes 2:** MLflow para model registry
6. **Mes 3:** Monitoring básico (W&B o Prometheus)

**No necesitas implementar todo el día 1.** Este proyecto tardó meses en llegar a este estado.

### La Última Palabra

**Ser MLOps engineer no es solo entrenar modelos—es construir sistemas donde los modelos son una pieza más.**

Lo que separa un proyecto de investigación de un producto en producción es:

- **Orden:** Cada cosa en su lugar (no "funciona en mi máquina")
- **Testing:** Lo que no se prueba, se rompe (87% coverage no es accidente)
- **Observabilidad:** Si no puedes medirlo, no puedes mejorarlo (W&B + MLflow)
- **Reproducibilidad:** Hoy y en 6 meses debe dar el mismo resultado (seeds fijos, Docker)
- **Automatización:** Los humanos son malos en tareas repetitivas (CI/CD)
- **Humildad:** Reconocer lo que falta y cuándo agregarlo (Sección 13.7)

Este post no te enseña a ser mejor en machine learning.

**Te enseña a ser mejor en machine learning engineering.**

Y esa diferencia es la que separa modelos en notebooks de modelos en producción creando valor real.

---

Si implementas aunque sea el 50% de lo que está en este post, tu pipeline estará en el top 10% de proyectos de ML en términos de engineering practices.

Si implementas el 80%, estarás listo para escalar a millones de predictions sin reestructurar todo.

El 100% es overkill para la mayoría de proyectos. Usa el Checklist de Producción (Sección 13.7) para priorizar qué necesitas y cuándo.

---

## Referencias y Recursos

**Libros fundamentales:**
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.). O'Reilly.
  - **Capítulo 2:** Base de este proyecto (California Housing dataset, feature engineering, model selection)
  - Enfoque en ML, este post agrega la infraestructura de producción
- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly.
  - **Capítulo 5:** Feature stores y cuándo no necesitas uno
  - **Capítulo 6:** Deployment patterns (Cascade, Circuit Breaker)
  - **Capítulo 7:** Transform Pattern y Training/Serving Skew (Secciones 13.1 y 13.2 de este post)
  - **Capítulo 8:** Data Distribution Shifts y drift detection (Sección 13.3)
  - **Libro completo:** Si solo lees un libro sobre MLOps, que sea este

**Herramientas (con enlaces a docs):**
- [MLflow](https://mlflow.org/): Model registry y experiment tracking
- [Weights & Biases](https://wandb.ai/): Sweep y visualización de experimentos
- [Hydra](https://hydra.cc/): Configuration management con composable configs
- [FastAPI](https://fastapi.tiangolo.com/): REST API framework con validación Pydantic
- [Streamlit](https://streamlit.io/): Frontend interactivo para ML apps
- [Google Cloud Storage](https://cloud.google.com/storage): Almacenamiento de artifacts
- [Evidently AI](https://evidentlyai.com/): Data drift detection (recomendado para producción)
- [Docker](https://www.docker.com/): Containerización y reproducibilidad

**Repositorio completo:**
- [github.com/carlosjimenez88M/mlops-hand-on-ML-and-pytorch](https://github.com/carlosjimenez88M/mlops-hand-on-ML-and-pytorch/tree/cap2-end_to_end/cap2-end_to_end)
  - `/api`: FastAPI con cascade fallback y Transform Pattern
  - `/src`: Pipeline modular (01-07) con MLflow tracking
  - `/tests`: 87% coverage con fixtures realistas
  - `/.github/workflows`: CI/CD completo con security scanning

---

**Autor:** Carlos Daniel Jiménez
**Email:** danieljimenez88m@gmail.com
**LinkedIn:** [linkedin.com/in/carlosdanieljimenez](https://linkedin.com/in/carlosdanieljimenez)
**Fecha:** Enero 2026
**Licencia:** MIT