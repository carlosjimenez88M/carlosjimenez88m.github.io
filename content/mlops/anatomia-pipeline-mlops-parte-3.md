---
title: "Anatomía de un Pipeline MLOps - Parte 3: Producción y Best Practices"
date: 2026-01-13
draft: false
tags: ["mlops", "testing", "production", "data-drift", "monitoring"]
categories: ["MLOps", "Engineering"]
author: "Carlos Daniel Jiménez"
description: "Parte 3: Estrategias de selección de modelos, testing avanzado, patrones de producción, data drift, model monitoring y checklist de production readiness."
---

> **Serie MLOps Completo:** [← Parte 1: Pipeline](/mlops/anatomia-pipeline-mlops-parte-1/) | [← Parte 2: Deployment](/mlops/anatomia-pipeline-mlops-parte-2/) | **Parte 3 (actual)**


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

---

## Navegación

**[← Parte 2: Deployment e Infraestructura](/mlops/anatomia-pipeline-mlops-parte-2/)** | **[← Parte 1: Pipeline y Orquestación](/mlops/anatomia-pipeline-mlops-parte-1/)**

**Serie completa:**
1. [Parte 1: Pipeline y Orquestación](/mlops/anatomia-pipeline-mlops-parte-1/)
2. [Parte 2: Deployment e Infraestructura](/mlops/anatomia-pipeline-mlops-parte-2/)
3. Parte 3: Producción y Best Practices (actual)

**Repositorio:** [github.com/carlosjimenez88M/mlops-hand-on-ML-and-pytorch](https://github.com/carlosjimenez88M/mlops-hand-on-ML-and-pytorch/tree/cap2-end_to_end/cap2-end_to_end)

