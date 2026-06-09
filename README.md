# Supply Chain Intelligent System

## Demo en video
[![Ver demo en Loom](https://img.shields.io/badge/Ver%20Demo-Loom-625DF5?style=for-the-badge&logo=loom)](https://www.loom.com/share/92b8ea63b7d94b26801402fe4d50f852)

Sistema computacional integrado que resuelve tres decisiones simultáneas en una red de logística industrial: **optimización de distribución**, **predicción de demanda y clasificación de riesgo**, y **monitoreo autónomo con alertas proactivas** mediante un agente inteligente.

---

## Características principales

| Módulo | Tecnología | Qué resuelve |
|--------|-----------|--------------|
| **1 · Optimización** | PuLP + CBC | Minimiza costo de distribución desde 5 almacenes a 20 puntos de demanda respetando capacidades |
| **2 · Machine Learning** | Random Forest + XGBoost | Predice demanda futura y clasifica puntos en riesgo de desabasto |
| **3 · Agente Inteligente** | ReAct (implementación propia) | Monitorea el sistema, detecta anomalías y emite alertas priorizadas |

---

## Arquitectura del sistema

```
logistics-challenge/
├── data/                        # Artefactos generados en ejecución
│   └── agent_trace.json         # Traza completa del razonamiento del agente
├── notebooks/
│   ├── exploration.py           # EDA: estadísticas, series de tiempo, heatmaps, riesgo
│   ├── fig1_demand_distribution.png
│   ├── fig2_demand_time_series.png
│   ├── fig3_cost_matrix.png
│   └── fig4_risk_features.png
├── src/
│   ├── data_generator.py        # Generación de datos sintéticos reproducibles (seed=42)
│   ├── optimization.py          # Modelo de transporte con PuLP (Módulo 1)
│   ├── ml_models.py             # Pipeline ML completo: forecast + clasificación (Módulo 2)
│   ├── agent.py                 # Agente ReAct con loop OBSERVE→REASON→ACT (Módulo 3)
│   ├── visualization.py         # Dashboard de flujos y costos (matplotlib/seaborn)
│   └── utils.py                 # Funciones auxiliares compartidas
├── tests/
│   ├── test_data_generator.py   # 4 tests: dimensiones, reproducibilidad, factibilidad
│   ├── test_optimization.py     # 3 tests: costo, demanda satisfecha, análisis sensibilidad
│   └── test_agent.py            # 5 tests: detección de riesgo, alertas, herramientas
├── main.py                      # Punto de entrada único — ejecuta los 3 módulos en secuencia
├── requirements.txt             # Dependencias con versiones fijas
└── README.md
```

### Flujo de datos entre módulos

```
data_generator.py
      │
      ├──► optimization.py   →  solve_transport()          (Módulo 1)
      │
      ├──► ml_models.py      →  walk_forward_validation()  (Módulo 2A · evaluación)
      │                      →  train_risk_models()         (Módulo 2B · evaluación)
      │                      →  build_forecast_model()      (modelo de predicción RF)
      │                      →  build_risk_classifier()     (modelo de predicción XGB)
      │
      └──► agent.py          →  consume modelos de ml_models.py
                             →  llama a solve_transport() de optimization.py
                             →  emite alertas y exporta traza JSON
```

---

## Módulo 1 · Optimización

**Problema:** Minimizar el costo total de distribución en una red de 5 almacenes → 20 puntos de demanda.

**Formulación matemática:**

```
Minimizar:   Σ c_ij · x_ij   para todo (i,j)

Sujeto a:
  Σ_i x_ij  ≥  d_j          (demanda satisfecha en cada punto j)
  Σ_j x_ij  ≤  cap_i        (capacidad respetada en cada almacén i)
  x_ij      ≥  0            (flujos no negativos)
```

**Análisis de sensibilidad:** Se re-resuelve el modelo con demanda +20% para cuantificar el impacto en costo.

---

## Módulo 2 · Machine Learning

### 2A · Pronóstico de demanda (Regresión)

- Serie temporal sintética de 52 semanas con tendencia, estacionalidad y ruido
- Features: lags 1, 2 y 3 de demanda semanal
- Modelos: **Random Forest** y **XGBoost Regressor**
- Validación: **Walk-forward** (TimeSeriesSplit, 5 folds) — sin data leakage
- Métricas: MAE, RMSE, MAPE

### 2B · Clasificación de riesgo de desabasto

| Feature | Descripción |
|---------|-------------|
| `stock` | Inventario actual en el punto de demanda |
| `lead_time` | Días hasta recibir reabastecimiento |
| `demand` | Demanda base del punto |
| `distance` | Distancia proxy al almacén (costo × 10) |

- Modelos: **Random Forest Classifier** + **XGBoost Classifier** con `class_weight="balanced"`
- Métricas: F1-score, ROC-AUC, matriz de confusión
- Target: `riesgo_alto` (top 25% del score de riesgo por cobertura)

### 2C · Interpretabilidad

Feature importance via `feature_importances_` (equivalente a SHAP para modelos de árbol). La variable de mayor impacto es `stock`, seguida de `lead_time`.

---

## Análisis exploratorio — Insights clave

> Generado por `notebooks/exploration.py` sobre los datos sintéticos con `seed=42`.

### Red de distribución

| Métrica | Valor |
|---------|-------|
| Demanda total semanal | 3,434 unidades |
| Capacidad total | 4,461 unidades |
| Holgura de capacidad | 1,027 unidades (**+29.9%** sobre la demanda) |
| Costo mínimo por ruta | 5 $/unidad |
| Costo máximo por ruta | 49 $/unidad |
| Costo promedio | 27.5 $/unidad (σ = 13.5) |

La holgura del 29.9% garantiza que el problema de optimización siempre tiene solución factible, incluso en el escenario de estrés con demanda +20%.

### Variabilidad de la demanda

| Métrica | Valor |
|---------|-------|
| Demanda semanal promedio | 172.1 unidades |
| Desviación estándar | 44.9 unidades |
| Coeficiente de variación | 26.1% |

Un CV de 26% confirma variabilidad moderada — suficiente para justificar modelos de pronóstico, pero sin outliers extremos que distorsionen el entrenamiento.

### Dataset de clasificación de riesgo

| Métrica | Valor |
|---------|-------|
| Total de muestras | 200 |
| Muestras alto riesgo | 50 (25%) |
| Muestras bajo riesgo | 150 (75%) |
| Ratio de desbalance | 3:1 |
| Correlación `stock` vs riesgo | **-0.444** (mayor correlación absoluta) |

El desbalance 3:1 justifica el uso de `class_weight="balanced"` en Random Forest y el criterio `scale_pos_weight` implícito en XGBoost. La correlación negativa de `stock` con el riesgo (-0.444) es consistente con la intuición de negocio: **a menor inventario, mayor probabilidad de desabasto** — y con los resultados de feature importance del Módulo 2.

### Figuras generadas

| Figura | Contenido |
|--------|-----------|
| `fig1_demand_distribution.png` | Demanda por punto, capacidad por almacén, distribución de costos |
| `fig2_demand_time_series.png` | Series de tiempo de demanda semanal (puntos seleccionados) |
| `fig3_cost_matrix.png` | Heatmap anotado de la matriz de costos de transporte |
| `fig4_risk_features.png` | Distribución de features por clase, correlaciones y balance de clases |

Para regenerar las figuras:
```bash
python notebooks/exploration.py
```

---

## Módulo 3 · Agente Inteligente (ReAct)

El agente implementa el patrón **ReAct** (Reason + Act) con lógica determinista — no requiere LLM externo, garantizando reproducibilidad total sin API keys.

### Herramientas disponibles

| Herramienta | Firma | Descripción |
|------------|-------|-------------|
| `get_stock_status` | `(warehouse_id: int)` | Retorna inventario actual y fill-rate del almacén |
| `get_demand_forecast` | `(point_id: int, weeks: int)` | Pronóstico multi-paso con RF + clasificación de riesgo con XGB |
| `run_optimization` | `(scenario: str)` | Ejecuta el solver PuLP del Módulo 1 (`"base"` o `"stress"`) |
| `send_alert` | `(point_id, message, severity)` | Simula notificación proactiva al cliente |

### Loop de razonamiento

```
┌─────────────────────────────────────────────────────────┐
│  OBSERVE  →  consulta stock de 5 almacenes              │
│           →  consulta pronóstico y riesgo de 20 puntos  │
│                                                         │
│  REASON   →  identifica puntos HIGH/CRITICAL            │
│           →  detecta almacenes en nivel crítico         │
│                                                         │
│  ACT      →  run_optimization() si hay almacén crítico  │
│           →  send_alert() por cada punto en riesgo      │
│                                                         │
│  OBSERVE  →  verifica condición de paro                 │
│                                                         │
│  STOP     →  todos los puntos en riesgo fueron          │
│              atendidos (o MAX_ITERATIONS alcanzado)     │
└─────────────────────────────────────────────────────────┘
```

**Severidad de alertas:**

| Probabilidad de riesgo | Severidad |
|------------------------|-----------|
| ≥ 0.75 | `CRITICAL` |
| 0.50 – 0.74 | `HIGH` |
| < 0.50 | No se alerta |

**Trazabilidad:** Cada paso del razonamiento se exporta a `data/agent_trace.json` con timestamp, herramienta utilizada, inputs, outputs y texto de razonamiento.

---

## Requisitos previos

- Python **3.10** o superior
- pip

Dependencias principales:

| Librería | Versión | Uso |
|----------|---------|-----|
| `PuLP` | 3.3.2 | Solver de programación lineal (Módulo 1) |
| `scikit-learn` | 1.9.0 | Random Forest regressor y clasificador (Módulo 2) |
| `xgboost` | 3.2.0 | XGBoost regressor y clasificador (Módulo 2) |
| `pandas` | 3.0.3 | Manipulación de datos |
| `numpy` | 2.4.6 | Cálculo numérico y semillas aleatorias |
| `matplotlib` / `seaborn` | 3.10.9 / 0.13.2 | Visualizaciones (Módulo 1) |
| `pytest` | 9.0.3 | Suite de tests unitarios |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/bernardosepulvedag-design/Logistics-Challenge-BSG.git
cd Logistics-Challenge-BSG

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución

### Sistema completo (los 3 módulos en secuencia)

```bash
python main.py
```

Esto ejecuta en orden:
1. **Módulo 1** — genera datos, resuelve optimización, muestra análisis de sensibilidad y abre el dashboard de visualización *(cerrar la ventana para continuar)*
2. **Módulo 2** — entrena modelos, imprime métricas y resumen ejecutivo en terminal
3. **Módulo 3** — ejecuta el agente ReAct, imprime cada paso del razonamiento y exporta la traza a `data/agent_trace.json`

### Solo el agente (Módulo 3)

```python
from src.agent import LogisticsAgent

agent = LogisticsAgent(seed=42)
summary = agent.run()
agent.export_trace(path="data/agent_trace.json")
```

### Solo la optimización (Módulo 1)

```python
from src.data_generator import generate_data
from src.optimization import solve_transport, solve_transport_sensitivity

warehouses, demand_points, cost_matrix, _ = generate_data(seed=42)

flow_df, base_cost = solve_transport(warehouses, demand_points, cost_matrix)
_, stress_cost = solve_transport_sensitivity(warehouses, demand_points, cost_matrix, increase=0.2)

print(f"Costo base:   {base_cost:,.2f}")
print(f"Costo stress: {stress_cost:,.2f}")
```

### Análisis exploratorio (notebooks)

```bash
python notebooks/exploration.py
```

Genera estadísticas descriptivas en terminal y guarda 4 figuras en `notebooks/`.

### Solo los modelos ML (Módulo 2)

```python
from src.ml_models import (
    generate_time_series, create_features, walk_forward_validation,
    generate_risk_dataset, train_risk_models, get_feature_importance
)

# Pronóstico
df_feat = create_features(generate_time_series())
metrics = walk_forward_validation(df_feat)

# Clasificación de riesgo
df_risk = generate_risk_dataset(seed=42)
results = train_risk_models(df_risk)
print(f"XGBoost AUC: {results['XGB']['AUC']:.3f}")
```

---

## Tests unitarios

```bash
# Correr todos los tests
python -m pytest tests/ -v

# Correr un archivo específico
python -m pytest tests/test_agent.py -v

# Suprimir warnings de deprecación de librerías
python -m pytest tests/ -v -W ignore
```

**Cobertura:** 12 tests en 3 archivos — todos deben pasar en verde.

| Archivo | Tests | Qué valida |
|---------|-------|------------|
| `test_data_generator.py` | 4 | Dimensiones, reproducibilidad, factibilidad del problema |
| `test_optimization.py` | 3 | Costo positivo, demanda satisfecha, stress > base |
| `test_agent.py` | 5 | Detección de riesgo, alertas, herramientas del agente |

---

## Reproducibilidad

Todo el sistema usa `seed=42` de forma consistente:

- `generate_data(seed=42)` — datos sintéticos idénticos en cada ejecución
- `RandomForestRegressor(random_state=42)` — modelo de pronóstico determinista
- `XGBClassifier(random_state=42)` — clasificador de riesgo determinista
- `LogisticsAgent(seed=42)` — inventarios simulados y modelos idénticos

Cambiar la semilla altera los datos y los resultados, pero el sistema siempre produce resultados consistentes para la misma semilla.

---

## Decisiones de diseño

**¿Por qué implementación propia del agente y no LangChain?**
El razonamiento en supply chain es determinista por naturaleza. Una implementación propia garantiza reproducibilidad total (sin API keys, sin variabilidad de respuestas externas), trazabilidad completa del loop y cero dependencias externas de red.

**¿Por qué Random Forest para el modelo de predicción?**
Walk-forward validation en el Módulo 2 confirmó que RF supera a XGBoost en este dataset (MAE: 16.62 vs superior en RF). El agente usa el modelo validado, no uno reentrenado.

**¿Por qué XGBoost para clasificación de riesgo?**
XGBoost alcanzó F1=0.812 y AUC=0.948, superando a Random Forest. Además maneja nativamente el desbalance de clases con `scale_pos_weight`.

---

## Limitaciones identificadas

- Los datos son 100% sintéticos — los modelos no han sido validados con datos reales de operación
- El agente no persiste estado entre ejecuciones (cada corrida comienza desde cero)
- La visualización del Módulo 1 requiere interfaz gráfica (no compatible con entornos headless sin configuración adicional)
