# 11. El pipeline de Machine Learning: de los datos a la evaluación

> Requiere: [02-fundamentos-de-machine-learning.md](02-fundamentos-de-machine-learning.md). Este capítulo une las piezas que los capítulos 03 a 08 explican por técnica: cómo recorre un dato el camino completo, desde su origen hasta el resultado que devuelve la API.

Un modelo de ML rara vez falla por el algoritmo; suele fallar por lo que pasa **antes** (datos sucios, fuga de información) o **después** (evaluación engañosa). Por eso el trabajo se organiza como un **pipeline**: una secuencia de etapas donde la salida de una es la entrada de la siguiente.

## 11.1 Visión general

```
┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐
│ 1.         │   │ 2. Limpieza  │   │ 3. División│   │ 4.           │   │ 5.         │   │ 6. Entrega │
│ Extracción │──►│ y preparación│──►│ train/test │──►│ Entrenamiento│──►│ Evaluación │──►│ del        │
│ de datos   │   │ de datos     │   │            │   │ del modelo   │   │            │   │ resultado  │
└────────────┘   └──────────────┘   └────────────┘   └──────────────┘   └────────────┘   └────────────┘
  CSV, OpenML,     nulos, nuevas      80 % / 20 %      fit(X_train,       métricas sobre    JSON con
  datos sintéticos variables,         reproducible     Y_train)           test, train y     métricas +
                   codificación,                                          validación        gráficos
                   escalado                                               cruzada
```

| Etapa | Pregunta que responde | Dónde vive en el código |
|---|---|---|
| 1. Extracción | ¿De dónde salen los datos y cómo se cargan? | `core/paths.py`, `load_*` en `services/` |
| 2. Limpieza y preparación | ¿Los datos están en una forma que el modelo entienda? | `services/shared/housing_preprocessing.py`, `social_ads_preprocessing.py` |
| 3. División | ¿Qué datos se reservan para evaluar? | `train_test_split` en cada service |
| 4. Entrenamiento | ¿Cómo aprende el modelo? | Métodos `_train*` de cada service |
| 5. Evaluación | ¿Qué tan bueno es el modelo con datos nuevos? | `services/shared/evaluation.py` y los `_train*` |
| 6. Entrega | ¿Cómo se comunica el resultado? | Routers, `services/shared/plotting.py`, handlers de `main.py` |

Una regla atraviesa todas las etapas: **nada de lo que se aprende de los datos de test puede influir en el modelo** (ver §11.3 y [02](02-fundamentos-de-machine-learning.md), §2.4).

## 11.2 Etapa 1 — Extracción de datos

Los datos entran al proyecto por tres vías:

| Fuente | Datasets | Cómo se carga |
|---|---|---|
| **CSV local** | `Advertising.csv`, `Social_Network_Ads.csv`, `housing.csv` | `pd.read_csv(sample_data_path(...))` |
| **Datos sintéticos** | Salarios por puesto (regresión polinómica) | `build_dataset()` construye un `DataFrame` en memoria |
| **Servicio externo** | MNIST desde OpenML | `fetch_openml("mnist_784")`, con timeout de 30 s |

### Los datasets

| Dataset | Filas × columnas | Objetivo (`Y`) | Se usa en |
|---|---|---|---|
| `Advertising.csv` | 200 × 5 | `Sales` | Regresión lineal simple y múltiple, SVR |
| `Social_Network_Ads.csv` | 400 × 5 | `Purchased` (0/1; 35.75 % de compradores) | Regresión logística, KNN |
| `housing.csv` | 20 640 × 10 | `median_house_value` | Regresión lineal, árbol de decisión, random forest |
| Salarios (sintético) | 10 × 3 | `salary` | Regresión polinómica |
| MNIST (OpenML) | 70 000 × 784 | ¿es un 5? (binario) | Clasificación de imágenes |

`Mall_Customers.csv` está en `sample_data/` pero ningún endpoint lo usa todavía.

### Buenas prácticas de extracción

- **Rutas independientes de la máquina.** Todos los CSV se resuelven con `core/paths.py::sample_data_path()`, que parte de `__file__`. Así el proyecto funciona igual en local y en Docker.
- **Fallos de red controlados.** La descarga de MNIST está acotada por un timeout y, si falla, se levanta `UpstreamServiceError`, que la API traduce a un 503 en lugar de un error opaco.
- **Cada petición recarga los datos.** No hay caché ni modelo persistido: cada endpoint carga, prepara y entrena desde cero (ver la hoja de ruta en [10](10-hoja-de-ruta.md)).

## 11.3 Etapa 2 — Limpieza y preparación

Objetivo: dejar los datos en una forma que el modelo pueda usar sin distorsionar lo que aprende.

| Técnica | Qué resuelve | Dónde se aplica |
|---|---|---|
| **Eliminar columnas irrelevantes** | Identificadores y variables sin señal | `Newspaper` (poco correlacionada con `Sales`) en regresión múltiple y SVR; índice del CSV en SVR (`iloc[:, 1:]`) |
| **Imputación de nulos** | Valores faltantes que rompen el entrenamiento | `total_bedrooms` de `housing.csv` (207 nulos) se rellena con la **mediana** |
| **Ingeniería de atributos** | Variables que expresan mejor la información | `rooms_per_household`, `bedrooms_per_room`, `population_per_household` |
| **Codificación de categorías** | Los modelos solo entienden números | `OneHotEncoder` para `ocean_proximity` (housing) y `Gender` (Social Ads) |
| **Escalado** | Variables con escalas muy distintas dominan las distancias y los gradientes | `StandardScaler` en SVR, regresión logística y KNN |

### Qué modelos necesitan escalado

| Modelo | ¿Necesita escalado? | Por qué |
|---|---|---|
| Regresión lineal | No | Los coeficientes absorben la escala |
| Árbol de decisión, random forest | No | Solo comparan valores dentro de una misma columna |
| SVR (`rbf`, `poly`), KNN | **Sí** | Se basan en distancias entre puntos |
| Regresión logística | **Sí** | El optimizador converge mejor con escalas comparables |

### Cómo se evita la fuga de datos (data leakage)

Cualquier paso que **aprenda** algo de los datos (la media de un escalador, la mediana de una imputación) debe ajustarse **solo con train**. El proyecto lo resuelve así:

```
Datos sin escalar ──► split train / test
                          │
              Pipeline( StandardScaler → modelo )
                          │
   fit(train):  el escalador aprende media y desviación de TRAIN
   predict(test): reutiliza esas mismas estadísticas, sin reajustarse
```

En `Social_Network_Ads.csv`, el escalado y el modelo se encapsulan en un `Pipeline` de scikit-learn. Es más que una comodidad: en la validación cruzada (§11.5) el `Pipeline` reajusta el escalador **dentro de cada partición**, de modo que ninguna partición de validación influye en su propio escalado.

> **Nota de rigor.** La imputación por mediana de `housing.csv` y el `OneHotEncoder` se calculan sobre el dataset completo antes de dividir. El efecto práctico es pequeño (207 nulos sobre 20 640 filas, y un codificador que solo aprende el conjunto de categorías), pero la práctica estricta es ajustarlos solo con train dentro de un `Pipeline`, con `ColumnTransformer`. Está anotado en la [hoja de ruta](10-hoja-de-ruta.md).

## 11.4 Etapa 3 — División en train y test

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

| Decisión | Valor en el proyecto | Motivo |
|---|---|---|
| Proporción de test | 20 % | Equilibrio entre datos para aprender y datos para evaluar |
| Semilla (`random_state`) | 42 en regresión, 0 en clasificación sobre Social Ads | Reproducibilidad: la misma división en cada ejecución |
| MNIST | Primeras 60 000 filas para train, últimas 10 000 para test | Es la partición estándar del dataset, ya definida |
| Regresión polinómica | Sin división (solo 10 filas) | Un 20 % dejaría 2 filas de test; se usa validación cruzada *leave-one-out* (ver [12](12-metricas-de-evaluacion.md), §12.8) |

## 11.5 Etapa 4 — Entrenamiento

Entrenar es ajustar los **parámetros** del modelo (coeficientes, umbrales de un árbol, vectores de soporte) para que sus predicciones se acerquen a `Y` en los datos de train:

```python
modelo.fit(X_train, Y_train)      # aprende
y_pred = modelo.predict(X_test)   # aplica lo aprendido a datos que no vio
```

Los **hiperparámetros** (lo que el modelo no aprende sino que se elige antes) se exponen como campos opcionales del body de cada endpoint:

| Modelo | Endpoint | Hiperparámetros configurables | Por defecto |
|---|---|---|---|
| Regresión lineal | `/v1/linear-regression`, `/v1/multi-linear-regression` | `column_name` (solo la simple) | — |
| Regresión polinómica | `/v1/polynomial-regression` | `degree` (1 a 10) | 4 |
| SVR | `/v1/svr-regression` | `kernel` (`linear`, `poly`, `rbf`) | `rbf` |
| Árbol de decisión | `/v1/decision-tree-regression` | `max_depth` | sin límite |
| Random forest | `/v1/random-forest-regression` | `n_estimators`, `max_depth` | 50 árboles |
| Regresión logística | `/v1/logistic-regression-classification` | — | — |
| KNN | `/v1/knn-classification` | `n_neighbors` (1 a 50) | 5 |
| SGD sobre MNIST | `/v1/classification-algorithm` | — | — |

Cada llamada **entrena el modelo desde cero**. Es adecuado para estudiar, pero no para producción, donde se entrena una vez, se guarda el modelo (`joblib`) y la API solo predice (ver [10](10-hoja-de-ruta.md), §10.5).

## 11.6 Etapa 5 — Evaluación

Entrenar no dice si el modelo es bueno. La evaluación mide cuánto se equivoca con datos que no vio y **diagnostica** si memoriza, si es estable y cuánto se puede confiar en el resultado. Cada modelo se evalúa con tres lentes complementarias:

```
                          ┌──────────────────────────────────────────┐
                          │             MODELO ENTRENADO             │
                          └──────────────┬───────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  Métricas sobre TEST            Métrica sobre TRAIN            VALIDACIÓN CRUZADA
  (¿generaliza?)                 (¿memoriza?)                   (¿es estable?)
  R², RMSE  /  precision,        R² (regresión) /               media ± desviación de
  recall, F1, ROC-AUC            F1 (clasificación)             R², RMSE  /  F1, ROC-AUC
        │                                │                                │
        └────────────────────┬───────────┴────────────────────────────────┘
                             ▼
              Comparar: test contra train → overfitting
                        test contra validación cruzada → ¿el split fue afortunado?
```

El detalle de cada métrica, cuándo usarla y cómo leer estas comparaciones está en [12-metricas-de-evaluacion.md](12-metricas-de-evaluacion.md).

## 11.7 Etapa 6 — Entrega del resultado

| Elemento | Qué hace |
|---|---|
| **Respuesta JSON** | Cada endpoint devuelve las métricas de la etapa 5 como JSON estructurado |
| **Gráficos** | `saved_figure()` guarda un PNG único por petición en `results_graphics/` y devuelve su nombre en `plot_file`; siempre cierra la figura, incluso si el dibujo falla |
| **Errores de negocio** | `InvalidTrainingDataError` → 422 (petición incompatible con el dataset), `UpstreamServiceError` → 503 (servicio externo caído) |
| **Errores inesperados** | Un handler global responde 500 genérico sin filtrar detalles internos |

## 11.8 Mapa completo: el pipeline por endpoint

| Endpoint | Extracción | Limpieza y preparación | División | Modelo | Evaluación |
|---|---|---|---|---|---|
| `/v1/linear-regression` | `Advertising.csv` | Una columna elegida | 80/20 | `LinearRegression` | R², RMSE, R² train, validación cruzada |
| `/v1/multi-linear-regression` | `Advertising.csv` | Quita `Newspaper` | 80/20 | `LinearRegression` | R², RMSE, R² train, validación cruzada |
| `/v1/svr-regression` | `Advertising.csv` | Quita `Newspaper`; escala `X` e `Y` | 80/20 | `SVR` | R², RMSE, R² train, validación cruzada |
| `/v1/polynomial-regression` | Datos sintéticos | `PolynomialFeatures` | Ninguna | `LinearRegression` sobre polinomio | R² de ajuste, R² y RMSE *leave-one-out* |
| `/v1/housing-linear-regression` | `housing.csv` | Mediana, ratios, `OneHotEncoder` | 80/20 | `LinearRegression` | R², RMSE, R² train por paso; validación cruzada del modelo completo |
| `/v1/decision-tree-regression` | `housing.csv` | Igual que arriba | 80/20 | `DecisionTreeRegressor` | Igual que arriba |
| `/v1/random-forest-regression` | `housing.csv` | Igual que arriba | 80/20 | `RandomForestRegressor` | Igual que arriba |
| `/v1/logistic-regression-classification` | `Social_Network_Ads.csv` | `OneHotEncoder`; `Pipeline` con escalado | 80/20 | `LogisticRegression` | Matriz de confusión, precision, recall, F1, ROC-AUC, F1 train, validación cruzada |
| `/v1/knn-classification` | `Social_Network_Ads.csv` | Igual que arriba | 80/20 | `KNeighborsClassifier` | Igual que arriba |
| `/v1/classification-algorithm` | OpenML (MNIST) | Binariza: ¿es un 5? | 60 000 / 10 000 | `SGDClassifier` | Validación cruzada sobre train y métricas sobre test |

## 11.9 Errores comunes en el pipeline

- **Ajustar transformaciones con todo el dataset** (escalar antes de dividir): filtra información del test.
- **Evaluar con los datos de entrenamiento:** mide memoria, no capacidad de generalizar.
- **Olvidar el escalado** en modelos basados en distancias (SVR, KNN).
- **Codificar categorías sin orden con números enteros:** el modelo interpretaría una jerarquía inexistente; usar `OneHotEncoder`.
- **Elegir hiperparámetros mirando el test:** ese conjunto deja de ser imparcial (ver [12](12-metricas-de-evaluacion.md), §12.12).
- **Cambiar la semilla hasta obtener un buen resultado:** invalida la evaluación.
- **Ignorar valores topados:** en `housing.csv`, `median_house_value` tiene 965 filas con el valor máximo (500 001), lo que sugiere censura en los datos originales y limita la precisión del modelo en ese rango.

## 11.10 Para seguir practicando

- Recorre `POST /v1/knn-classification` etapa por etapa: identifica en el código dónde ocurre cada una (carga, codificación, división, `Pipeline`, entrenamiento, métricas).
- Modifica `n_neighbors` y observa en la respuesta cómo cambian `f1_score` (test), `f1_train` y `cv_f1_mean`.
- Quita el escalado del `Pipeline` de KNN en una copia del servicio y compara las métricas.
- Aplica la imputación por mediana de `housing.csv` solo sobre train y compara el resultado con el pipeline actual.
- Sustituye el preprocesamiento manual de `housing.csv` por un `ColumnTransformer` dentro de un `Pipeline` (ver [10](10-hoja-de-ruta.md), §10.4).
