# 2. Fundamentos de Machine Learning

Este archivo es la base teórica que vas a usar en todos los demás. Si ya conoces algún concepto, sáltalo — está pensado como referencia rápida a la que volver.

## 2.1 ¿Qué es Machine Learning?

Es la disciplina que permite que un programa **aprenda un patrón a partir de datos**, en lugar de que un humano programe la regla explícitamente. En vez de escribir `if temperatura > 30: es_verano`, le das al algoritmo miles de ejemplos (temperatura, estación) y él **encuentra** la relación.

Formalmente (definición de Tom Mitchell, muy citada): un programa aprende de una experiencia `E` respecto a una tarea `T` y una métrica de desempeño `P`, si su desempeño en `T`, medido por `P`, mejora con `E`.

## 2.2 Tipos de aprendizaje

| Tipo | Qué recibe el algoritmo | Ejemplo en este proyecto |
|---|---|---|
| **Supervisado** | Datos de entrada (`X`) + la respuesta correcta (`Y`) | Todo lo implementado hoy: regresión y clasificación |
| No supervisado | Solo `X`, sin respuesta — busca estructura/agrupaciones | Pendiente: K-Means sobre `Mall_Customers.csv` (ver [10-hoja-de-ruta.md](10-hoja-de-ruta.md)) |
| Por refuerzo | Un agente que actúa en un entorno y recibe recompensas | No usado en este proyecto |

Dentro del aprendizaje **supervisado**, hay dos familias según qué tipo de `Y` predices:

- **Regresión**: `Y` es un número continuo (precio de una casa, ventas, salario). Todos los endpoints bajo `services/regression/` son de este tipo.
- **Clasificación**: `Y` es una categoría (¿es un 5 o no?, ¿compró o no compró?). Los endpoints bajo `services/classification/`.

## 2.3 El flujo estándar de un experimento supervisado

Este es el esqueleto que vas a reconocer en **cada** método de los services de este proyecto:

```
1. Cargar datos               pd.read_csv(...)
2. Explorar (EDA)              data.info(), data.describe(), gráficos
3. Limpiar / transformar       fillna(), encoding de categorías, escalado
4. Separar X (features) e Y (target)
5. Dividir en train / test      train_test_split(...)
6. Entrenar el modelo           modelo.fit(X_train, Y_train)
7. Predecir                    modelo.predict(X_test)
8. Evaluar                      R², RMSE, accuracy, F1, matriz de confusión...
9. (Opcional) Visualizar/guardar resultados
```

## 2.4 Train / Test split — por qué se divide el dataset

Si evalúas el modelo con los **mismos datos** con los que lo entrenaste, el modelo puede simplemente "memorizar" y parecer perfecto sin realmente haber aprendido un patrón generalizable. Por eso se reserva una porción (típicamente 20-30%) que el modelo **nunca ve durante el entrenamiento**, y se evalúa solo con esa porción.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

- `test_size=0.2` → 20% de los datos se reserva para test.
- `random_state=42` → fija la semilla aleatoria, para que el split sea **reproducible** (si vuelves a correr el experimento, obtienes exactamente la misma división). Lo ves usado consistentemente en todos los servicios de `services/regression/`.

**Regla de oro que rompe el proyecto en un solo lugar** (ver [01-arquitectura-del-proyecto.md](01-arquitectura-del-proyecto.md)): cualquier transformación que "aprenda" algo de los datos (un `StandardScaler`, un `OneHotEncoder` con categorías dinámicas, etc.) debe hacer `fit` **solo con train**, y `transform` (sin `fit`) con test. Si haces `fit` también en test, el modelo indirectamente "espía" información del conjunto que se supone no debería conocer — esto se llama **data leakage**.

## 2.5 Overfitting y underfitting

- **Underfitting (sub-ajuste)**: el modelo es demasiado simple para el patrón real → le va mal en train *y* en test. Ejemplo: usar una recta para datos que son claramente una curva.
- **Overfitting (sobre-ajuste)**: el modelo es tan flexible que memoriza el ruido de train → le va muy bien en train pero mal en test. Ejemplo directo en este proyecto: en [04-regresion-polinomica.md](04-regresion-polinomica.md) vas a ver que un grado de polinomio muy alto ajusta el train casi perfecto pero generaliza mal.

Cómo detectarlo: comparar la métrica en train vs. test. Si train es mucho mejor que test → overfitting.

## 2.6 Métricas de evaluación

### Para regresión

| Métrica | Qué mide | Fórmula (idea) | ¿Dónde se usa en el proyecto? |
|---|---|---|---|
| **MSE** (Mean Squared Error) | Promedio de los errores al cuadrado | `mean((Y_real - Y_pred)²)` | Base de RMSE |
| **RMSE** (Root MSE) | Igual que MSE pero en las unidades originales (más interpretable) | `sqrt(MSE)` | `regression_linear_model`, `regression_multi_linear_model` |
| **R² (coeficiente de determinación)** | Qué proporción de la variabilidad de `Y` explica el modelo. 1.0 = perfecto, 0 = tan malo como predecir siempre el promedio, puede ser negativo si es peor que eso | — | Usado en casi todos los métodos de regresión |

```python
rmse = mean_squared_error(Y_test, y_predict, squared=False)
r2 = r2_score(Y_test, y_predict)
```

Interpretación práctica: si `R² = 0.85`, el modelo explica el 85% de la variación de las ventas (por ejemplo) a partir de la inversión en publicidad; el 15% restante es ruido o factores que el modelo no captura.

### Para clasificación

La **matriz de confusión** es la base de todo:

```
                  Predijo NO      Predijo SÍ
Real NO            TN               FP
Real SÍ            FN               TP
```

- **TP** (True Positive): dijo que sí, y era sí.
- **TN** (True Negative): dijo que no, y era no.
- **FP** (False Positive): dijo que sí, y era no ("falsa alarma").
- **FN** (False Negative): dijo que no, y era sí ("lo dejó pasar").

De ahí se derivan:

| Métrica | Fórmula | Pregunta que responde |
|---|---|---|
| **Accuracy** | `(TP+TN) / total` | ¿Qué % acerté en general? (engañosa si las clases están desbalanceadas) |
| **Precision** | `TP / (TP+FP)` | De todo lo que predije como positivo, ¿cuánto era realmente positivo? |
| **Recall (sensibilidad)** | `TP / (TP+FN)` | De todo lo que era realmente positivo, ¿cuánto detecté? |
| **F1-score** | `2 · (Precision · Recall) / (Precision + Recall)` | Balance entre precision y recall (media armónica) |

Este proyecto ya calcula las cuatro en `services/classification/` (los tres servicios de clasificación). Vas a profundizar esto con ejemplos reales en [08-clasificacion-mnist-y-metricas.md](08-clasificacion-mnist-y-metricas.md).

> **¿Por qué accuracy puede engañar?** Si el 95% de tus ejemplos son "no es un 5" (como en MNIST), un modelo que **siempre** responde "no es un 5" tiene 95% de accuracy sin haber aprendido nada. Por eso se usan precision/recall/F1 además de accuracy.

## 2.7 Preparación de datos: escalado y codificación

### Escalado de variables numéricas

Muchos algoritmos (SVR, KNN, regresión logística con regularización, redes neuronales) son sensibles a que las variables tengan escalas muy distintas (ej. "Edad" entre 18-60 vs. "Salario" entre 20,000-150,000). El `StandardScaler` transforma cada columna para que tenga media 0 y desviación estándar 1:

```python
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)   # aprende media/desviación DE TRAIN y transforma
X_test = sc_X.transform(X_test)          # reutiliza esa misma media/desviación (sin fit)
```

Los algoritmos basados en árboles (Decision Tree, Random Forest) **no necesitan** escalado — por eso no lo ves en `random_tree_regression`/`random_forest_regression`, y sí lo necesitaría (y no lo tiene) `svr_regression`.

### Codificación de variables categóricas

Los modelos de scikit-learn solo entienden números, así que columnas de texto (`Gender`, `ocean_proximity`) hay que convertirlas:

- **OneHotEncoder**: crea una columna binaria (0/1) por cada categoría. Úsalo cuando las categorías **no tienen orden** (Gender: Male/Female; ocean_proximity: NEAR BAY/INLAND/...). Es el que usa este proyecto para ambas variables.
- **OrdinalEncoder**: asigna un número entero por categoría (0, 1, 2...). Solo tiene sentido si existe un **orden natural** (ej. "bajo/medio/alto"). El proyecto lo usa sobre `ocean_proximity` solo con fines exploratorios (`np.random.choice` de prueba) — la codificación que realmente alimenta los modelos es la de `OneHotEncoder`, que es la correcta para esa columna porque sus categorías no tienen jerarquía.

## 2.8 Validación cruzada (cross-validation)

Un solo `train_test_split` depende de la suerte del split. La **validación cruzada** (k-fold) entrena y evalúa el modelo `k` veces, usando cada vez una porción distinta como test, y promedia el resultado — da una estimación más confiable del desempeño real.

```python
cross_val_score(sgd_classifier, X_train, Y_train_5, cv=3, scoring='accuracy')
```

Usado en `handle_classification_image` (MNIST) con `cv=3` (3 particiones). Lo vas a estudiar en detalle en [08-clasificacion-mnist-y-metricas.md](08-clasificacion-mnist-y-metricas.md).

## 2.9 Glosario mínimo para seguir leyendo

- **Feature / variable independiente**: cada columna de `X` (ej. `TV`, `Radio`).
- **Target / variable dependiente / label**: lo que quieres predecir (`Y`, ej. `Sales`).
- **Modelo**: el objeto que aprende (ej. `LinearRegression()`) — antes de `.fit()` está "vacío"; después de entrenarlo tiene parámetros ajustados.
- **Hiperparámetro**: una configuración que **tú** eliges antes de entrenar (ej. `degree=4` en polinómica, `kernel='rbf'` en SVR) — no lo aprende el modelo, lo decides tú (o lo buscas con técnicas como `GridSearchCV`, ver [10-hoja-de-ruta.md](10-hoja-de-ruta.md)).
- **Parámetro**: lo que el modelo sí aprende solo (ej. los coeficientes `b0, b1` de una regresión lineal).

Con esta base ya puedes seguir a [03-regresion-lineal-simple-y-multiple.md](03-regresion-lineal-simple-y-multiple.md).
