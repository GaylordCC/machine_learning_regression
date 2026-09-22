# 12. Métricas de evaluación de modelos

> Requiere: [02-fundamentos-de-machine-learning.md](02-fundamentos-de-machine-learning.md) (§2.4 train/test y §2.6 métricas). Es una guía general de **cómo se usa cada métrica y cuándo elegirla**, válida para cualquier modelo. Los ejemplos numéricos salen de los datasets del proyecto (`Advertising.csv` y `Social_Network_Ads.csv`).

Entrenar un modelo no dice si es bueno. Las **métricas de evaluación** ponen un número a qué tan bien predice, y ese número es la base para comparar modelos, ajustar hiperparámetros y decidir si el resultado sirve.

## 12.1 El flujo de evaluación

```
        Datos completos
              │
              ▼
   ┌─────────────────────┐
   │  train_test_split    │   80 % train / 20 % test
   └───────┬─────────┬───┘
           │         │
        train       test  ◄── el modelo nunca lo ve al entrenar
           │         │
           ▼         │
     modelo.fit()    │
           │         │
           ▼         ▼
     modelo.predict(X_test)  ──►  predicciones
                                       │
                 Y_test (real) ────────┤
                                       ▼
                            MÉTRICAS (comparar real vs predicho)
                                       │
                                       ▼
                 ¿train ≈ test?  ── no ──►  overfitting / underfitting
                                       │ sí
                                       ▼
                       comparar modelos o ajustar hiperparámetros
```

Regla de oro: **la métrica que se reporta se calcula sobre datos que el modelo no vio durante el entrenamiento**. Si se evalúa sobre train, el resultado mide memoria, no capacidad de generalizar.

## 12.2 Qué métrica aplica a qué tipo de problema

Las métricas dependen del tipo de `Y` que se predice y **no se mezclan**: una métrica de clasificación no tiene sentido sobre un modelo de regresión, y al revés.

| Tipo de problema | `Y` es… | Métricas | Ejemplos de modelos |
|---|---|---|---|
| **Regresión** | un número continuo | R², MSE, RMSE, MAE | Regresión lineal, polinómica, SVR, árboles, random forest |
| **Clasificación** | una categoría | Matriz de confusión, accuracy, precision, recall, F1, ROC-AUC | Regresión logística, KNN, SGD, árboles de clasificación |

Qué necesita cada métrica del modelo:

| Métrica | Necesita |
|---|---|
| R², MSE, RMSE, MAE | Las predicciones numéricas (`predict`) |
| Matriz de confusión, accuracy, precision, recall, F1 | Las clases predichas (`predict`) |
| ROC-AUC | **Probabilidades o puntajes** (`predict_proba` o `decision_function`), no solo la clase; en su forma básica, para problemas **binarios** |

### Guía rápida: cuándo usar cada métrica

| Métrica | Úsala cuando… | Evítala o complétala cuando… |
|---|---|---|
| **R²** | Se quiere una visión global y comparar modelos sobre el mismo dataset | Se necesita el error en unidades reales, o se calcula sobre train |
| **RMSE** | Interesa el error típico en unidades reales y los errores grandes son costosos | Hay valores atípicos que no deberían dominar |
| **MAE** | Se quiere un error interpretable y robusto a valores atípicos | Los errores grandes deben penalizarse con más fuerza |
| **MSE** | Como función de pérdida o paso intermedio hacia el RMSE | Se quiere reportar un error legible (sus unidades están al cuadrado) |
| **Accuracy** | Las clases están equilibradas y ambos errores pesan parecido | Las clases están desbalanceadas |
| **Precision** | Una falsa alarma es lo más costoso | Dejar pasar un positivo es lo grave |
| **Recall** | Dejar pasar un positivo es lo más costoso | Las falsas alarmas son muy costosas |
| **F1** | Se necesita balance entre precision y recall, sobre todo con clases desbalanceadas | Un error pesa mucho más que el otro |
| **ROC-AUC** | Se comparan clasificadores sin haber fijado aún el umbral | Solo interesa un umbral concreto, o el desbalance es extremo (ahí conviene la curva precision–recall) |
| **Matriz de confusión** | Siempre, como base para entender qué tipo de error comete el modelo | — |

## 12.3 Métricas de regresión

Ejemplo de referencia en esta sección: regresión lineal simple de `Sales` contra `TV` sobre `Advertising.csv` (40 filas de test; `Sales` tiene media 14.02 y desviación estándar 5.20).

### R² (coeficiente de determinación)

```
R² = 1 − (suma de errores del modelo al cuadrado) / (suma de errores de predecir siempre el promedio al cuadrado)
```

- **Qué responde**: ¿qué proporción de la variación de `Y` explica el modelo?
- **Cómo leerlo**: `1.0` es perfecto; `0` equivale a predecir siempre el promedio; **puede ser negativo** si el modelo es peor que el promedio.
- **En scikit-learn**: `r2_score(Y_test, y_pred)`. Para un regresor, `modelo.score(X_test, Y_test)` devuelve exactamente lo mismo.
- **Ventaja**: no tiene unidades, así que permite comparar modelos sobre el mismo dataset.
- **Límite**: no dice cuánto se equivoca el modelo en unidades reales (para eso están RMSE y MAE), y no distingue entre un modelo bueno y uno sobreajustado si se calcula sobre train.

### MSE (Mean Squared Error)

```
MSE = promedio( (Y_real − Y_predicho)² )
```

Promedia los errores al cuadrado. Penaliza mucho los errores grandes. Sus unidades son las de `Y` **al cuadrado**, por eso rara vez se reporta solo: es la base de RMSE y de la función que muchos algoritmos minimizan al entrenar.

### RMSE (Root Mean Squared Error)

```
RMSE = √MSE
```

Es el MSE devuelto a las unidades originales de `Y`. Se interpreta como el **error típico de una predicción**, dando más peso a los errores grandes.

- **En scikit-learn**: `root_mean_squared_error(Y_test, y_pred)`.

### MAE (Mean Absolute Error)

```
MAE = promedio( |Y_real − Y_predicho| )
```

El error absoluto promedio, en las unidades de `Y`. Trata todos los errores por igual, así que es **más robusto a valores extremos** que el RMSE.

- **En scikit-learn**: `mean_absolute_error(Y_test, y_pred)`.
- **RMSE vs MAE**: si RMSE es mucho mayor que MAE, hay algunos errores grandes que dominan; si son parecidos, los errores son homogéneos.

### Ejemplo con datos reales del proyecto

Regresión lineal simple de `Sales` contra cada medio en `Advertising.csv`, evaluada sobre test:

| Variable | R² | RMSE | MAE |
|---|---|---|---|
| `TV` | 0.677 | 3.19 | 2.44 |
| `Radio` | 0.263 | 4.82 | 3.93 |
| `Newspaper` | 0.030 | 5.53 | 4.78 |

Cómo se lee: con `TV`, el modelo explica ~68 % de la variación de las ventas y se equivoca en promedio ~2.4 unidades de venta (MAE), frente a una desviación estándar de `Sales` de 5.2. Con `Newspaper`, el R² es casi 0 y el RMSE (5.53) es prácticamente la desviación estándar de `Sales` (5.20), es decir, **apenas mejora a predecir siempre el promedio**. Es la evidencia numérica de que esa variable casi no aporta.

```python
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

y_pred = lin_reg.predict(X_test)
r2 = r2_score(Y_test, y_pred)
rmse = root_mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
```

### Cuándo usar cada una

| Situación | Métrica recomendada |
|---|---|
| Comparar modelos sobre el mismo dataset | R² |
| Saber cuánto se equivoca el modelo en unidades reales | RMSE y MAE |
| Los errores grandes son especialmente costosos (por ejemplo, precio de una vivienda) | RMSE |
| Hay valores atípicos que no deben dominar la evaluación | MAE |
| Comparar modelos sobre datasets con distinta escala | R² (no tiene unidades) |

## 12.4 Métricas de clasificación

Todas parten de la **matriz de confusión**. Ejemplo de referencia: regresión logística sobre `Social_Network_Ads.csv` (80 filas de test, 22 son compradores reales).

```
                     Predijo NO (0)    Predijo SÍ (1)
Real NO  (0)              56  (TN)          2  (FP)
Real SÍ  (1)               5  (FN)         17  (TP)
```

- **TP** (verdadero positivo): predijo que compraba y compró.
- **TN** (verdadero negativo): predijo que no compraba y no compró.
- **FP** (falso positivo): predijo que compraba y no compró ("falsa alarma").
- **FN** (falso negativo): predijo que no compraba y sí compró ("caso que se dejó pasar").

### Accuracy (exactitud)

```
accuracy = (TP + TN) / total  =  (17 + 56) / 80  =  0.9125
```

Porcentaje de aciertos en general. **Engaña cuando las clases están desbalanceadas**: en MNIST, "¿es un 5?" es cierto solo en ~10 % de los casos, y un modelo que siempre responde "no" acierta ~90 % sin haber aprendido nada.

- **En scikit-learn**: `accuracy_score(Y_test, y_pred)`; para un clasificador, `modelo.score(X_test, Y_test)` devuelve lo mismo.
- **Cuándo usarla**: con clases equilibradas y cuando falsos positivos y falsos negativos cuestan parecido. Nunca como única métrica con clases desbalanceadas.

### Precision (precisión)

```
precision = TP / (TP + FP)  =  17 / (17 + 2)  =  0.8947
```

De todo lo que el modelo **predijo positivo**, ¿qué proporción lo era? Sube cuando se reducen las falsas alarmas. Ejemplo: de cada 100 clientes que el modelo marca como compradores, ~89 lo son.

- **En scikit-learn**: `precision_score(Y_test, y_pred)`.
- **Cuándo usarla**: cuando una falsa alarma es costosa (filtro de spam, alertas que molestan al usuario, aprobar a alguien que no debería aprobarse).

### Recall (exhaustividad o sensibilidad)

```
recall = TP / (TP + FN)  =  17 / (17 + 5)  =  0.7727
```

De todos los **positivos reales**, ¿qué proporción detectó? Sube cuando se reducen los casos que se dejan pasar. Ejemplo: de cada 100 compradores reales, el modelo detecta ~77.

- **En scikit-learn**: `recall_score(Y_test, y_pred)`.
- **Cuándo usarla**: cuando dejar pasar un positivo es lo grave (enfermedades, fraude, fallas de equipos).

### F1-score

```
F1 = 2 · (precision · recall) / (precision + recall)  =  0.8293
```

Media armónica de precision y recall. Es baja si **cualquiera** de los dos es baja, así que resume ambos en un solo número. Es la métrica habitual cuando las clases están desbalanceadas y no se quiere sacrificar ni precision ni recall.

- **En scikit-learn**: `f1_score(Y_test, y_pred)`.
- **Cuándo usarla**: cuando se busca un balance entre precision y recall, o para comparar modelos con un único número en problemas desbalanceados.

### ROC-AUC

Los clasificadores como la regresión logística no responden directamente "sí/no": calculan una **probabilidad** y la comparan con un **umbral** (0.5 por defecto). ROC-AUC evalúa el modelo **para todos los umbrales posibles a la vez**.

```
                         Curva ROC
    Tasa de       1.0 ┤        ╭────────────
    verdaderos        │     ╭──╯
    positivos         │   ╭─╯          AUC = área bajo la curva
    (recall)          │  ╭╯
                      │ ╭╯          ····· diagonal = modelo aleatorio (AUC 0.5)
                  0.0 ┼─┴──────────────────
                     0.0            1.0
                     Tasa de falsos positivos
```

- **Cómo leerlo**: `1.0` = separa perfectamente las clases; `0.5` = no mejora a lanzar una moneda; `< 0.5` = peor que el azar.
- **Interpretación intuitiva**: la probabilidad de que el modelo asigne una probabilidad mayor a un positivo elegido al azar que a un negativo elegido al azar.
- **Ventaja**: no depende de un umbral concreto y es menos sensible al desbalance de clases que el accuracy.
- **Límite**: resume el modelo en un solo número; para elegir el umbral concreto de operación hace falta ver precision y recall a cada umbral.
- **Requisito**: el modelo debe entregar probabilidades o puntajes (`predict_proba` o `decision_function`).

```python
from sklearn.metrics import roc_auc_score

y_proba = log_reg.predict_proba(X_test)[:, 1]   # probabilidad de la clase positiva
auc = roc_auc_score(Y_test, y_proba)
```

En el ejemplo de `Social_Network_Ads.csv`, el ROC-AUC sobre test es **0.979** para regresión logística y **0.957** para KNN con `k=5`.

- **Cuándo usarlo**: para comparar clasificadores sin haber decidido todavía el umbral. Si las clases están muy desbalanceadas, conviene complementarlo con la curva precision–recall (`precision_recall_curve`).

### Comparación con datos reales del proyecto

Mismo dataset y mismo split (`Social_Network_Ads.csv`, 80 filas de test):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regresión logística | 0.913 | 0.895 | 0.773 | 0.829 | 0.979 |
| KNN, `k=5` | 0.950 | 0.875 | 0.955 | 0.913 | 0.957 |

Cómo se lee: KNN detecta más compradores reales (recall 0.955 frente a 0.773) a costa de una precision algo menor, por eso su F1 es mayor. La regresión logística separa mejor las clases de forma global (mayor AUC). Ningún modelo gana en todas las métricas, y elegir uno depende de qué error es más costoso.

## 12.5 Comparación de todas las métricas

### Tabla comparativa

| Métrica | Problema | Qué evalúa | Rango (mejor) | Unidades | Sensible a valores atípicos | Sensible al desbalance de clases | Necesita | Limitación principal |
|---|---|---|---|---|---|---|---|---|
| **R²** | Regresión | Qué proporción de la variación de `Y` explica el modelo | ≤ 1 (1); 0 = predecir el promedio | Ninguna | Sí | — | Predicciones | No dice cuánto se equivoca en unidades reales |
| **MSE** | Regresión | Magnitud media del error, con penalización cuadrática | ≥ 0 (0) | Unidades de `Y` al cuadrado | Muy sensible | — | Predicciones | Difícil de interpretar directamente |
| **RMSE** | Regresión | Error típico, castigando más los errores grandes | ≥ 0 (0) | Unidades de `Y` | Sensible | — | Predicciones | Depende de la escala de `Y`: no se compara entre datasets |
| **MAE** | Regresión | Error absoluto medio | ≥ 0 (0) | Unidades de `Y` | Poco (robusto) | — | Predicciones | No distingue pocos errores grandes de muchos pequeños |
| **Matriz de confusión** | Clasificación | Reparto de aciertos y errores por tipo (TP, TN, FP, FN) | Conteos | Casos | — | Refleja el desbalance | Clases predichas | No es un solo número: no permite ordenar modelos directamente |
| **Accuracy** | Clasificación | Proporción de aciertos totales | 0 a 1 (1) | Ninguna | — | **Sí: engaña** | Clases predichas | Oculta qué tipo de error se comete |
| **Precision** | Clasificación | Pureza de las predicciones positivas (falsas alarmas) | 0 a 1 (1) | Ninguna | — | Sí (depende de la proporción de positivos) | Clases predichas | Ignora los positivos que se dejaron pasar |
| **Recall** | Clasificación | Cobertura de los positivos reales (omisiones) | 0 a 1 (1) | Ninguna | — | No | Clases predichas | Ignora las falsas alarmas |
| **F1** | Clasificación | Balance entre precision y recall | 0 a 1 (1) | Ninguna | — | Parcial: ignora los negativos verdaderos | Clases predichas | Trata ambos errores como igual de graves |
| **ROC-AUC** | Clasificación | Capacidad de ordenar los positivos por encima de los negativos, para **todos** los umbrales | 0 a 1 (1); 0.5 = azar | Ninguna | — | Poco (con desbalance extremo, mejor la curva precision–recall) | Probabilidades o puntajes | No elige un umbral; incluye umbrales que no se usarían |

### Qué tienen en común

- Todas comparan lo que el modelo predijo con lo que ocurrió en realidad.
- Deben calcularse sobre datos que el modelo no vio; sobre train solo sirven para diagnosticar.
- Cada una tiene una versión de train, de test y de validación cruzada.
- Ninguna es "la mejor" en abstracto: se elige según qué error cuesta más.
- Un solo número resume el modelo pero oculta detalles; conviene combinarlas.

### En qué se diferencian

| Eje | Diferencia |
|---|---|
| **Tipo de problema** | Regresión (número) frente a clasificación (categoría): no se intercambian |
| **Qué error cuentan** | Regresión: cuánto se falla (al cuadrado o en absoluto). Clasificación: de qué tipo es el fallo (falsa alarma u omisión) |
| **Escala** | R², precision, recall, F1, accuracy y ROC-AUC no tienen unidades; RMSE, MAE y MSE sí |
| **Dependencia del umbral** | Precision, recall, F1 y accuracy dependen de un umbral; ROC-AUC no |
| **Qué necesitan del modelo** | Predicciones (regresión), clases (clasificación) o probabilidades (ROC-AUC) |
| **Robustez** | MAE es robusto a valores atípicos; MSE y RMSE los amplifican |

### Solapamientos: qué métricas miden casi lo mismo

Cuando dos métricas responden la misma pregunta, reportar ambas no aporta información nueva. Estas son las relaciones y la decisión que toma el proyecto:

| Par | Relación | Decisión en el proyecto |
|---|---|---|
| MSE y RMSE | RMSE = √MSE | Solo RMSE |
| RMSE y MAE | Ambas miden la magnitud del error y difieren solo en cuánto penalizan los errores grandes | Solo RMSE |
| R² y RMSE | Sobre los mismos datos, R² = 1 − RMSE² / varianza de `Y`: contienen la misma información en escalas distintas | Se reportan ambas porque se leen distinto: R² es relativo (¿cuánto explica?) y RMSE está en unidades (¿cuánto falla?) |
| Accuracy y matriz de confusión | accuracy = (TP + TN) / total: sale de la matriz | No se añade accuracy |
| F1, precision y recall | F1 es su media armónica | Se conservan las tres: F1 permite comparar train, test y validación cruzada con un solo número |
| ROC-AUC y las demás | Mide el orden de las probabilidades, sin depender del umbral; **no se deduce** de la matriz | Se añade: es complementaria |

### Qué pregunta responde cada herramienta

Un conjunto no redundante cubre preguntas distintas:

| Pregunta | Herramienta |
|---|---|
| ¿Qué proporción de la variación explica el modelo? | R² |
| ¿Cuánto se equivoca, en unidades reales? | RMSE |
| ¿Qué tipo de error comete? | Matriz de confusión, precision, recall |
| ¿Cómo se comporta sin depender del umbral? | ROC-AUC |
| ¿Memoriza los datos de entrenamiento? | Métrica de train frente a la de test |
| ¿El resultado es estable o dependió del split? | Validación cruzada |

## 12.6 El umbral y el trade-off precision–recall

Precision y recall se **oponen**: al mover el umbral de decisión, una sube y la otra baja. Resultados reales de la regresión logística sobre el mismo test:

| Umbral | Precision | Recall | F1 |
|---|---|---|---|
| 0.3 (más permisivo) | 0.724 | 0.955 | 0.824 |
| 0.5 (por defecto) | 0.895 | 0.773 | 0.829 |
| 0.7 (más estricto) | 1.000 | 0.636 | 0.778 |

```
Umbral bajo ──────────────────────────────► Umbral alto
Marca casi todo como positivo               Solo marca positivo si está muy seguro
   recall ↑   precision ↓                       recall ↓   precision ↑
```

Cómo elegir el umbral según el costo del error:

| Si lo costoso es… | Priorizar | Ejemplo |
|---|---|---|
| Dejar pasar un positivo (falso negativo) | **Recall** (umbral bajo) | Detección de una enfermedad grave, fraude |
| Una falsa alarma (falso positivo) | **Precision** (umbral alto) | Filtro de spam, recomendaciones que molestan al usuario |
| Ambos por igual | **F1** | Clasificación general con clases desbalanceadas |

## 12.7 Train vs. test: detectar overfitting y underfitting

Calcular la métrica **en train y en test** y compararlas es la forma más directa de diagnosticar el modelo.

| Train | Test | Diagnóstico |
|---|---|---|
| Alta | Alta y parecida | Buen ajuste |
| Alta | Mucho menor | **Overfitting**: el modelo memoriza |
| Baja | Baja | **Underfitting**: el modelo es demasiado simple |

Ejemplo real con KNN (accuracy) sobre `Social_Network_Ads.csv`:

| `k` | Accuracy en train | Accuracy en test |
|---|---|---|
| 1 | 1.000 | 0.925 |
| 5 | 0.909 | 0.950 |
| 21 | 0.897 | 0.950 |

Con `k=1` el modelo acierta el 100 % de train porque memoriza cada punto, pero baja en test: overfitting. Con `k` mayor, el modelo se suaviza y el desempeño en train y test se acerca.

> Con pocos datos, la métrica de test puede salir incluso mejor que la de train por azar del split. Por ejemplo, en la regresión lineal con `TV`, el R² fue 0.591 en train y 0.677 en test, con solo 40 filas de test. Es una señal de que **un único split es poco fiable**, y de ahí la validación cruzada.

## 12.8 Validación cruzada

Un solo `train_test_split` depende de qué filas caen en test. La **validación cruzada de `k` particiones** entrena y evalúa `k` veces rotando la partición de test, y promedia:

```
Vuelta 1: [ TEST ][ train ][ train ]   → métrica 1
Vuelta 2: [ train ][ TEST ][ train ]   → métrica 2
Vuelta 3: [ train ][ train ][ TEST ]   → métrica 3
                                         └─► promedio ± desviación
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(modelo, X, Y, cv=5, scoring="r2")        # regresión
scores = cross_val_score(modelo, X, Y, cv=5, scoring="f1")        # clasificación
print(scores.mean(), scores.std())
```

El parámetro `scoring` acepta el nombre de cualquiera de las métricas de este capítulo (`"r2"`, `"neg_root_mean_squared_error"`, `"neg_mean_absolute_error"`, `"accuracy"`, `"precision"`, `"recall"`, `"f1"`, `"roc_auc"`). Las métricas de error llevan el prefijo `neg_` porque scikit-learn maximiza siempre el puntaje.

Cuándo usarla: con datasets pequeños, o cuando se quiere una estimación más estable que la de un solo split. Es también la base para comparar hiperparámetros sin tocar el conjunto de test.

### Detalles importantes

- **Se hace sobre train.** El conjunto de test queda intacto como comprobación final.
- **El preprocesamiento va dentro de la validación.** Si el modelo necesita escalado, se encapsula con un `Pipeline` para que el escalador se reajuste en cada partición. Escalar antes de validar filtra información de la partición de validación.
- **Clasificación: particiones estratificadas** (`StratifiedKFold`), que conservan la proporción de clases en cada partición.
- **Se reporta media y desviación.** Una desviación alta indica que el resultado depende mucho de qué datos caen en cada partición.
- **Se fija la semilla** para que el resultado sea reproducible.

### Leave-one-out (validación cruzada de uno fuera)

Es el caso extremo con `k = n`: cada fila se predice con un modelo entrenado con todas las demás. Sirve cuando el dataset es tan pequeño que separar un test dejaría muy pocas filas.

```
Fila 1: entrena con filas 2..n → predice la fila 1
Fila 2: entrena con filas 1,3..n → predice la fila 2
...
Fila n: entrena con filas 1..n−1 → predice la fila n
                    └─► métricas calculadas sobre las n predicciones
```

```python
from sklearn.model_selection import LeaveOneOut, cross_val_predict

y_pred = cross_val_predict(modelo, X, y, cv=LeaveOneOut())
r2 = r2_score(y, y_pred)          # las n predicciones son todas "fuera de muestra"
```

Con solo 10 filas (como en la regresión polinómica), la estimación sigue siendo ruidosa, pero es mucho más honesta que evaluar sobre los mismos datos con los que se entrenó.

## 12.9 Casos de uso: qué métrica elegir

| Problema | Tipo | Métrica principal | Por qué |
|---|---|---|---|
| Predecir el precio de una vivienda | Regresión | RMSE (y R²) | Los errores grandes son los más costosos, y se quiere el error en unidades de dinero |
| Predecir demanda con picos ocasionales | Regresión | MAE | Los picos atípicos no deben dominar la evaluación |
| Comparar tres modelos de regresión sobre el mismo dataset | Regresión | R² | No depende de las unidades |
| Predecir si un cliente compra (clases equilibradas) | Clasificación | Accuracy y F1 | Ambos errores pesan parecido |
| Detectar fraude (1 caso en 1000) | Clasificación desbalanceada | Recall, precision, F1 | El accuracy sería ~99.9 % sin detectar ningún fraude |
| Diagnóstico médico de una enfermedad grave | Clasificación | Recall | No dejar pasar casos reales pesa más que las falsas alarmas |
| Filtro de spam | Clasificación | Precision | Marcar un correo válido como spam es lo más costoso |
| Comparar clasificadores sin fijar aún el umbral | Clasificación | ROC-AUC | Evalúa el modelo en todos los umbrales |

## 12.10 Flujo de decisión

```
¿Qué se predice?
      │
      ├── Un número ───────────────► REGRESIÓN
      │                                 │
      │                                 ├─ ¿Comparar modelos / visión global? ──► R²
      │                                 ├─ ¿Error en unidades reales? ─────────► RMSE / MAE
      │                                 └─ ¿Hay valores atípicos? ─────────────► MAE (no RMSE)
      │
      └── Una categoría ───────────► CLASIFICACIÓN
                                        │
                                        └─ ¿Clases desbalanceadas?
                                              ├─ No ──► Accuracy + F1
                                              └─ Sí ──► NO usar solo accuracy
                                                          │
                                                          ├─ ¿Lo grave es no detectar? ──► Recall
                                                          ├─ ¿Lo grave es la falsa alarma? ► Precision
                                                          ├─ ¿Ambos? ───────────────────► F1
                                                          └─ ¿Aún sin umbral definido? ──► ROC-AUC
```

## 12.11 El proceso de evaluación en este proyecto

Cada endpoint combina tres lentes, que responden preguntas distintas:

```
Datos ──► split train/test ──► modelo (con su preprocesamiento, dentro de un Pipeline si aplica)
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
 Métricas sobre TEST             Métrica sobre TRAIN             VALIDACIÓN CRUZADA (sobre train)
 ¿Generaliza?                    ¿Memoriza?                      ¿Es estable?
        │                               │                               │
        └───────────────┬───────────────┴───────────────┬───────────────┘
                        ▼                               ▼
              test frente a train              test frente a validación cruzada
              (brecha grande = overfitting)    (test mucho mejor = split afortunado)
```

### Qué devuelve cada endpoint

| Endpoint | Sobre test | Sobre train | Validación cruzada |
|---|---|---|---|
| Regresión lineal simple y múltiple | `r2_score`, `rmse` | `r2_train` | `cv_r2_*`, `cv_rmse_*` |
| SVR | `r2_score`, `rmse` | `r2_train` | `cv_r2_*`, `cv_rmse_*` (con el escalado dentro de cada partición) |
| Housing (lineal, árbol, forest) | `r2_score` y `rmse` en cada paso de columnas | `r2_train` en cada paso | `cross_validation` sobre el modelo con todas las columnas |
| Regresión polinómica | — (sin split) | `r2_linear`, `r2_polynomial` (ajuste) | *Leave-one-out*: `r2_*_cv`, `rmse_*_cv` |
| Regresión logística y KNN | matriz de confusión, `precision`, `recall`, `f1_score`, `roc_auc` | `f1_train` | `cv_f1_*`, `cv_roc_auc_*` |
| MNIST | `test_precision`, `test_recall`, `test_f1_score`, `test_roc_auc` | — | `cross_val_accuracy`, y matriz de confusión, `precision`, `recall`, `f1_score` de las predicciones de la validación cruzada |

`*` = `_mean` y `_std`.

### Cómo leer los resultados: ejemplos reales

| Caso | Test | Train | Validación cruzada | Lectura |
|---|---|---|---|---|
| Árbol de decisión sin límite (R²) | 0.627 | **1.000** | 0.635 ± 0.019 | Memoriza el train por completo: brecha de 0.37. El test y la validación cruzada coinciden, así que el resultado real ronda 0.63 |
| Random forest, 50 árboles (R²) | 0.815 | 0.974 | 0.809 ± 0.006 | Brecha moderada, pero resultado muy estable (desviación 0.006) |
| KNN con `k=1` (F1) | 0.870 | **1.000** | 0.808 ± 0.035 | Cada punto es su propio vecino: el train es perfecto y el modelo no generaliza |
| KNN con `k=5` (F1) | 0.913 | 0.883 | 0.860 ± 0.048 | Buen equilibrio; el test es algo optimista frente a la validación cruzada |
| Regresión logística (F1) | 0.829 | 0.748 | 0.741 ± 0.091 | El test salió mejor que el train y que la validación cruzada: el split fue favorable, y la desviación alta confirma la inestabilidad |
| Regresión lineal con `TV` (R²) | 0.677 | 0.591 | 0.514 ± 0.208 | Con solo 40 filas de test y una desviación de 0.21, un solo split sobrestima el modelo |

Para la regresión polinómica, el R² de ajuste y el de *leave-one-out* cuentan historias opuestas:

| `degree` | R² de ajuste | R² *leave-one-out* | RMSE *leave-one-out* |
|---|---|---|---|
| 1 (lineal) | 0.707 | 0.408 | 10 804 |
| 4 | 0.993 | **0.869** | 5 079 |
| 9 | 1.000 | **−15.19** | 56 515 |

El polinomio de grado 9 pasa por todos los puntos (ajuste perfecto) pero predice muy mal cada punto que no vio (R² negativo): overfitting severo que el R² de ajuste, por sí solo, ocultaba.

> Random forest con 50 árboles tarda ~40 s en la petición completa: la validación cruzada del modelo con todas las columnas suma cinco entrenamientos más. Por eso se ejecuta solo sobre ese modelo y no en cada paso de columnas.

## 12.12 Errores comunes

- **Reportar la métrica de train como resultado.** Mide memoria, no generalización.
- **Usar solo accuracy con clases desbalanceadas.** Un modelo que nunca predice la clase minoritaria puede tener un accuracy altísimo.
- **Mezclar métricas de otro tipo de problema.** No existe "precision" para una regresión, ni R² para una clasificación.
- **Elegir hiperparámetros mirando el test.** Cada vez que se ajusta `degree`, `max_depth` o `k` según la métrica de test, ese conjunto deja de ser imparcial. Lo correcto es un conjunto de validación aparte o validación cruzada (ver `GridSearchCV` en [10](10-hoja-de-ruta.md)).
- **Confiar en un solo split.** Con pocos datos, el resultado cambia según `random_state`.
- **Comparar R² de datasets distintos.** Un R² alto en un dataset fácil no significa un mejor modelo que un R² menor en uno difícil.
- **Ignorar el umbral.** El 0.5 por defecto no es necesariamente el mejor para el problema.

## 12.13 Para seguir practicando

- Calcula R², RMSE y MAE de una regresión lineal sobre `Advertising.csv` para `TV`, `Radio` y `Newspaper`, y compara RMSE contra MAE.
- Calcula `roc_auc_score` (con `predict_proba`) para una regresión logística y para KNN sobre `Social_Network_Ads.csv`, y compara ambos modelos.
- Calcula precision y recall de la regresión logística con umbrales 0.2, 0.4, 0.6 y 0.8, y grafica la curva precision–recall con `precision_recall_curve`.
- Calcula la métrica de train y la de test de un árbol de decisión sobre `housing.csv` y observa cómo crece la brecha al quitar el límite de `max_depth`.
- Sustituye un `train_test_split` por `cross_val_score(cv=5)` y compara la media con el resultado de un solo split.
