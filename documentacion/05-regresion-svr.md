# 5. Support Vector Regression (SVR)

> Requiere: [03](03-regresion-lineal-simple-y-multiple.md), conviene haber leído sobre kernels si vienes de cero (se explica abajo).

## 5.1 Dataset usado

De nuevo `Advertising.csv`, pero esta vez usando **TV y Radio** como features (se descarta `Newspaper` igual que en la regresión múltiple) para predecir `Sales`.

## 5.2 Teoría: de SVM a SVR

**SVM (Support Vector Machine)** nació para clasificación: busca el "límite" (hiperplano) que separa dos clases dejando el mayor margen posible respecto a los puntos más cercanos de cada clase (esos puntos cercanos son los "vectores de soporte", de ahí el nombre).

**SVR (Support Vector Regression)** adapta esa misma idea a regresión, con un giro: en vez de buscar un límite que separe clases, busca una **franja (tubo)** alrededor de la función de predicción, de ancho `epsilon`, y:

- Los puntos que caen **dentro** del tubo no penalizan al modelo (se consideran "suficientemente bien predichos").
- Solo los puntos que quedan **fuera** del tubo generan error y afectan el ajuste.

Esto lo hace distinto a la regresión lineal (que minimiza el error de *todos* los puntos): SVR es más tolerante a pequeñas desviaciones y se enfoca en corregir los errores grandes.

### El kernel

`SVR(kernel='rbf')` — el **kernel** define qué tipo de "forma" puede tomar la función de predicción:

| Kernel | Forma que puede aprender |
|---|---|
| `linear` | Una recta/plano (equivalente a una regresión lineal con margen) |
| `poly` | Una curva polinómica |
| `rbf` (Radial Basis Function, el usado aquí) | Formas muy flexibles/no lineales — es el kernel por defecto y el más usado cuando no sabes de antemano la forma de la relación |

El kernel `rbf` funciona midiendo qué tan "cerca" está un punto nuevo de los puntos de entrenamiento (usando una función de distancia gaussiana) — por eso es sensible a la escala de las variables (ver §5.4).

## 5.3 Código en este proyecto

📍 `machine_learning/services/regression/svr_service.py` · Endpoint: `POST /svr-regression` · body opcional: `{"kernel": "rbf" | "linear" | "poly"}`

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# SVR es sensible a la escala: se escala X (fit solo en train) y tambien Y.
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)          # sin fit, reutilizando lo aprendido en train

sc_Y = StandardScaler()
Y_train_scaled = sc_Y.fit_transform(Y_train).ravel()

svr = SVR(kernel=request.kernel.value)
svr.fit(X_train_scaled, Y_train_scaled)

y_predict_scaled = svr.predict(X_test_scaled)
y_predict = sc_Y.inverse_transform(y_predict_scaled.reshape(-1, 1)).ravel()  # de vuelta a unidades reales
```

## 5.4 ✅ Hallazgo corregido: escalado de variables

**Este era el bug documentado originalmente**: la primera versión de este servicio no escalaba `X` ni `Y` antes de entrenar el `SVR`. A diferencia de `LinearRegression` (indiferente a la escala), **SVR con kernel `rbf`/`poly` es sensible a la escala** de las variables, porque el kernel calcula distancias entre puntos — si `TV` va de 0 a 300 y `Radio` va de 0 a 50, `TV` dominaría el cálculo de distancia solo por tener números más grandes, no porque sea más importante.

**Impacto medido**: en pruebas locales, el `R²` pasó de **~0.60 (sin escalar)** a **~0.98 (escalado)** sobre los mismos datos — la diferencia es enorme y es la evidencia más directa, dentro de este proyecto, de por qué el escalado importa en modelos basados en distancia (compáralo con la teoría de KNN en [07](07-regresion-logistica-y-knn.md), que es aún más sensible a esto).

Nota que además de escalar `X`, aquí también se escala `Y` (algo que no hace falta en regresión lineal ni en árboles) y luego se **des-escala** (`sc_Y.inverse_transform(...)`) el resultado, para que las predicciones vuelvan a estar en unidades de "ventas reales" y no en unidades estandarizadas.

## 5.5 El kernel ahora es configurable

El endpoint acepta `kernel` en el body (`linear`, `poly` o `rbf`, default `rbf`) vía `SvrRegressionSchema` (`machine_learning/schemas.py`). Pruébalo:

```bash
curl -X POST http://localhost:8080/svr-regression -H "Content-Type: application/json" -d '{"kernel": "linear"}'
curl -X POST http://localhost:8080/svr-regression -H "Content-Type: application/json" -d '{"kernel": "rbf"}'
```

Compara el `r2_score` de ambas respuestas: ¿la relación entre publicidad y ventas es realmente no lineal, o un kernel lineal ya es suficiente?

## 5.6 Para seguir practicando

- Ya puedes comparar los tres kernels (`linear`, `poly`, `rbf`) directamente desde Swagger — anota cuál da mejor `r2_score` en este dataset.
- Investiga el hiperparámetro `C` de `SVR` (controla qué tan estricta es la penalización de puntos fuera del tubo) y `epsilon` (el ancho del tubo) — son los dos hiperparámetros más importantes a ajustar en la práctica. Podrías agregarlos como campos opcionales a `SvrRegressionSchema`.
