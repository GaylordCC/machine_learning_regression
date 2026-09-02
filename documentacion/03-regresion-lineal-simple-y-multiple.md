# 3. Regresión lineal simple y múltiple

> Requiere: [02-fundamentos-de-machine-learning.md](02-fundamentos-de-machine-learning.md)

## 3.1 Dataset usado: `Advertising.csv`

Columnas: `TV`, `Radio`, `Newspaper` (inversión publicitaria en cada medio, en miles de USD) y `Sales` (ventas resultantes). Es el dataset clásico del libro *An Introduction to Statistical Learning*.

## 3.2 Teoría: regresión lineal simple

Busca la mejor **línea recta** que relaciona una variable `X` con `Y`:

```
Y = b0 + b1 · X
```

- `b0` (intercepto): valor de `Y` cuando `X = 0`.
- `b1` (pendiente): cuánto cambia `Y` por cada unidad que aumenta `X`.

El algoritmo encuentra `b0` y `b1` minimizando el **error cuadrático** entre las predicciones y los valores reales — esto se llama **Mínimos Cuadrados Ordinarios (OLS)**. Geométricamente: de todas las rectas posibles, elige la que hace mínima la suma de las distancias verticales al cuadrado entre cada punto real y la recta.

## 3.3 Código en este proyecto: `regression_linear_model`

📍 `machine_learning/services/regression/linear_regression_service.py` · Endpoint: `POST /v1/linear-regression`

```python
X = data[request.column_name].values.reshape(-1,1)   # una sola columna: TV, Radio o Newspaper
Y = data['Sales'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)          # aquí se calculan b0 y b1
y_predict = lin_reg.predict(X_test)

rmse = mean_squared_error(Y_test, y_predict, squared=False)
r2 = r2_score(Y_test, y_predict)
```

Puntos a entender:

- `request.column_name` viene del `RegressionSchema` (`schemas.py:9`) — el `Enum MediaType` obliga a que solo puedas elegir `TV`, `Radio` o `Newspaper`, evitando que llegue un nombre de columna inválido al `.py` (validación en la frontera, buena práctica).
- `.reshape(-1,1)`: scikit-learn espera que `X` sea una **matriz 2D** (filas = ejemplos, columnas = features), incluso si solo hay una feature. `Y` en cambio puede ser un vector 1D. Este `reshape` es la forma estándar de convertir una `Series` de pandas (1D) en una matriz de una sola columna.
- El gráfico generado (nombre en la respuesta, key `plot_file`, ej. `plotregression_TV_20260901_a1b2c3d4.png` dentro de `results_graphics/` — cada request genera un archivo propio, no se pisa entre llamadas) dibuja los puntos reales (`Y_test` vs `X_test`) y encima la recta de predicción — así se ve visualmente qué tan bien ajusta el modelo.
- La respuesta del endpoint es un JSON con `predictions`, `rmse` y `r2_score` — puedes ver las métricas directamente en Swagger, sin depender de los `print()` en la consola del servidor.

**Ejercicio para practicar**: corre el endpoint tres veces, una por cada columna (`TV`, `Radio`, `Newspaper`), y compara el `R²` impreso en consola. Vas a notar que `TV` explica las ventas mucho mejor que `Newspaper` — eso ya es una conclusión de negocio real ("la publicidad en periódico casi no impacta ventas").

## 3.4 Teoría: regresión lineal múltiple

Generaliza la fórmula a varias variables:

```
Y = b0 + b1·X1 + b2·X2 + ... + bn·Xn
```

Cada coeficiente `bi` representa cuánto cambia `Y` por cada unidad de `Xi`, **manteniendo las demás variables constantes**. Esto es clave: en regresión simple, el coeficiente de `TV` mide el efecto de `TV` ignorando todo lo demás; en regresión múltiple, mide el efecto de `TV` *aislando* el efecto de `Radio`.

## 3.5 Código en este proyecto: `regression_multi_linear_model`

📍 `machine_learning/services/regression/linear_regression_service.py` · Endpoint: `POST /v1/multi-linear-regression`

```python
X = data.drop(['Newspaper', 'Sales'], axis=1).values   # queda TV + Radio
Y = data['Sales'].values
```

Aquí se descarta `Newspaper` (justo la variable que en el punto anterior vimos que aportaba poco) y se usan `TV` y `Radio` como las dos features. El resto del pipeline es idéntico al de regresión simple: split, `fit`, `predict`, `RMSE`, `R²`.

La visualización cambia porque ahora hay más de una dimensión de entrada (no se puede graficar `X` vs `Y` en un plano simple): se usa `sns.regplot(x=Y_test, y=y_predict)`, que grafica **valores reales vs. valores predichos** — si el modelo fuera perfecto, todos los puntos caerían sobre la diagonal.

**Pregunta para reflexionar**: si comparas el `R²` de este modelo (TV+Radio) contra el `R²` de la regresión simple con solo `TV`, ¿mejora? Ese es el efecto esperado de agregar una variable relevante (`Radio`) al modelo — normalmente el `R²` sube cuando incorporas features que sí aportan información.

## 3.6 Errores comunes a vigilar

- **Multicolinealidad**: cuando dos variables independientes están muy correlacionadas entre sí (no es el caso obvio aquí, pero sí pasa en `housing.csv`, ver [06](06-arboles-de-decision-y-random-forest.md)), los coeficientes se vuelven inestables y difíciles de interpretar.
- **Extrapolación**: el modelo solo es confiable dentro del rango de datos que vio en train. Si entrenas con inversión en TV entre $5K-$300K, predecir con $10M de inversión no tiene sentido estadístico aunque el modelo "responda" un número.
- **Asumir causalidad**: un `R²` alto indica correlación/ajuste, no que `TV` **cause** las ventas — en este dataset sí hay una relación causal razonable (publicidad → ventas), pero no siempre es así.

## 3.7 Para seguir practicando

- Imprime `lin_reg.coef_` y `lin_reg.intercept_` después de entrenar y escribe en una frase qué significa cada coeficiente en términos de negocio ("por cada $1000 adicional en TV, las ventas suben en promedio X unidades").
- Implementa **regresión Ridge/Lasso** (regularización) sobre este mismo dataset y compara los coeficientes — te prepara para el tema de regularización en la [hoja de ruta](10-hoja-de-ruta.md).
