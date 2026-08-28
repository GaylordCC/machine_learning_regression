# 4. Regresión polinómica

> Requiere: [03-regresion-lineal-simple-y-multiple.md](03-regresion-lineal-simple-y-multiple.md)

## 4.1 Dataset usado

En `polynomical_regression()` los datos **no vienen de un CSV**: se construyen a mano en `build_dataset()` (`machine_learning/services/regression/polynomial_regression_service.py`), un dataset clásico de cursos de ML — 10 posiciones de una empresa de desarrollo (Pasante → CEO) con sus años de experiencia y salario:

```python
post   = ["Pasante de Desarrollo", ..., "Director Ejecutivo (CEO)"]
salary = [1200, 2500, 4000, 4800, 6500, 9000, 12850, 15000, 25000, 50000]
```

Es un dataset intencionalmente pequeño y **no lineal**: el salario no crece de forma constante entre niveles — crece cada vez más rápido al subir de nivel (una curva, no una recta).

## 4.2 Teoría: ¿por qué "polinómica" y para qué sirve?

Cuando la relación entre `X` e `Y` no es una línea recta sino una curva, forzar una regresión lineal simple da un mal ajuste (underfitting). La regresión polinómica sigue siendo, técnicamente, un **modelo lineal** — pero lineal en los *coeficientes*, no en `X`: se agregan potencias de `X` como nuevas features.

```
Y = b0 + b1·X + b2·X² + b3·X³ + ... + bn·Xⁿ
```

El truco es simple: generas nuevas columnas (`X²`, `X³`, `X⁴`...) y luego aplicas una regresión lineal normal sobre esas columnas ampliadas. Por eso en el código ves `LinearRegression()` otra vez, solo que entrenado sobre `X_poly` en lugar de `X`.

## 4.3 Código en este proyecto

📍 `machine_learning/services/regression/polynomial_regression_service.py` · Endpoint: `POST /polynomial-regression`

```python
X = data["years"].values.reshape(-1, 1)
Y = data["salary"].values

linear_model = LinearRegression()
linear_model.fit(X, Y)

poly = PolynomialFeatures(degree=request.degree)   # degree viene del body del request
X_poly = poly.fit_transform(X)                       # genera [1, X, X², X³, ..., X^degree]

poly_model = LinearRegression()
poly_model.fit(X_poly, Y)
```

Puntos clave:

- `PolynomialFeatures(degree=...)` transforma cada valor de `X` en un vector `[1, X, X², ..., X^degree]`. Es un **transformador de features**, igual en espíritu al `OneHotEncoder`/`StandardScaler` que viste en fundamentos — no entrena nada, solo reestructura los datos.
- A diferencia de la versión original del código, `degree` ya **no está fijo en 4** — es un parámetro del request (`PolynomialRegressionSchema`, `machine_learning/schemas.py`), validado entre 1 y 10. Puedes mandarlo desde Swagger sin tocar código.
- El endpoint entrena y grafica **ambos modelos** (lineal y polinómico) y devuelve `r2_linear` y `r2_polynomial` en la respuesta — puedes comparar numéricamente los dos sin mirar la consola.

## 4.4 El hiperparámetro más importante: `degree`

`degree` es un **hiperparámetro** (tú lo eliges, el modelo no lo aprende — repaso de [02](02-fundamentos-de-machine-learning.md)). Es el ejemplo más claro de underfitting vs. overfitting de todo el proyecto:

| `degree` | Comportamiento |
|---|---|
| `degree=1` | Es una regresión lineal simple — probablemente **underfitting** para este dataset curvo |
| `degree=4` (valor por defecto) | Sigue razonablemente bien la curva de salarios |
| `degree=9` o `10` (extremo) | Con solo 10 puntos de datos, un polinomio de grado 9 puede pasar **exactamente** por cada punto — ajuste perfecto en train, pero prediría valores absurdos entre los puntos conocidos: **overfitting** severo |

**Ejercicio recomendado**: llama a `POST /polynomial-regression` con `{"degree": 1}`, luego `{"degree": 4}`, luego `{"degree": 9}`, y compara `r2_polynomial` en cada respuesta además del gráfico generado en `results_graphics/polynomicalregression.png` (dibuja la curva lineal y la polinómica juntas). Vas a *ver* el overfitting con tus propios ojos — es la forma más rápida de que el concepto se quede grabado.

```bash
curl -X POST http://localhost:8080/polynomial-regression -H "Content-Type: application/json" -d '{"degree": 9}'
```

## 4.5 Por qué no hay train/test split aquí

Notarás que, a diferencia de los demás métodos de regresión, `polynomical_regression()` entrena con **todos** los datos (no hay `train_test_split`). Con solo 10 filas, separar un 20% para test dejaría 2 puntos de test — insuficiente para una evaluación confiable. Es una decisión razonable para un dataset de juguete como este, pero **no sería aceptable en un proyecto real**: siempre que tengas suficientes datos, evalúa con datos que el modelo no haya visto.

## 4.6 Para seguir practicando

- Ya puedes comparar `r2_linear` vs `r2_polynomial` directamente en la respuesta del endpoint — corre el experimento del §4.4 y anota en qué `degree` empieza a verse overfitting (pista: con solo 10 filas, pasado `degree=6-7` el `r2_polynomial` se acerca sospechosamente a 1.0).
- Agrega un endpoint (o parámetro) para predecir el salario de un valor específico de `years` no visto (ej. `years=2.5`) usando `poly.transform([[2.5]])` (sin `fit`, reutilizando el transformador ya ajustado) — practica la diferencia entre `fit_transform` y `transform` que se explicó en [02](02-fundamentos-de-machine-learning.md).
- Investiga `Pipeline` de scikit-learn (`sklearn.pipeline.Pipeline`) para encadenar `PolynomialFeatures` + `LinearRegression` en un solo objeto — es el patrón profesional que reemplaza hacer los pasos "a mano" como en este código (más en [10-hoja-de-ruta.md](10-hoja-de-ruta.md)).
