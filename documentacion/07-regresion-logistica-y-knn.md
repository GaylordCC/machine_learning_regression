# 7. Clasificación: Regresión Logística y K-Nearest Neighbors (KNN)

> Requiere: [02-fundamentos-de-machine-learning.md](02-fundamentos-de-machine-learning.md), especialmente la sección de métricas de clasificación.

## 7.1 Dataset usado: `Social_Network_Ads.csv`

Columnas: `Gender`, `Age`, `EstimatedSalary` y `Purchased` (0/1 — si la persona compró un producto tras ver un anuncio). Es un dataset de clasificación binaria clásico: predecir si alguien va a comprar en función de su edad, salario y género.

## 7.2 Teoría: Regresión Logística

A pesar del nombre, **no es un modelo de regresión, es de clasificación**. La confusión de nombre viene de que internamente usa una fórmula similar a la regresión lineal, pero el resultado se pasa por la **función sigmoide** para convertirlo en una probabilidad entre 0 y 1:

```
z = b0 + b1·X1 + b2·X2 + ...
p = 1 / (1 + e^(-z))        ← función sigmoide, siempre da un valor entre 0 y 1
```

Si `p > 0.5` (umbral por defecto), el modelo predice clase `1`; si no, clase `0`. La sigmoide es lo que convierte una recta (que puede dar cualquier valor de -∞ a +∞) en algo interpretable como "probabilidad de pertenecer a la clase positiva".

Geométricamente, el modelo aprende una **frontera de decisión** (una línea recta si hay 2 features, un plano si hay 3, un hiperplano en más dimensiones) que separa lo mejor posible los casos de clase 0 de los de clase 1.

## 7.3 Código en este proyecto: `handle_logistic_classification`

📍 `machine_learning/services/classification/logistic_regression_service.py` · Endpoint: `POST /logistic-regression-classification`

La preparación de datos (encoding de `Gender` + escalado) está factorizada en `services/shared/social_ads_preprocessing.py::prepare_train_test_split()`, porque **KNN usa exactamente el mismo pipeline** (§7.4).

### Preparación de datos

```python
X = data.iloc[:, [2, 3]]        # columnas Age, EstimatedSalary
Y = data.iloc[:, -1].values      # columna Purchased

encoder = OneHotEncoder()
gender_encoded = encoder.fit_transform(data[["Gender"]])
encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=encoder.get_feature_names_out())

X = pd.concat([X, encoded_df], axis=1)   # Age + EstimatedSalary + Gender codificado
```

Igual que en `housing.csv` ([06](06-arboles-de-decision-y-random-forest.md)), `Gender` es categórica sin orden → `OneHotEncoder` es la elección correcta.

### ✅ Escalado — hallazgo corregido

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # aprende Y transforma con train
X_test = scaler.transform(X_test)          # solo transforma, reutilizando lo aprendido en train
```

**Este era el bug documentado originalmente**: la primera versión de este código hacía `sc_X.fit_transform(X_test)` — reajustaba el escalador con datos de test en vez de reutilizar la media/desviación aprendidas en train. El resultado era que `X_train` y `X_test` quedaban escalados con "reglas" distintas — como si midieras algo con dos reglas calibradas diferente. Ya está corregido en `prepare_train_test_split()`, y como KNN reutiliza esa misma función, el fix aplica a ambos clasificadores a la vez.

**Por qué la regresión logística sí necesita escalado** (a diferencia de un árbol de decisión): el algoritmo de optimización que ajusta `b0, b1, b2...` (por defecto, en scikit-learn, una variante de descenso de gradiente/`lbfgs`) converge mejor y más rápido cuando las variables están en escalas comparables. Sin escalar, `EstimatedSalary` (decenas de miles) dominaría numéricamente sobre `Age` (decenas) aunque ambas sean igual de relevantes.

### Entrenamiento y evaluación

```python
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, Y_train)
y_pred = log_reg.predict(X_test)

return {
    "confusion_matrix": confusion_matrix(Y_test, y_pred).tolist(),
    "precision": precision_score(Y_test, y_pred),
    "recall": recall_score(Y_test, y_pred),
    "f1_score": f1_score(Y_test, y_pred),
}
```

La respuesta ya llega como JSON estructurado (antes solo se imprimía en consola y el endpoint devolvía un string genérico).

## 7.4 Teoría: K-Nearest Neighbors (KNN)

📍 `machine_learning/services/classification/knn_service.py` · Endpoint: `POST /knn-classification` · body opcional: `{"n_neighbors": 5}`

✅ **Ya implementado** — originalmente `handle_knn_classification` era un *stub* (`return "successfully knn classification"`) y además compartía ruta con la regresión logística, por lo que era inalcanzable por HTTP aunque hubiera tenido lógica real (ver el bug documentado en [01](01-arquitectura-del-proyecto.md#15-hallazgos-corregidos-en-esta-refactorización)). Ambos problemas están corregidos: la ruta es `/knn-classification` y el método entrena un `KNeighborsClassifier` de verdad.

**Idea central**: KNN es de los algoritmos más simples de ML — no "aprende" una fórmula ni una frontera durante el entrenamiento (de hecho, `.fit()` en KNN básicamente solo *memoriza* los datos de entrenamiento). Para clasificar un punto nuevo:

1. Calcula la distancia (normalmente euclidiana) del punto nuevo a **todos** los puntos de entrenamiento.
2. Toma los `k` vecinos más cercanos (`k` es el hiperparámetro — tú lo eliges, ej. `k=5`).
3. La clase predicha es la que tiene **mayoría de votos** entre esos `k` vecinos.

```
k=5, y de los 5 vecinos más cercanos: 3 son "Purchased=1", 2 son "Purchased=0"
→ predicción: 1 (gana la mayoría)
```

**Por qué el escalado es aún MÁS crítico en KNN que en regresión logística**: KNN se basa 100% en distancias entre puntos. Si `EstimatedSalary` va de 0 a 150,000 y `Age` de 18 a 60, la distancia euclidiana estará dominada casi por completo por `EstimatedSalary` — `Age` prácticamente no tendría influencia en qué vecinos se consideran "cercanos". **Nunca entrenes KNN sin escalar antes.**

**Elegir `k`**: valores pequeños (`k=1`) hacen al modelo muy sensible a ruido/outliers (overfitting); valores grandes suavizan demasiado la frontera de decisión (underfitting) y son más costosos de calcular. Es típico probar varios valores de `k` con validación cruzada y quedarse con el mejor.

### El código real

```python
def handle_knn_classification(self, request: KnnClassificationSchema):
    X_train, X_test, Y_train, Y_test = prepare_train_test_split(random_state=0)  # mismo pipeline que logistic regression

    knn = KNeighborsClassifier(n_neighbors=request.n_neighbors)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)

    return {
        "n_neighbors": request.n_neighbors,
        "confusion_matrix": confusion_matrix(Y_test, y_pred).tolist(),
        "precision": precision_score(Y_test, y_pred),
        "recall": recall_score(Y_test, y_pred),
        "f1_score": f1_score(Y_test, y_pred),
    }
```

`n_neighbors` (el hiperparámetro `k`) viene del body (`KnnClassificationSchema`, default `5`, validado entre 1 y 50) — pruébalo directamente:

```bash
curl -X POST http://localhost:8080/knn-classification -H "Content-Type: application/json" -d '{"n_neighbors": 1}'
curl -X POST http://localhost:8080/knn-classification -H "Content-Type: application/json" -d '{"n_neighbors": 21}'
```

Compara `f1_score` entre ambas respuestas para ver el efecto de `k` muy pequeño vs. muy grande.

## 7.5 Regresión Logística vs. KNN — cuándo usar cada uno

| | Regresión Logística | KNN |
|---|---|---|
| ¿Qué aprende? | Una frontera de decisión explícita (coeficientes) | Nada explícito — memoriza los datos |
| Interpretabilidad | Alta (puedes leer los coeficientes) | Baja |
| Velocidad de predicción | Rápida (es solo una fórmula) | Lenta si hay muchos datos (compara contra todos) |
| Sensible a escala | Sí | Sí, mucho más |
| Funciona bien con fronteras no lineales | No directamente (es lineal) | Sí, naturalmente |

## 7.6 Para seguir practicando

- Llama a `/knn-classification` con varios valores de `k` (1, 3, 5, 11, 21, 35) y anota `f1_score` en cada uno — grafica `f1_score` vs. `k` para visualizar el trade-off underfitting/overfitting con datos reales de tu propio proyecto.
- Con el mismo dataset, compara Regresión Logística vs. KNN vs. un Árbol de Decisión de clasificación (`DecisionTreeClassifier`, que aún no está en el proyecto) usando F1-score — buen ejercicio de comparación de modelos. `prepare_train_test_split()` ya te da los datos listos para reutilizar en un tercer servicio.
- Agrega `weights='distance'` a `KNeighborsClassifier` (en vez del default `'uniform'`) — hace que vecinos más cercanos pesen más en la votación — y compara el resultado.
