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

📍 `machine_learning/services/classification/logistic_regression_service.py` · Endpoint: `POST /v1/logistic-regression-classification`

La preparación de datos (encoding de `Gender` y división train/test) está factorizada en `services/shared/social_ads_preprocessing.py::split_train_test()`, porque **KNN usa exactamente la misma preparación** (§7.4). El escalado se encapsula junto con el modelo en un `Pipeline`.

### Preparación de datos

Resumen de lo que hace `split_train_test()`:

```python
X = data.iloc[:, [2, 3]]        # columnas Age, EstimatedSalary
Y = data.iloc[:, -1].values      # columna Purchased

encoder = OneHotEncoder()
gender_encoded = encoder.fit_transform(data[["Gender"]])
encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=encoder.get_feature_names_out())

X = pd.concat([X, encoded_df], axis=1)   # Age + EstimatedSalary + Gender codificado
```

Igual que en `housing.csv` ([06](06-arboles-de-decision-y-random-forest.md)), `Gender` es categórica sin orden → `OneHotEncoder` es la elección correcta.

### Escalado sin fuga de datos

```python
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
model.fit(X_train, Y_train)       # el escalador aprende media y desviación SOLO de train
y_pred = model.predict(X_test)    # aplica esas mismas estadísticas a test, sin reajustar
```

**Error frecuente que se evita aquí**: hacer `sc_X.fit_transform(X_test)` reajusta el escalador con datos de test en vez de reutilizar la media/desviación aprendidas en train. Así `X_train` y `X_test` quedan escalados con "reglas" distintas — como medir algo con dos reglas calibradas diferente. Con el `Pipeline`, el escalador se ajusta solo con train y el error es estructuralmente imposible de cometer. `split_train_test()` devuelve los datos **sin escalar** precisamente para que el `Pipeline` decida cuándo ajustar: además, en la validación cruzada reajusta el escalador dentro de cada partición. KNN usa la misma construcción.

**Por qué la regresión logística sí necesita escalado** (a diferencia de un árbol de decisión): el algoritmo de optimización que ajusta `b0, b1, b2...` (por defecto, en scikit-learn, una variante de descenso de gradiente/`lbfgs`) converge mejor y más rápido cuando las variables están en escalas comparables. Sin escalar, `EstimatedSalary` (decenas de miles) dominaría numéricamente sobre `Age` (decenas) aunque ambas sean igual de relevantes.

### Entrenamiento y evaluación

```python
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
return evaluate_binary_classifier(model, X_train, Y_train, X_test, Y_test)
```

`evaluate_binary_classifier` (`services/shared/evaluation.py`) entrena el modelo y devuelve, como JSON estructurado:

| Campo | Qué es |
|---|---|
| `confusion_matrix`, `precision`, `recall`, `f1_score` | Métricas sobre el conjunto de test |
| `roc_auc` | Calidad de la separación de clases, sin depender del umbral (usa `predict_proba`) |
| `f1_train` | F1 sobre train, para compararlo con el de test y detectar overfitting |
| `cv_f1_mean`, `cv_f1_std`, `cv_roc_auc_mean`, `cv_roc_auc_std` | Validación cruzada de 5 particiones sobre train: media y desviación |

Con los datos del proyecto, la regresión logística da `f1_score` 0.829 en test, `f1_train` 0.748 y `cv_f1_mean` 0.741 ± 0.091: el split de test fue favorable, y la desviación alta avisa de que el resultado de un solo split es poco estable. El detalle de cada métrica está en [12](12-metricas-de-evaluacion.md).

## 7.4 Teoría: K-Nearest Neighbors (KNN)

📍 `machine_learning/services/classification/knn_service.py` · Endpoint: `POST /v1/knn-classification` · body opcional: `{"n_neighbors": 5}`

KNN tiene su propia ruta (`/knn-classification`) y el método entrena un `KNeighborsClassifier` sobre el mismo pipeline de datos que la regresión logística.

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
    X_train, X_test, Y_train, Y_test = split_train_test(random_state=0)  # misma preparación que la regresión logística

    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=request.n_neighbors))
    metrics = evaluate_binary_classifier(model, X_train, Y_train, X_test, Y_test)
    return {"n_neighbors": request.n_neighbors, **metrics}
```

La respuesta incluye los mismos campos que la regresión logística (ver §7.3). Con `k=1`, el `f1_train` es 1.000 (cada punto es su propio vecino) frente a 0.870 en test y 0.808 en validación cruzada: el efecto de memorizar. Con `k=5`, esas cifras son 0.883 en train, 0.913 en test y 0.860 en validación cruzada.

`n_neighbors` (el hiperparámetro `k`) viene del body (`KnnClassificationSchema`, default `5`, validado entre 1 y 50) — pruébalo directamente:

```bash
curl -X POST http://localhost:8080/v1/knn-classification -H "Content-Type: application/json" -d '{"n_neighbors": 1}'
curl -X POST http://localhost:8080/v1/knn-classification -H "Content-Type: application/json" -d '{"n_neighbors": 21}'
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
- Con el mismo dataset, compara Regresión Logística vs. KNN vs. un Árbol de Decisión de clasificación (`DecisionTreeClassifier`, que aún no está en el proyecto) usando F1-score — buen ejercicio de comparación de modelos. `split_train_test()` ya te da los datos listos para reutilizar en un tercer servicio, y `evaluate_binary_classifier()` calcula todas las métricas.
- Agrega `weights='distance'` a `KNeighborsClassifier` (en vez del default `'uniform'`) — hace que vecinos más cercanos pesen más en la votación — y compara el resultado.
