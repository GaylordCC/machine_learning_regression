# 8. Clasificación de imágenes (MNIST): validación cruzada y métricas en profundidad

> Requiere: [02](02-fundamentos-de-machine-learning.md) y [07](07-regresion-logistica-y-knn.md) (matriz de confusión).

Este capítulo cubre `handle_classification_image` (📍 `machine_learning/services/classification/image_classification_service.py` · Endpoint: `POST /v1/classification-algorithm`) — el ejercicio más avanzado del proyecto en cuanto a **evaluación rigurosa de un modelo**, aunque el modelo en sí (SGDClassifier) es relativamente simple. Está inspirado directamente en el capítulo 3 de *Hands-On Machine Learning* (Aurélien Géron), el mismo origen que el pipeline de `housing.csv`.

La respuesta del endpoint ya llega como JSON estructurado (`cross_val_accuracy`, `confusion_matrix`, `precision`, `recall`, `f1_score`) en vez de solo imprimirse en consola.

## 8.1 Dataset: MNIST

`fetch_openml('mnist_784', version=1)` descarga el dataset **MNIST**: 70,000 imágenes de dígitos escritos a mano (0-9), cada una de 28×28 píxeles (784 valores de gris, de ahí "784" en el nombre). Es *el* dataset de referencia histórico para clasificación de imágenes — casi todo curso de ML lo usa como primer contacto con visión por computadora.

```python
mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist['data'], mnist['target']   # X: 70000 filas × 784 columnas (píxeles), Y: el dígito (0-9) como texto
```

> Nota práctica: esta descarga requiere conexión a internet la primera vez (se cachea localmente después) y puede tardar. Si vas a experimentar mucho con este endpoint, considera descargar el dataset una vez y guardarlo en `sample_data/` para no depender de la red en cada corrida.

## 8.2 De clasificación multiclase a clasificación binaria

MNIST tiene 10 clases posibles (dígitos 0-9), pero este ejercicio simplifica el problema a uno **binario**: "¿es un 5, o no lo es?"

```python
Y = Y.astype(np.uint8)
X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
Y_train_5 = (Y_train == 5)   # True/False
Y_test_5 = (Y_test == 5)
```

Es una técnica de aprendizaje muy común: empezar con la versión más simple del problema (binaria) antes de pasar a la versión completa (multiclase) — los mismos conceptos de evaluación aplican, pero son más fáciles de entender con solo dos clases.

También nota el split: en vez de `train_test_split`, se usa un **slicing directo** (`X[:60000]` / `X[60000:]`) porque MNIST ya viene ordenado de forma que las primeras 60,000 imágenes son la partición de entrenamiento estándar y las últimas 10,000 son la de test — es la convención histórica del dataset, así que no hace falta aleatorizar.

## 8.3 El modelo: `SGDClassifier`

```python
sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, Y_train_5)
```

**SGD** = *Stochastic Gradient Descent* (Descenso de Gradiente Estocástico). No es un algoritmo de clasificación en sí mismo — es un **método de optimización**: ajusta los parámetros del modelo (por defecto, `SGDClassifier` entrena internamente algo equivalente a una SVM lineal o regresión logística, según el parámetro `loss`) de forma iterativa, viendo los ejemplos de entrenamiento **uno a la vez** (o en pequeños lotes) en vez de todos juntos. Esto lo hace mucho más rápido y escalable que otros métodos cuando hay muchos datos (como aquí: 60,000 imágenes × 784 píxeles).

Es un buen momento para notar algo que el código **no hace** pero sería buena práctica: escalar los píxeles (que van de 0 a 255) a un rango 0-1 (`X / 255.0`) antes de entrenar — SGD, como todo método basado en gradiente, converge mejor con features en escalas pequeñas y comparables.

## 8.4 Validación cruzada en profundidad

```python
cross_val_score(sgd_classifier, X_train, Y_train_5, cv=3, scoring='accuracy')
```

Ya se introdujo el concepto en [02-fundamentos-de-machine-learning.md](02-fundamentos-de-machine-learning.md). Aquí lo ves con `cv=3`: los 60,000 ejemplos de train se dividen en 3 partes iguales (*folds*). El proceso:

```
Vuelta 1: entrena con folds 2+3, evalúa con fold 1
Vuelta 2: entrena con folds 1+3, evalúa con fold 2
Vuelta 3: entrena con folds 1+2, evalúa con fold 3
```

`cross_val_score` devuelve un array de 3 valores de accuracy (uno por vuelta) — si los tres son similares y altos, hay más confianza en que el modelo generaliza bien, no que tuvo "suerte" con un split particular.

### ⚠️ Por qué accuracy es engañosa aquí — el ejemplo perfecto

Solo ~10% de las imágenes son "5" (aproximadamente 1 de cada 10 dígitos). Un clasificador que **siempre** responde "no es un 5", sin haber aprendido nada, tendría ~90% de accuracy. Por eso el código no se queda solo con `cross_val_score(..., scoring='accuracy')` — calcula también matriz de confusión, precision, recall y F1, que sí exponen si el modelo realmente está detectando los 5s o simplemente "apostando" a la clase mayoritaria.

**Ejercicio de verificación** (no está en el código, pero es muy revelador): compara el accuracy del `SGDClassifier` contra un "clasificador tonto" que siempre predice `False`:

```python
from sklearn.base import BaseEstimator

class NuncaEsCincoClassifier(BaseEstimator):
    def fit(self, X, y=None): return self
    def predict(self, X): return np.zeros((len(X),), dtype=bool)

print(cross_val_score(NuncaEsCincoClassifier(), X_train, Y_train_5, cv=3, scoring='accuracy'))
# Vas a ver ~90% de accuracy, ¡sin que el modelo haya aprendido absolutamente nada!
```

## 8.5 `cross_val_predict` + matriz de confusión

```python
Y_train_predict = cross_val_predict(sgd_classifier, X_train, Y_train_5, cv=3)
print(confusion_matrix(Y_train_5, Y_train_predict))
```

Diferencia clave con `cross_val_score`: en vez de devolver una métrica resumida por fold, `cross_val_predict` devuelve **la predicción real para cada ejemplo de entrenamiento** (cada ejemplo fue predicho por el fold que *no* lo usó para entrenar, así que sigue siendo una evaluación "honesta", no memorizada). Esto te permite construir la matriz de confusión completa sobre todo el conjunto de train.

```
                  Predijo "no 5"    Predijo "sí 5"
Real "no 5"            TN                FP
Real "sí 5"            FN                TP
```

Con estos 4 números:

```python
precision_score(Y_train_5, Y_train_predict)   # TP / (TP + FP) → de lo que dijo "es un 5", ¿cuánto acertó?
recall_score(Y_train_5, Y_train_predict)       # TP / (TP + FN) → de todos los 5 reales, ¿cuántos detectó?
f1_score(Y_train_5, Y_train_predict)            # balance entre ambas
```

## 8.6 El trade-off Precision vs. Recall

Es uno de los conceptos más importantes de clasificación y no está explícito en el código, pero vale la pena entenderlo aquí:

- **Priorizar precision** (pocos falsos positivos): útil cuando una falsa alarma es costosa. Ej: un filtro de spam agresivo que casi nunca marca un correo válido como spam, aunque deje pasar algo de spam real.
- **Priorizar recall** (pocos falsos negativos): útil cuando dejar pasar un caso positivo es lo costoso. Ej: un test médico para una enfermedad grave — prefieres algunos falsos positivos (que luego se descartan con más pruebas) antes que dejar pasar un caso real.

`SGDClassifier` internamente calcula un **puntaje de decisión** (*decision score*) para cada ejemplo y lo compara contra un umbral (por defecto 0) para decidir la clase. Puedes mover ese umbral manualmente para favorecer precision o recall según el problema — con `sgd_classifier.decision_function(X)` en vez de `.predict(X)`. No está implementado en este proyecto, pero es un excelente siguiente ejercicio.

## 8.7 Para seguir practicando

- Agrega el ejercicio del "clasificador tonto" de §8.4 al código para comparar visualmente contra `SGDClassifier`.
- Cambia el problema binario ("¿es un 5?") a **multiclase** completo (predecir el dígito real 0-9) usando `SGDClassifier` directamente (scikit-learn maneja multiclase automáticamente vía estrategia One-vs-Rest) y evalúa con una matriz de confusión 10×10.
- Grafica la curva Precision-Recall variando el umbral de decisión (`precision_recall_curve` de scikit-learn) — es la herramienta estándar para elegir el punto de operación correcto según el problema de negocio.
- Compara `SGDClassifier` contra `RandomForestClassifier` en esta misma tarea (ya conoces Random Forest de [06](06-arboles-de-decision-y-random-forest.md), aquí en su versión de clasificación).
