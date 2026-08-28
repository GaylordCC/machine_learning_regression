# 9. Glosario de términos de ML

Referencia rápida en español. Ordenado alfabéticamente. Cada término enlaza (cuando aplica) al capítulo donde se explica con más profundidad.

- **Accuracy (exactitud)**: proporción de predicciones correctas sobre el total. Ver [02](02-fundamentos-de-machine-learning.md#26-métricas-de-evaluación).
- **Bagging (Bootstrap Aggregating)**: técnica de *ensemble* que entrena varios modelos sobre muestras aleatorias distintas de los datos y promedia sus resultados. Base de Random Forest. Ver [06](06-arboles-de-decision-y-random-forest.md).
- **Baseline**: modelo simple usado como punto de referencia antes de probar modelos más complejos.
- **Clasificación**: tarea de ML donde el target es una categoría discreta (ej. sí/no, tipo A/B/C). Ver [02](02-fundamentos-de-machine-learning.md).
- **Coeficiente**: parámetro que un modelo lineal aprende para cada variable (`b1`, `b2`...), indica el peso/dirección de esa variable en la predicción. Ver [03](03-regresion-lineal-simple-y-multiple.md).
- **Cross-validation (validación cruzada)**: técnica de evaluación que divide los datos en `k` particiones y entrena/evalúa `k` veces, rotando cuál partición se usa como test. Ver [08](08-clasificacion-mnist-y-metricas.md).
- **Data leakage (fuga de datos)**: cuando información del conjunto de test "se filtra" al proceso de entrenamiento o de ajuste de transformadores, invalidando la evaluación. Ver [01](01-arquitectura-del-proyecto.md) y [07](07-regresion-logistica-y-knn.md).
- **Ensemble (conjunto)**: modelo compuesto por varios modelos más simples combinados (ej. Random Forest = muchos árboles).
- **Epsilon (SVR)**: ancho del "tubo" dentro del cual SVR no penaliza errores. Ver [05](05-regresion-svr.md).
- **Feature (característica/variable)**: cada columna de entrada (`X`) usada para predecir.
- **Feature engineering (ingeniería de atributos)**: crear nuevas variables a partir de las existentes para mejorar el modelo (ej. `rooms_per_household`). Ver [06](06-arboles-de-decision-y-random-forest.md).
- **F1-score**: media armónica entre precision y recall. Ver [02](02-fundamentos-de-machine-learning.md).
- **Hiperparámetro**: configuración elegida por la persona antes de entrenar (ej. `degree`, `k`, `n_estimators`), no aprendida por el modelo.
- **Kernel**: función que define la "forma" de las fronteras/curvas que un modelo SVM/SVR puede aprender (`linear`, `poly`, `rbf`). Ver [05](05-regresion-svr.md).
- **Matriz de confusión**: tabla que cruza predicciones vs. valores reales en clasificación (TP, TN, FP, FN). Ver [02](02-fundamentos-de-machine-learning.md).
- **Modelo**: objeto que aprende un patrón de los datos (ej. `LinearRegression()`, `SVR()`).
- **OneHotEncoder**: convierte una variable categórica sin orden en varias columnas binarias. Ver [02](02-fundamentos-de-machine-learning.md).
- **OrdinalEncoder**: convierte una variable categórica **con orden** en números enteros. Ver [02](02-fundamentos-de-machine-learning.md).
- **Overfitting (sobre-ajuste)**: el modelo memoriza el ruido de entrenamiento y generaliza mal a datos nuevos. Ver [02](02-fundamentos-de-machine-learning.md), [04](04-regresion-polinomica.md).
- **Parámetro**: valor que el modelo aprende automáticamente durante el entrenamiento (ej. los coeficientes de una regresión).
- **Pipeline**: encadenamiento de pasos de transformación + modelo en un solo objeto reutilizable (`sklearn.pipeline.Pipeline`). Ver [10](10-hoja-de-ruta.md).
- **Precision (precisión)**: de lo que el modelo predijo como positivo, cuánto era realmente positivo. Ver [02](02-fundamentos-de-machine-learning.md).
- **R² (coeficiente de determinación)**: proporción de la variabilidad de `Y` explicada por el modelo, en regresión. Ver [02](02-fundamentos-de-machine-learning.md).
- **Recall (exhaustividad/sensibilidad)**: de todos los positivos reales, cuántos detectó el modelo. Ver [02](02-fundamentos-de-machine-learning.md).
- **Regresión**: tarea de ML donde el target es un valor numérico continuo. Ver [02](02-fundamentos-de-machine-learning.md).
- **Regularización**: técnica que penaliza coeficientes grandes para reducir overfitting (ej. Ridge, Lasso). Ver [10](10-hoja-de-ruta.md).
- **RMSE (Root Mean Squared Error)**: raíz del error cuadrático medio, en las mismas unidades que `Y`. Ver [02](02-fundamentos-de-machine-learning.md).
- **StandardScaler**: transforma variables numéricas a media 0 y desviación estándar 1. Ver [02](02-fundamentos-de-machine-learning.md).
- **Target / label**: la variable que se quiere predecir (`Y`).
- **Train/test split**: dividir los datos en un conjunto para entrenar y otro para evaluar, sin solaparse. Ver [02](02-fundamentos-de-machine-learning.md).
- **Underfitting (sub-ajuste)**: el modelo es demasiado simple y no captura el patrón real, ni siquiera en entrenamiento. Ver [02](02-fundamentos-de-machine-learning.md).
- **Vector de soporte**: puntos más cercanos a la frontera/tubo de decisión en SVM/SVR, los que determinan su forma. Ver [05](05-regresion-svr.md).
