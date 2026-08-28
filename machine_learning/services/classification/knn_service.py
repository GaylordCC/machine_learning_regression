"""K-Nearest Neighbors on Social_Network_Ads.csv.

See documentacion/07-regresion-logistica-y-knn.md for the theory.
Scaling matters even more here than in logistic regression: KNN is
100% distance-based, so an unscaled feature with a larger numeric
range (e.g. EstimatedSalary vs Age) would dominate the distance
calculation regardless of its real predictive power.
"""
from fastapi import HTTPException

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from ..shared.social_ads_preprocessing import prepare_train_test_split
from ...schemas import KnnClassificationSchema


class KnnService:
    def handle_knn_classification(self, request: KnnClassificationSchema):
        try:
            X_train, X_test, Y_train, Y_test = prepare_train_test_split(random_state=0)

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
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"knn classification: {str(e)}")
