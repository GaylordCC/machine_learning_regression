"""K-Nearest Neighbors on Social_Network_Ads.csv.

See documentacion/07-regresion-logistica-y-knn.md for the theory.
Scaling matters even more here than in logistic regression: KNN is
100% distance-based, so an unscaled feature with a larger numeric
range (e.g. EstimatedSalary vs Age) would dominate the distance
calculation regardless of its real predictive power.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ...core.exceptions import InvalidTrainingDataError
from ..shared.evaluation import evaluate_binary_classifier
from ..shared.social_ads_preprocessing import split_train_test
from ...schemas import KnnClassificationSchema


class KnnService:
    def handle_knn_classification(self, request: KnnClassificationSchema):
        X_train, X_test, Y_train, Y_test = split_train_test(random_state=0)

        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=request.n_neighbors))
        try:
            metrics = evaluate_binary_classifier(model, X_train, Y_train, X_test, Y_test)
        except ValueError as e:
            # sklearn raises this at predict()/kneighbors() time (not fit()) when
            # n_neighbors exceeds the number of training samples available, on
            # the full train split or on a smaller cross-validation fold.
            raise InvalidTrainingDataError(
                f"n_neighbors={request.n_neighbors} is not valid for this dataset: {e}"
            ) from e

        return {"n_neighbors": request.n_neighbors, **metrics}
