"""Logistic regression on Social_Network_Ads.csv.

See documentacion/07-regresion-logistica-y-knn.md for the theory.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from ..shared.social_ads_preprocessing import prepare_train_test_split


class LogisticRegressionService:
    def handle_logistic_classification(self):
        X_train, X_test, Y_train, Y_test = prepare_train_test_split(random_state=0)

        log_reg = LogisticRegression(random_state=0)
        log_reg.fit(X_train, Y_train)
        y_pred = log_reg.predict(X_test)

        return {
            "confusion_matrix": confusion_matrix(Y_test, y_pred).tolist(),
            "precision": precision_score(Y_test, y_pred),
            "recall": recall_score(Y_test, y_pred),
            "f1_score": f1_score(Y_test, y_pred),
        }
