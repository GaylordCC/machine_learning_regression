"""Logistic regression on Social_Network_Ads.csv.

See documentacion/07-regresion-logistica-y-knn.md for the theory.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..shared.evaluation import evaluate_binary_classifier
from ..shared.social_ads_preprocessing import split_train_test


class LogisticRegressionService:
    def handle_logistic_classification(self):
        X_train, X_test, Y_train, Y_test = split_train_test(random_state=0)

        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
        return evaluate_binary_classifier(model, X_train, Y_train, X_test, Y_test)
