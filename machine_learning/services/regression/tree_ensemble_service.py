"""Regression models trained on the California housing dataset.

Three techniques share the same feature engineering / cleaning /
encoding pipeline (see services/shared/housing_preprocessing.py) and
only differ in which estimator is trained on each incremental set of
columns. See documentacion/06-arboles-de-decision-y-random-forest.md.
"""
from fastapi import HTTPException
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from ...core.paths import results_graphics_path
from ...schemas import TreeRegressionSchema, RandomForestRegressionSchema
from ..shared.housing_preprocessing import prepare_housing_dataset, HOUSING_MODEL_COLUMNS


def _incremental_column_scores(model_factory, data_for_corr: pd.DataFrame, encoded_df: pd.DataFrame):
    """Train `model_factory()` adding one column at a time, return R2 per step.

    This mirrors a manual feature-selection experiment: does R2 improve
    as we give the model more information?
    """
    y = data_for_corr["median_house_value"].values
    columns_used = []
    scores = []

    for col in HOUSING_MODEL_COLUMNS:
        columns_used.append(col)
        X = pd.concat([data_for_corr[columns_used], encoded_df], axis=1).values
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = model_factory()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(Y_test, y_pred)
        scores.append({"columns": list(columns_used), "r2_score": r2})
        print("Columnas:", columns_used, "Calificacion:", r2)

    return scores


class TreeEnsembleService:
    def housing_linear_regression(self):
        """Baseline: plain linear regression + exploratory plots on housing.csv."""
        try:
            data, data_for_corr, encoded_df = prepare_housing_dataset()

            scores = _incremental_column_scores(LinearRegression, data_for_corr, encoded_df)

            data.hist(bins=50, figsize=(20, 15))
            plt.savefig(results_graphics_path("histograms.png"))
            plt.close("all")

            data.plot(
                kind="scatter", x="longitude", y="latitude", alpha=0.4,
                s=data["population"] / 100, label="population", figsize=(15, 7),
                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            )
            plt.legend()
            plt.savefig(results_graphics_path("scatter_plot.png"))
            plt.close()

            plt.figure(figsize=(20, 10))
            sns.heatmap(data_for_corr.corr(), annot=True)
            plt.savefig(results_graphics_path("correlation_plot.png"))
            plt.close()

            return {"model": "linear_regression", "scores_by_columns": scores}
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"housing linear regression: {str(e)}")

    def decision_tree_regression(self, request: TreeRegressionSchema):
        try:
            _, data_for_corr, encoded_df = prepare_housing_dataset()
            model_factory = lambda: DecisionTreeRegressor(max_depth=request.max_depth, random_state=42)
            scores = _incremental_column_scores(model_factory, data_for_corr, encoded_df)
            return {"model": "decision_tree", "max_depth": request.max_depth, "scores_by_columns": scores}
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"decision tree regression: {str(e)}")

    def random_forest_regression(self, request: RandomForestRegressionSchema):
        try:
            _, data_for_corr, encoded_df = prepare_housing_dataset()
            model_factory = lambda: RandomForestRegressor(
                n_estimators=request.n_estimators, max_depth=request.max_depth, random_state=42
            )
            scores = _incremental_column_scores(model_factory, data_for_corr, encoded_df)
            return {
                "model": "random_forest",
                "n_estimators": request.n_estimators,
                "max_depth": request.max_depth,
                "scores_by_columns": scores,
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"random forest regression: {str(e)}")
