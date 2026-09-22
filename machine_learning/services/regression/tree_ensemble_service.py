"""Regression models trained on the California housing dataset.

Three techniques share the same feature engineering / cleaning /
encoding pipeline (see services/shared/housing_preprocessing.py) and
only differ in which estimator is trained on each incremental set of
columns. See documentacion/06-arboles-de-decision-y-random-forest.md.
"""
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from ...schemas import TreeRegressionSchema, RandomForestRegressionSchema
from ..shared.evaluation import cross_validated_regression
from ..shared.housing_preprocessing import prepare_housing_dataset, HOUSING_MODEL_COLUMNS
from ..shared.plotting import saved_figure


def _train_test_for_columns(columns: list, data_for_corr: pd.DataFrame, encoded_df: pd.DataFrame):
    X = pd.concat([data_for_corr[columns], encoded_df], axis=1).values
    y = data_for_corr["median_house_value"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _incremental_column_scores(model_factory, data_for_corr: pd.DataFrame, encoded_df: pd.DataFrame):
    """Train `model_factory()` adding one column at a time, return R2 and RMSE per step.

    This mirrors a manual feature-selection experiment: does R2 improve
    as we give the model more information? The train R2 next to the test R2
    shows whether the extra columns are being memorized. Pure: no I/O, no plotting.
    """
    columns_used = []
    scores = []

    for col in HOUSING_MODEL_COLUMNS:
        columns_used.append(col)
        X_train, X_test, Y_train, Y_test = _train_test_for_columns(columns_used, data_for_corr, encoded_df)

        model = model_factory()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(Y_test, y_pred)
        scores.append({
            "columns": list(columns_used),
            "r2_score": r2,
            "r2_train": r2_score(Y_train, model.predict(X_train)),
            "rmse": root_mean_squared_error(Y_test, y_pred),
        })
        print("Columnas:", columns_used, "Calificacion:", r2)

    return scores


def _full_model_cross_validation(model_factory, data_for_corr: pd.DataFrame, encoded_df: pd.DataFrame) -> dict:
    """Cross-validate only the model with every column: running it at each
    incremental step would multiply the request time by the number of steps."""
    X_train, _, Y_train, _ = _train_test_for_columns(HOUSING_MODEL_COLUMNS, data_for_corr, encoded_df)
    return cross_validated_regression(model_factory(), X_train, Y_train)


def _save_housing_exploratory_plots(data: pd.DataFrame, data_for_corr: pd.DataFrame) -> dict:
    histograms_fig = saved_figure("histograms.png")
    with histograms_fig:
        data.hist(bins=50, figsize=(20, 15))

    scatter_fig = saved_figure("scatter_plot.png")
    with scatter_fig as plt:
        data.plot(
            kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=data["population"] / 100, label="population", figsize=(15, 7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
        )
        plt.legend()

    correlation_fig = saved_figure("correlation_plot.png")
    with correlation_fig as plt:
        plt.figure(figsize=(20, 10))
        sns.heatmap(data_for_corr.corr(), annot=True)

    return {
        "histograms": histograms_fig.filename,
        "scatter_plot": scatter_fig.filename,
        "correlation_plot": correlation_fig.filename,
    }


class TreeEnsembleService:
    def housing_linear_regression(self):
        """Baseline: plain linear regression + exploratory plots on housing.csv."""
        data, data_for_corr, encoded_df = prepare_housing_dataset()
        scores = _incremental_column_scores(LinearRegression, data_for_corr, encoded_df)
        cross_validation = _full_model_cross_validation(LinearRegression, data_for_corr, encoded_df)
        plot_files = _save_housing_exploratory_plots(data, data_for_corr)
        return {
            "model": "linear_regression",
            "scores_by_columns": scores,
            "cross_validation": cross_validation,
            "plot_files": plot_files,
        }

    def decision_tree_regression(self, request: TreeRegressionSchema):
        _, data_for_corr, encoded_df = prepare_housing_dataset()
        model_factory = lambda: DecisionTreeRegressor(max_depth=request.max_depth, random_state=42)
        scores = _incremental_column_scores(model_factory, data_for_corr, encoded_df)
        cross_validation = _full_model_cross_validation(model_factory, data_for_corr, encoded_df)
        return {
            "model": "decision_tree",
            "max_depth": request.max_depth,
            "scores_by_columns": scores,
            "cross_validation": cross_validation,
        }

    def random_forest_regression(self, request: RandomForestRegressionSchema):
        _, data_for_corr, encoded_df = prepare_housing_dataset()
        model_factory = lambda: RandomForestRegressor(
            n_estimators=request.n_estimators, max_depth=request.max_depth, random_state=42
        )
        scores = _incremental_column_scores(model_factory, data_for_corr, encoded_df)
        cross_validation = _full_model_cross_validation(model_factory, data_for_corr, encoded_df)
        return {
            "model": "random_forest",
            "n_estimators": request.n_estimators,
            "max_depth": request.max_depth,
            "scores_by_columns": scores,
            "cross_validation": cross_validation,
        }
