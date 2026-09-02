"""Exploratory analysis + simple/multiple linear regression on Advertising.csv.

See documentacion/03-regresion-lineal-simple-y-multiple.md for the theory.
"""
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

from ...core.paths import sample_data_path
from ...schemas import RegressionSchema
from ..shared.plotting import saved_figure

ADVERTISING_CSV = "Advertising.csv"

# Keys only needed to draw the plot -- stripped before a _train_* result is
# returned to the caller as the public response body.
_INTERNAL_KEYS = ("X_test", "Y_test", "y_predict")


def _public_metrics(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in _INTERNAL_KEYS}


class LinearRegressionService:
    def load_advertising_data(self) -> pd.DataFrame:
        return pd.read_csv(sample_data_path(ADVERTISING_CSV))

    def handle_user_query(self):
        """Quick exploratory look at how each advertising medium relates to sales."""
        data = self.load_advertising_data()
        data_dict = data.to_dict(orient="records")

        for col in ["TV", "Radio", "Newspaper"]:
            with saved_figure(f"plot_{col}.png") as plt:
                plt.plot(data[col], data["Sales"], "ro")
                plt.title(f"Ventas respecto a la publicidad en {col}")

        return data_dict

    def _train_simple(self, data: pd.DataFrame, column_name: str) -> dict:
        """Fit + score a simple linear regression. Pure: no I/O, no plotting --
        testable with any DataFrame, including a synthetic one."""
        X = data[column_name].values.reshape(-1, 1)
        Y = data["Sales"].values

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        lin_reg = LinearRegression()
        lin_reg.fit(X_train, Y_train)
        y_predict = lin_reg.predict(X_test)

        rmse = root_mean_squared_error(Y_test, y_predict)
        r2 = r2_score(Y_test, y_predict)
        print(f"RMSE: {rmse}, R2: {r2}")

        return {
            "predictions": y_predict.tolist(),
            "rmse": rmse,
            "r2_score": r2,
            "X_test": X_test,
            "Y_test": Y_test,
            "y_predict": y_predict,
        }

    def regression_linear_model(self, request: RegressionSchema):
        """Simple linear regression: one advertising medium -> Sales."""
        data = self.load_advertising_data()
        result = self._train_simple(data, request.column_name.value)

        fig = saved_figure(f"plotregression_{request.column_name.value}.png")
        with fig as plt:
            plt.plot(result["X_test"], result["Y_test"], "ro")
            plt.title(f"Regresion lineal para predecir datos de {request.column_name.value}")
            plt.plot(result["X_test"], result["y_predict"])

        return {**_public_metrics(result), "plot_file": fig.filename}

    def _train_multi(self, data: pd.DataFrame) -> dict:
        """Fit + score linear regression on TV + Radio -> Sales (Newspaper
        dropped, low correlation). Pure: no I/O, no plotting."""
        X = data.drop(["Newspaper", "Sales"], axis=1).values
        Y = data["Sales"].values

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        lin_reg = LinearRegression()
        lin_reg.fit(X_train, Y_train)
        y_predict = lin_reg.predict(X_test)

        rmse = root_mean_squared_error(Y_test, y_predict)
        r2 = r2_score(Y_test, y_predict)
        print(f"RMSE: {rmse}, R2: {r2}")

        return {
            "predictions": y_predict.tolist(),
            "rmse": rmse,
            "r2_score": r2,
            "Y_test": Y_test,
            "y_predict": y_predict,
        }

    def regression_multi_linear_model(self):
        """Multiple linear regression: TV + Radio -> Sales."""
        data = self.load_advertising_data()
        result = self._train_multi(data)

        fig = saved_figure("plotmultiregression.png")
        with fig:
            sns.regplot(x=result["Y_test"], y=result["y_predict"])

        return {**_public_metrics(result), "plot_file": fig.filename}
