"""Exploratory analysis + simple/multiple linear regression on Advertising.csv.

See documentacion/03-regresion-lineal-simple-y-multiple.md for the theory.
"""
from fastapi import HTTPException
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from ...core.paths import sample_data_path, results_graphics_path
from ...schemas import RegressionSchema

ADVERTISING_CSV = "Advertising.csv"


class LinearRegressionService:
    def load_advertising_data(self) -> pd.DataFrame:
        return pd.read_csv(sample_data_path(ADVERTISING_CSV))

    def handle_user_query(self):
        """Quick exploratory look at how each advertising medium relates to sales."""
        try:
            data = self.load_advertising_data()
            data_dict = data.to_dict(orient="records")

            for col in ["TV", "Radio", "Newspaper"]:
                plt.plot(data[col], data["Sales"], "ro")
                plt.title(f"Ventas respecto a la publicidad en {col}")
                plt.savefig(results_graphics_path(f"plot_{col}.png"))
                plt.close()

            return data_dict
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"handle_user_query: {str(e)}")

    def regression_linear_model(self, request: RegressionSchema):
        """Simple linear regression: one advertising medium -> Sales."""
        try:
            data = self.load_advertising_data()
            X = data[request.column_name].values.reshape(-1, 1)
            Y = data["Sales"].values

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            lin_reg = LinearRegression()
            lin_reg.fit(X_train, Y_train)
            y_predict = lin_reg.predict(X_test)

            rmse = mean_squared_error(Y_test, y_predict, squared=False)
            r2 = r2_score(Y_test, y_predict)
            print(f"RMSE: {rmse}, R2: {r2}")

            plt.plot(X_test, Y_test, "ro")
            plt.title(f"Regresion lineal para predecir datos de {request.column_name.value}")
            plt.plot(X_test, y_predict)
            plt.savefig(results_graphics_path("plotregression.png"))
            plt.close()

            return {
                "predictions": y_predict.tolist(),
                "rmse": rmse,
                "r2_score": r2,
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"regression linear model: {str(e)}")

    def regression_multi_linear_model(self):
        """Multiple linear regression: TV + Radio -> Sales (Newspaper dropped, low correlation)."""
        try:
            data = self.load_advertising_data()
            X = data.drop(["Newspaper", "Sales"], axis=1).values
            Y = data["Sales"].values

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            lin_reg = LinearRegression()
            lin_reg.fit(X_train, Y_train)
            y_predict = lin_reg.predict(X_test)

            rmse = mean_squared_error(Y_test, y_predict, squared=False)
            r2 = r2_score(Y_test, y_predict)
            print(f"RMSE: {rmse}, R2: {r2}")

            plt.figure()
            sns.regplot(x=Y_test, y=y_predict)
            plt.savefig(results_graphics_path("plotmultiregression.png"))
            plt.close()

            return {
                "predictions": y_predict.tolist(),
                "rmse": rmse,
                "r2_score": r2,
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"regression multi linear model: {str(e)}")
