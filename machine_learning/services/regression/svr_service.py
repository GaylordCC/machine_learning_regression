"""Support Vector Regression on Advertising.csv (TV + Radio -> Sales).

See documentacion/05-regresion-svr.md for the theory. Unlike linear
regression, SVR with an rbf/poly kernel is distance-based, so features
(and the target) are scaled before training.
"""
from fastapi import HTTPException
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from ...core.paths import sample_data_path
from ...schemas import SvrRegressionSchema


class SvrRegressionService:
    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(sample_data_path("Advertising.csv"))
        return data.iloc[:, 1:]  # drop the CSV's row-index column

    def svr_regression(self, request: SvrRegressionSchema):
        try:
            data = self.load_data()
            X = data.drop(["Newspaper", "Sales"], axis=1).values
            Y = data["Sales"].values.reshape(-1, 1)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            # SVR is distance-based: scale X and Y, fitting only on train.
            sc_X = StandardScaler()
            X_train_scaled = sc_X.fit_transform(X_train)
            X_test_scaled = sc_X.transform(X_test)

            sc_Y = StandardScaler()
            Y_train_scaled = sc_Y.fit_transform(Y_train).ravel()

            svr = SVR(kernel=request.kernel.value)
            svr.fit(X_train_scaled, Y_train_scaled)

            y_predict_scaled = svr.predict(X_test_scaled)
            y_predict = sc_Y.inverse_transform(y_predict_scaled.reshape(-1, 1)).ravel()

            r2 = r2_score(Y_test, y_predict)
            rmse = root_mean_squared_error(Y_test, y_predict)
            print(f"SVR kernel={request.kernel.value} R2: {r2}, RMSE: {rmse}")

            return {
                "kernel": request.kernel.value,
                "predictions": y_predict.tolist(),
                "rmse": rmse,
                "r2_score": r2,
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"support vector regression: {str(e)}")
