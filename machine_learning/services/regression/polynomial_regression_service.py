"""Polynomial regression on a synthetic salary-by-position dataset.

See documentacion/04-regresion-polinomica.md for the theory, and try
different `degree` values via the API to see underfitting/overfitting
with your own eyes.
"""
from fastapi import HTTPException
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

from ...core.paths import results_graphics_path
from ...schemas import PolynomialRegressionSchema

POSITIONS = [
    "Pasante de Desarrollo",
    "Desarrollador Junior",
    "Desarrollador Intermedio",
    "Desarrollador Senior",
    "Lider de Proyecto",
    "Gerente de Proyecto",
    "Arquitecto de Software",
    "Director de Desarrollo",
    "Director de Tecnologia",
    "Director Ejecutivo (CEO)",
]
SALARIES = [1200, 2500, 4000, 4800, 6500, 9000, 12850, 15000, 25000, 50000]


class PolynomialRegressionService:
    def build_dataset(self) -> pd.DataFrame:
        return pd.DataFrame({
            "position": POSITIONS,
            "years": range(1, len(POSITIONS) + 1),
            "salary": SALARIES,
        })

    def polynomical_regression(self, request: PolynomialRegressionSchema):
        try:
            data = self.build_dataset()
            X = data["years"].values.reshape(-1, 1)
            Y = data["salary"].values

            linear_model = LinearRegression()
            linear_model.fit(X, Y)

            poly = PolynomialFeatures(degree=request.degree)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression()
            poly_model.fit(X_poly, Y)

            y_pred_poly = poly_model.predict(X_poly)
            y_pred_linear = linear_model.predict(X)

            plt.figure()
            plt.scatter(data["years"], data["salary"])
            plt.plot(X, y_pred_linear, color="gray", linestyle="--", label="Lineal (degree=1)")
            plt.plot(X, y_pred_poly, color="black", label=f"Polinomica (degree={request.degree})")
            plt.legend()
            plt.savefig(results_graphics_path("polynomicalregression.png"))
            plt.close()

            return {
                "degree": request.degree,
                "r2_linear": r2_score(Y, y_pred_linear),
                "r2_polynomial": r2_score(Y, y_pred_poly),
                "data": data.to_dict(orient="records"),
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"polynomical regression: {str(e)}")
