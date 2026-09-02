"""Polynomial regression on a synthetic salary-by-position dataset.

See documentacion/04-regresion-polinomica.md for the theory, and try
different `degree` values via the API to see underfitting/overfitting
with your own eyes.
"""
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

from ...schemas import PolynomialRegressionSchema
from ..shared.plotting import saved_figure

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

    def _train(self, data: pd.DataFrame, degree: int) -> dict:
        """Fit linear + polynomial regression and score both. Pure: no I/O,
        no plotting -- testable with any DataFrame shaped the same way."""
        X = data["years"].values.reshape(-1, 1)
        Y = data["salary"].values

        linear_model = LinearRegression()
        linear_model.fit(X, Y)

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, Y)

        y_pred_poly = poly_model.predict(X_poly)
        y_pred_linear = linear_model.predict(X)

        return {
            "degree": degree,
            "r2_linear": r2_score(Y, y_pred_linear),
            "r2_polynomial": r2_score(Y, y_pred_poly),
            "X": X,
            "y_pred_linear": y_pred_linear,
            "y_pred_poly": y_pred_poly,
        }

    def polynomical_regression(self, request: PolynomialRegressionSchema):
        data = self.build_dataset()
        result = self._train(data, request.degree)

        fig = saved_figure(f"polynomicalregression_degree{request.degree}.png")
        with fig as plt:
            plt.scatter(data["years"], data["salary"])
            plt.plot(result["X"], result["y_pred_linear"], color="gray", linestyle="--", label="Lineal (degree=1)")
            plt.plot(result["X"], result["y_pred_poly"], color="black", label=f"Polinomica (degree={request.degree})")
            plt.legend()

        return {
            "degree": result["degree"],
            "r2_linear": result["r2_linear"],
            "r2_polynomial": result["r2_polynomial"],
            "data": data.to_dict(orient="records"),
            "plot_file": fig.filename,
        }
