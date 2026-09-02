from fastapi import APIRouter

from ..services.regression.linear_regression_service import LinearRegressionService
from ..services.regression.polynomial_regression_service import PolynomialRegressionService
from ..services.regression.svr_service import SvrRegressionService
from ..services.regression.tree_ensemble_service import TreeEnsembleService
from ..schemas import (
    RegressionSchema,
    PolynomialRegressionSchema,
    SvrRegressionSchema,
    TreeRegressionSchema,
    RandomForestRegressionSchema,
)

router = APIRouter(prefix="/v1", tags=["Machine Learning Regression Endpoints"])


@router.post("/machine-learning")
def exploratory_analysis():
    return LinearRegressionService().handle_user_query()


@router.post("/linear-regression")
def linear_regression(request: RegressionSchema):
    return LinearRegressionService().regression_linear_model(request=request)


@router.post("/multi-linear-regression")
def multi_linear_regression():
    return LinearRegressionService().regression_multi_linear_model()


@router.post("/polynomial-regression")
def polynomial_regression(request: PolynomialRegressionSchema = PolynomialRegressionSchema()):
    return PolynomialRegressionService().polynomical_regression(request=request)


@router.post("/svr-regression")
def svr_regression(request: SvrRegressionSchema = SvrRegressionSchema()):
    return SvrRegressionService().svr_regression(request=request)


@router.post("/housing-linear-regression")
def housing_linear_regression():
    return TreeEnsembleService().housing_linear_regression()


@router.post("/decision-tree-regression")
def decision_tree_regression(request: TreeRegressionSchema = TreeRegressionSchema()):
    return TreeEnsembleService().decision_tree_regression(request=request)


@router.post("/random-forest-regression")
def random_forest_regression(request: RandomForestRegressionSchema = RandomForestRegressionSchema()):
    return TreeEnsembleService().random_forest_regression(request=request)
