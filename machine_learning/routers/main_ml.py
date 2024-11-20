from fastapi import APIRouter
from ..services.main_ml_service import MachineLearningService
from ..schemas import RegressionSchema


router = APIRouter(
    tags = ['Machine Learning']
)

@router.post('/machine-learning')
def process_request():
    ml_response = MachineLearningService().handle_user_query()
    return ml_response

@router.post('/linear-regression')
def process_request(
    request: RegressionSchema
):
    rlm_response = MachineLearningService().regression_linear_model(request=request)
    return rlm_response

@router.post('/multi-linear-regression')
def process_request():
    rmlm_response = MachineLearningService().regression_multi_linear_model()
    return rmlm_response

@router.post('/polynomical-regression')
def process_request():
    polynomical_response = MachineLearningService().polynomical_regression()
    return polynomical_response

@router.post('/svr-regression')
def process_request():
    polynomical_response = MachineLearningService().svr_regression()
    return polynomical_response