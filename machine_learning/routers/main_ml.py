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