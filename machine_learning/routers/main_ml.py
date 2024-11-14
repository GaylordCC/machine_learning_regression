from fastapi import APIRouter
from ..services.main_ml_service import MachineLearningService


router = APIRouter(
    tags = ['Machine Learning']
)

@router.post('/machine-learning')
def process_request():
    ml_response = MachineLearningService().handle_user_query()
    
    return ml_response

@router.post('/linear-regression')
def process_request():
    rlm_response = MachineLearningService().regression_linear_model()
    
    return rlm_response