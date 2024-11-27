from fastapi import APIRouter

from ..services.classification_ml_service import ClassificationAlgorithmService

# Define a new API Router
router = APIRouter(
    tags = ['Machine Learning Classification Endpoints']
)


@router.post('/classification-algorithm')
def process_request():
    
    return ClassificationAlgorithmService().handle_classification_image()