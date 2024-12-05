from fastapi import APIRouter

from ..services.classification_ml_service import ClassificationAlgorithmService

# Define a new API Router
router = APIRouter(
    tags = ['Machine Learning Classification Endpoints']
)

# Endpoint to implement classifiction ml algorithm
@router.post('/classification-algorithm')
def process_request():
    return ClassificationAlgorithmService().handle_classification_image()

# Endpoint to implement logistic regression classifiction algorithm
@router.post('/logistic-regression-classification')
def process_request():
    return ClassificationAlgorithmService().handle_logistic_classification()

# Endpoint to implement K-NN (K nearest neighbors) classifiction algorithm
@router.post('/logistic-regression-classification')
def process_request():
    return ClassificationAlgorithmService().handle_knn_classification()