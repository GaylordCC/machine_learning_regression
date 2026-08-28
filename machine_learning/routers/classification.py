from fastapi import APIRouter

from ..services.classification.image_classification_service import ImageClassificationService
from ..services.classification.logistic_regression_service import LogisticRegressionService
from ..services.classification.knn_service import KnnService
from ..schemas import KnnClassificationSchema

router = APIRouter(tags=["Machine Learning Classification Endpoints"])


@router.post("/classification-algorithm")
def classification_algorithm():
    return ImageClassificationService().handle_classification_image()


@router.post("/logistic-regression-classification")
def logistic_regression_classification():
    return LogisticRegressionService().handle_logistic_classification()


@router.post("/knn-classification")
def knn_classification(request: KnnClassificationSchema = KnnClassificationSchema()):
    return KnnService().handle_knn_classification(request=request)
