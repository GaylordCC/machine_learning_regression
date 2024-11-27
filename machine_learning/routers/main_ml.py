from fastapi import APIRouter
from ..services.main_ml_service import MachineLearningService
from ..schemas import RegressionSchema

# Define a new API router
router = APIRouter(
    tags = ['Machine Learning Regression Endpoints']
)

# Endpoint for processing a general machine learning request
@router.post('/machine-learning')
def process_request():
    ml_response = MachineLearningService().handle_user_query()
    return ml_response

# Endpoint for performing linear regression
@router.post('/linear-regression')
def process_request(
    request: RegressionSchema
):
    rlm_response = MachineLearningService().regression_linear_model(request=request)
    return rlm_response

# Endpoint for performing multiple linear regression
@router.post('/multi-linear-regression')
def process_request():
    rmlm_response = MachineLearningService().regression_multi_linear_model()
    return rmlm_response

# Endpoint for performing polynomial regression
@router.post('/polynomical-regression')
def process_request():
    polynomical_response = MachineLearningService().polynomical_regression()
    return polynomical_response

# Endpoint for performing support vector regression (SVR)
@router.post('/svr-regression')
def process_request():
    svr_response = MachineLearningService().svr_regression()
    return svr_response

# Endpoint for performing decision tree regression
@router.post('/tree-regression')
def process_request():
    tree_response = MachineLearningService().tree_regression()
    return tree_response

# Endpoint for performing random tree regression
@router.post('/random-tree-regression')
def process_request():
    random_tree_response = MachineLearningService().random_tree_regression()
    return random_tree_response

# Endpoint for performing random forest regression
@router.post('/random-forest-regression')
def process_request():
    random_forest_response = MachineLearningService().random_forest_regression()
    return random_forest_response
