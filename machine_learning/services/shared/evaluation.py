"""Shared evaluation helpers: cross-validation summaries and binary-classifier scoring.

Cross-validation runs on the training split only, so the test split stays
untouched as the final, unbiased check. Any estimator that needs scaling must
bundle it in a Pipeline: the scaler is then refit inside every fold, which
keeps the validation fold's statistics out of the fit (no data leakage).
"""
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

CV_FOLDS = 5
CV_RANDOM_STATE = 42

# R2 is undefined on a single-sample fold, so tiny datasets get fewer folds.
_MIN_SAMPLES_PER_FOLD = 2


def _n_splits(n_samples: int) -> int:
    return max(2, min(CV_FOLDS, n_samples // _MIN_SAMPLES_PER_FOLD))


def _mean_std(values, prefix: str) -> dict:
    return {f"{prefix}_mean": float(values.mean()), f"{prefix}_std": float(values.std())}


def cross_validated_regression(estimator, X, y) -> dict:
    """K-fold R2 and RMSE (mean and std across folds) for a regressor."""
    cv = KFold(n_splits=_n_splits(len(y)), shuffle=True, random_state=CV_RANDOM_STATE)
    scores = cross_validate(
        estimator, X, y, cv=cv,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
    )
    return {
        **_mean_std(scores["test_r2"], "cv_r2"),
        # scikit-learn negates error metrics so that higher is always better.
        **_mean_std(-scores["test_rmse"], "cv_rmse"),
    }


def cross_validated_classification(estimator, X, y) -> dict:
    """Stratified k-fold F1 and ROC-AUC (mean and std across folds) for a binary classifier."""
    cv = StratifiedKFold(n_splits=_n_splits(len(y)), shuffle=True, random_state=CV_RANDOM_STATE)
    scores = cross_validate(
        estimator, X, y, cv=cv,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
    )
    return {
        **_mean_std(scores["test_f1"], "cv_f1"),
        **_mean_std(scores["test_roc_auc"], "cv_roc_auc"),
    }


def leave_one_out_regression(estimator, X, y) -> dict:
    """R2 and RMSE over the out-of-sample predictions of leave-one-out CV.

    For datasets too small for a train/test split: each row is predicted by a
    model that never saw it, and the metrics are computed over all of them.
    """
    y_pred = cross_val_predict(estimator, X, y, cv=LeaveOneOut())
    return {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(root_mean_squared_error(y, y_pred)),
    }


def evaluate_binary_classifier(model, X_train, Y_train, X_test, Y_test) -> dict:
    """Fit `model` (an unfitted estimator with predict_proba) and score it.

    Test-split metrics measure generalization; the train F1 next to the test F1
    exposes overfitting; ROC-AUC is threshold-independent; cross-validation on
    the train split measures stability across splits.
    """
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    return {
        "confusion_matrix": confusion_matrix(Y_test, y_pred).tolist(),
        "precision": precision_score(Y_test, y_pred),
        "recall": recall_score(Y_test, y_pred),
        "f1_score": f1_score(Y_test, y_pred),
        "f1_train": f1_score(Y_train, model.predict(X_train)),
        "roc_auc": roc_auc_score(Y_test, y_score),
        **cross_validated_classification(model, X_train, Y_train),
    }
