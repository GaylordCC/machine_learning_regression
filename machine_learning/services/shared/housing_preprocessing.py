"""Shared preprocessing pipeline for the California housing dataset.

Used by every regression technique trained on `housing.csv` (linear
regression baseline, decision tree, random forest) so the feature
engineering / cleaning / encoding logic lives in a single place instead
of being copy-pasted per model. See documentacion/06 for the theory
behind each step.
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ...core.paths import sample_data_path

HOUSING_MODEL_COLUMNS = [
    "median_income",
    "rooms_per_household",
    "total_rooms",
    "housing_median_age",
    "households",
    "latitude",
    "longitude",
]


def load_housing_dataset() -> pd.DataFrame:
    return pd.read_csv(sample_data_path("housing.csv"))


def engineer_housing_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add derived ratio features and drop the categorical column."""
    data_for_corr = data.drop(["ocean_proximity"], axis=1).copy()
    data_for_corr["rooms_per_household"] = data_for_corr["total_rooms"] / data_for_corr["households"]
    data_for_corr["bedrooms_per_room"] = data_for_corr["total_bedrooms"] / data_for_corr["households"]
    data_for_corr["population_per_household"] = data_for_corr["population"] / data_for_corr["households"]

    # Median imputation: robust to outliers, simple to explain (see documentacion/06).
    data_for_corr["total_bedrooms"] = data_for_corr["total_bedrooms"].fillna(
        data_for_corr["total_bedrooms"].median()
    )
    return data_for_corr


def one_hot_encode_ocean_proximity(data: pd.DataFrame) -> pd.DataFrame:
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(data[["ocean_proximity"]])
    return pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out())


def prepare_housing_dataset():
    """Load, clean and encode the housing dataset.

    Returns (data, data_for_corr, encoded_ocean_df) ready to build
    (X, y) pairs for any of the housing regression models.
    """
    data = load_housing_dataset()
    data_for_corr = engineer_housing_features(data)
    encoded_df = one_hot_encode_ocean_proximity(data)
    return data, data_for_corr, encoded_df
