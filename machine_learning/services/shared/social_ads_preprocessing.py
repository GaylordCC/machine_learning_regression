"""Shared preprocessing for Social_Network_Ads.csv, used by both the
logistic regression and KNN classifiers (see documentacion/07).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ...core.paths import sample_data_path


def load_social_ads_dataset() -> pd.DataFrame:
    return pd.read_csv(sample_data_path("Social_Network_Ads.csv"))


def split_train_test(random_state: int = 0):
    """Return the unscaled (X_train, X_test, Y_train, Y_test).

    Age + EstimatedSalary + one-hot encoded Gender -> Purchased.
    Scaling is left to the caller, inside a Pipeline: the scaler is then fit
    on train only (and refit inside each cross-validation fold), so no test or
    validation statistics leak into the fit.
    """
    data = load_social_ads_dataset()
    X = data.iloc[:, [2, 3]]
    Y = data.iloc[:, -1].values

    encoder = OneHotEncoder()
    gender_encoded = encoder.fit_transform(data[["Gender"]])
    encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=encoder.get_feature_names_out())

    X = pd.concat([X, encoded_df], axis=1)

    return train_test_split(X, Y, test_size=0.2, random_state=random_state)
