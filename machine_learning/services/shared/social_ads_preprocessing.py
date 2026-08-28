"""Shared preprocessing for Social_Network_Ads.csv, used by both the
logistic regression and KNN classifiers (see documentacion/07).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ...core.paths import sample_data_path


def load_social_ads_dataset() -> pd.DataFrame:
    return pd.read_csv(sample_data_path("Social_Network_Ads.csv"))


def prepare_train_test_split(random_state: int = 0):
    """Return scaled (X_train, X_test, Y_train, Y_test) ready to fit a classifier.

    Age + EstimatedSalary + one-hot encoded Gender -> Purchased.
    Scaling is fit on train only, then reused (transform) on test —
    fitting a scaler on test data would leak test statistics into the
    "unseen" evaluation set.
    """
    data = load_social_ads_dataset()
    X = data.iloc[:, [2, 3]]
    Y = data.iloc[:, -1].values

    encoder = OneHotEncoder()
    gender_encoded = encoder.fit_transform(data[["Gender"]])
    encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=encoder.get_feature_names_out())

    X = pd.concat([X, encoded_df], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test
