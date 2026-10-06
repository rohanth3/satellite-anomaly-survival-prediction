import os

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


FEATURES = [
    "perigee",
    "inclination",
    "ssn",
    "month"
]


def _drop_unnamed(df):
    return df.loc[
        :,
        ~df.columns.astype(str).str.startswith("Unnamed")
    ]


def load_real(datadir="."):

    candidates = [
        datadir,
        os.path.join(datadir, "data"),
        ".",
        "data",
        "..",
        os.path.join("..", "data")
    ]
    resolved_dir = datadir
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "data.csv")) and os.path.exists(os.path.join(cand, "satfeat.csv")):
            resolved_dir = cand
            break

    ev = _drop_unnamed(
        pd.read_csv(
            os.path.join(
                resolved_dir,
                "data.csv"
            )
        )
    )

    sf = _drop_unnamed(
        pd.read_csv(
            os.path.join(
                resolved_dir,
                "satfeat.csv"
            )
        )
    )

    sf = sf.rename(
        columns={
            "Name": "BIRD",
            "Perigee (km)": "perigee",
            "Inclination (deg)": "inclination"
        }
    )

    df = ev.merge(
        sf,
        on="BIRD",
        how="inner"
    )

    df = df.rename(
        columns={
            "TTE": "tte",
            "SSN": "ssn",
            "Month": "month"
        }
    )

    df = df[
        [
            "BIRD",
            "tte"
        ] + FEATURES
    ].dropna()

    return df.reset_index(drop=True)


def mean_relative_error(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    return np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask])
            / y_true[mask]
        )
    ) * 100


def main():

    df = load_real(".")

    print(
        f"Dataset: {len(df)} events, "
        f"{df['BIRD'].nunique()} satellites"
    )

    X = df[FEATURES].values
    y = df["tte"].values

    kf = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    train_relative_errors = []
    test_relative_errors = []

    test_actual = []
    test_predicted = []

    for train_idx, test_idx in kf.split(X):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        model = SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
            gamma="auto"
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        train_pred = model.predict(
            X_train_scaled
        )

        test_pred = model.predict(
            X_test_scaled
        )

        train_relative_errors.append(
            mean_relative_error(
                y_train,
                train_pred
            )
        )

        test_relative_errors.append(
            mean_relative_error(
                y_test,
                test_pred
            )
        )

        test_actual.extend(
            y_test
        )

        test_predicted.extend(
            test_pred
        )

    test_actual = np.asarray(
        test_actual
    )

    test_predicted = np.asarray(
        test_predicted
    )

    mae = mean_absolute_error(
        test_actual,
        test_predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            test_actual,
            test_predicted
        )
    )

    print()
    print(
        "SVR - 10-fold cross-validation"
    )

    print(
        f"Train mean relative error : "
        f"{np.mean(train_relative_errors):.1f}%"
    )

    print(
        f"Test mean relative error  : "
        f"{np.mean(test_relative_errors):.1f}%"
    )

    print(
        f"Test MAE  : {mae:.2f} days"
    )

    print(
        f"Test RMSE : {rmse:.2f} days"
    )


if __name__ == "__main__":
    main()