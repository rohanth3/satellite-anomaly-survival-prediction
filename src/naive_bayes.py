import os

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


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


def kaplan_meier_prior(tte):

    tte = np.asarray(tte)

    times = np.sort(
        np.unique(tte)
    )

    survival = 1.0

    prior = {}

    n = len(tte)

    for t in times:

        at_risk = np.sum(
            tte >= t
        )

        events = np.sum(
            tte == t
        )

        if at_risk > 0:
            survival *= (
                1.0 -
                events / at_risk
            )

        prior[t] = survival

    return prior


def gaussian_pdf(x, mean, std):

    std = max(
        std,
        1e-6
    )

    coefficient = (
        1.0 /
        (
            np.sqrt(2.0 * np.pi)
            * std
        )
    )

    exponent = -0.5 * (
        (x - mean) / std
    ) ** 2

    return coefficient * np.exp(
        exponent
    )


def fit_feature_distributions(
    X,
    y
):

    distributions = {}

    unique_times = np.sort(
        np.unique(y)
    )

    for time in unique_times:

        event_mask = (
            y <= time
        )

        class_event = X[
            event_mask
        ]

        class_survival = X[
            ~event_mask
        ]

        event_params = []
        survival_params = []

        for j in range(
            X.shape[1]
        ):

            event_params.append(
                (
                    np.mean(
                        class_event[:, j]
                    ),
                    np.std(
                        class_event[:, j]
                    )
                )
            )

            if len(class_survival) > 0:

                survival_params.append(
                    (
                        np.mean(
                            class_survival[:, j]
                        ),
                        np.std(
                            class_survival[:, j]
                        )
                    )
                )

            else:

                survival_params.append(
                    (
                        np.mean(
                            X[:, j]
                        ),
                        np.std(
                            X[:, j]
                        )
                    )
                )

        distributions[time] = {
            "event": event_params,
            "survival": survival_params
        }

    return distributions


def predict_tte(
    X,
    distributions,
    km_prior
):

    times = sorted(
        distributions.keys()
    )

    predictions = []

    for sample in X:

        best_time = times[-1]
        best_score = -np.inf

        for time in times:

            params = distributions[
                time
            ]

            event_likelihood = 1.0
            survival_likelihood = 1.0

            for j in range(
                len(FEATURES)
            ):

                event_mean, event_std = (
                    params["event"][j]
                )

                survival_mean, survival_std = (
                    params["survival"][j]
                )

                event_likelihood *= (
                    gaussian_pdf(
                        sample[j],
                        event_mean,
                        event_std
                    )
                )

                survival_likelihood *= (
                    gaussian_pdf(
                        sample[j],
                        survival_mean,
                        survival_std
                    )
                )

            prior_survival = km_prior.get(
                time,
                1e-6
            )

            prior_event = max(
                1.0 - prior_survival,
                1e-6
            )

            numerator = (
                event_likelihood
                * prior_event
            )

            denominator = (
                numerator
                +
                survival_likelihood
                * prior_survival
            )

            posterior_event = (
                numerator /
                max(
                    denominator,
                    1e-12
                )
            )

            score = posterior_event

            if score >= 0.5:

                best_time = time
                break

            if score > best_score:

                best_score = score

        predictions.append(
            best_time
        )

    return np.asarray(
        predictions
    )


def mean_relative_error(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    mask = y_true != 0

    return np.mean(
        np.abs(
            (
                y_true[mask]
                -
                y_pred[mask]
            )
            /
            y_true[mask]
        )
    ) * 100


def main():

    df = load_real(".")

    print(
        f"Dataset: {len(df)} events, "
        f"{df['BIRD'].nunique()} satellites"
    )

    X = df[
        FEATURES
    ].values

    y = df[
        "tte"
    ].values

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

        X_train = X[
            train_idx
        ]

        X_test = X[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        km_prior = kaplan_meier_prior(
            y_train
        )

        distributions = fit_feature_distributions(
            X_train_scaled,
            y_train
        )

        train_pred = predict_tte(
            X_train_scaled,
            distributions,
            km_prior
        )

        test_pred = predict_tte(
            X_test_scaled,
            distributions,
            km_prior
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

    mae = np.mean(
        np.abs(
            test_actual
            -
            test_predicted
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                test_actual
                -
                test_predicted
            ) ** 2
        )
    )

    print()
    print(
        "Modified Naive Bayes - "
        "10-fold cross-validation"
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
        f"Test MAE  : "
        f"{mae:.2f} days"
    )

    print(
        f"Test RMSE : "
        f"{rmse:.2f} days"
    )


if __name__ == "__main__":
    main()