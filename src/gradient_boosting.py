import os
import sys
import joblib
import numpy as np
import pandas as pd

# Support running directly or as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import KFold

try:
    from src.linear_regression import load_real, FEATURES, mean_relative_error
except ModuleNotFoundError:
    from linear_regression import load_real, FEATURES, mean_relative_error


def main():
    df = load_real(".")
    print(f"Dataset: {len(df)} events, {df['BIRD'].nunique()} satellites")

    X = df[FEATURES].values
    y = df["tte"].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    train_relative_errors = []
    test_relative_errors = []
    test_actual = []
    test_predicted = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Gradient Boosting with L1 (absolute error) loss directly optimizes MAE
        model = GradientBoostingRegressor(
            loss="absolute_error",
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        train_pred = np.clip(model.predict(X_train), 1, 365)
        test_pred = np.clip(model.predict(X_test), 1, 365)

        train_relative_errors.append(mean_relative_error(y_train, train_pred))
        test_relative_errors.append(mean_relative_error(y_test, test_pred))

        test_actual.extend(y_test)
        test_predicted.extend(test_pred)

    test_actual = np.asarray(test_actual)
    test_predicted = np.asarray(test_predicted)

    mae = mean_absolute_error(test_actual, test_predicted)
    rmse = np.sqrt(mean_squared_error(test_actual, test_predicted))
    medae = median_absolute_error(test_actual, test_predicted)

    print()
    print("Gradient Boosting Regressor (L1 Loss) - 10-fold cross-validation")
    print(f"Train mean relative error : {np.mean(train_relative_errors):.1f}%")
    print(f"Test mean relative error  : {np.mean(test_relative_errors):.1f}%")
    print(f"Test MAE                  : {mae:.2f} days")
    print(f"Test Median AE            : {medae:.2f} days")
    print(f"Test RMSE                 : {rmse:.2f} days")

    # Fit and save final model
    os.makedirs("models", exist_ok=True)
    final_model = GradientBoostingRegressor(
        loss="absolute_error",
        n_estimators=80,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    ).fit(X, y)
    joblib.dump(final_model, os.path.join("models", "gradient_boosting.joblib"))
    print("\n[model] Saved trained model to models/gradient_boosting.joblib")


if __name__ == "__main__":
    main()
