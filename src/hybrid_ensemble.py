import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.linear_regression import load_real, FEATURES, mean_relative_error
    from src.naive_bayes import kaplan_meier_prior, fit_feature_distributions, predict_tte
except ModuleNotFoundError:
    from linear_regression import load_real, FEATURES, mean_relative_error
    from naive_bayes import kaplan_meier_prior, fit_feature_distributions, predict_tte


class HybridSurvivalEnsemble:
    """
    Hybrid Survival Ensemble: Combines non-parametric Kaplan-Meier Bayesian Survival
    Prior with Gradient Boosted Decision Trees (L1 loss).
    """
    def __init__(self, features=FEATURES, alpha=0.7):
        self.features = features
        self.alpha = alpha  # Weight for Gradient Boosting (1 - alpha for KM Bayes)
        self.scaler = StandardScaler()
        self.gbr = GradientBoostingRegressor(
            loss="absolute_error",
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.km_prior = None
        self.distributions = None

    def fit(self, X, y):
        self.gbr.fit(X, y)
        X_scaled = self.scaler.fit_transform(X)
        self.km_prior = kaplan_meier_prior(y)
        self.distributions = fit_feature_distributions(X_scaled, y)
        return self

    def predict(self, X):
        pred_gbr = np.clip(self.gbr.predict(X), 1, 365)
        X_scaled = self.scaler.transform(X)
        pred_nb = predict_tte(X_scaled, self.distributions, self.km_prior)
        combined = self.alpha * pred_gbr + (1.0 - self.alpha) * pred_nb
        return np.clip(combined, 1, 365)


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

        ensemble = HybridSurvivalEnsemble(alpha=0.7)
        ensemble.fit(X_train, y_train)

        train_pred = ensemble.predict(X_train)
        test_pred = ensemble.predict(X_test)

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
    print("Hybrid Survival-Boosting Ensemble (GBR + KM Bayes) - 10-fold CV")
    print(f"Train mean relative error : {np.mean(train_relative_errors):.1f}%")
    print(f"Test mean relative error  : {np.mean(test_relative_errors):.1f}%")
    print(f"Test MAE                  : {mae:.2f} days")
    print(f"Test Median AE            : {medae:.2f} days")
    print(f"Test RMSE                 : {rmse:.2f} days")

    os.makedirs("models", exist_ok=True)
    final_model = HybridSurvivalEnsemble(alpha=0.7).fit(X, y)
    joblib.dump(final_model, os.path.join("models", "hybrid_ensemble.joblib"))
    print("\n[model] Saved trained model to models/hybrid_ensemble.joblib")


if __name__ == "__main__":
    main()
