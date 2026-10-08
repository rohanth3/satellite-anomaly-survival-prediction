"""
Satellite Anomaly Prediction Pipeline
Comprehensive Benchmark: Reproduction of Stanford CS229 paper baseline models
plus Proposed Modern Survival-ML Extensions.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from src.linear_regression import load_real, FEATURES, mean_relative_error
from src.naive_bayes import kaplan_meier_prior, fit_feature_distributions, predict_tte
from src.hybrid_ensemble import HybridSurvivalEnsemble


class ModifiedNaiveBayesModel:
    """Wrapper class for Modified Naive Bayes with Kaplan-Meier survival prior."""
    def __init__(self, features):
        self.features = features
        self.km_prior = None
        self.distributions = None

    def fit(self, X, y):
        self.km_prior = kaplan_meier_prior(y)
        self.distributions = fit_feature_distributions(X, y)
        return self

    def predict(self, X):
        return predict_tte(X, self.distributions, self.km_prior)


def run_pipeline(datadir=".", out_models="models", out_results="results"):
    os.makedirs(out_models, exist_ok=True)
    os.makedirs(out_results, exist_ok=True)

    print("=" * 72)
    print("SATELLITE ANOMALY TIME-TO-EVENT PREDICTION PIPELINE")
    print("Paper Reproduction Baseline + Proposed Modern Extensions")
    print("=" * 72)

    df = load_real(datadir)
    print(f"[pipeline] Loaded dataset: {len(df)} events across {df['BIRD'].nunique()} satellites")
    print(f"[pipeline] Features used: {FEATURES}")

    X = df[FEATURES].values
    y = df["tte"].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    models_info = [
        ("Linear Regression", "Paper Baseline", True),
        ("Support Vector Regression", "Paper Baseline", True),
        ("Modified Naive Bayes", "Paper Baseline", True),
        ("Gradient Boosting (L1)", "Proposed Extension", False),
        ("Hybrid Survival Ensemble", "Proposed Extension", False),
    ]

    results = {
        name: {
            "type": mtype,
            "train_mre": [],
            "test_mre": [],
            "test_actual": [],
            "test_pred": []
        }
        for name, mtype, _ in models_info
    }

    print("\n[pipeline] Running 10-Fold Cross-Validation across all models...")
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 1. Linear Regression
        lr = LinearRegression().fit(X_train_scaled, y_train)
        p_tr = lr.predict(X_train_scaled)
        p_te = lr.predict(X_test_scaled)
        results["Linear Regression"]["train_mre"].append(mean_relative_error(y_train, p_tr))
        results["Linear Regression"]["test_mre"].append(mean_relative_error(y_test, p_te))
        results["Linear Regression"]["test_actual"].extend(y_test)
        results["Linear Regression"]["test_pred"].extend(p_te)

        # 2. SVR
        svr = SVR(C=10.0, epsilon=0.1, kernel="rbf").fit(X_train_scaled, y_train)
        p_tr = svr.predict(X_train_scaled)
        p_te = svr.predict(X_test_scaled)
        results["Support Vector Regression"]["train_mre"].append(mean_relative_error(y_train, p_tr))
        results["Support Vector Regression"]["test_mre"].append(mean_relative_error(y_test, p_te))
        results["Support Vector Regression"]["test_actual"].extend(y_test)
        results["Support Vector Regression"]["test_pred"].extend(p_te)

        # 3. Modified Naive Bayes
        nb = ModifiedNaiveBayesModel(FEATURES).fit(X_train_scaled, y_train)
        p_tr = nb.predict(X_train_scaled)
        p_te = nb.predict(X_test_scaled)
        results["Modified Naive Bayes"]["train_mre"].append(mean_relative_error(y_train, p_tr))
        results["Modified Naive Bayes"]["test_mre"].append(mean_relative_error(y_test, p_te))
        results["Modified Naive Bayes"]["test_actual"].extend(y_test)
        results["Modified Naive Bayes"]["test_pred"].extend(p_te)

        # 4. Gradient Boosting (L1 loss)
        gbr = GradientBoostingRegressor(
            loss="absolute_error",
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        ).fit(X_train, y_train)
        p_tr = np.clip(gbr.predict(X_train), 1, 365)
        p_te = np.clip(gbr.predict(X_test), 1, 365)
        results["Gradient Boosting (L1)"]["train_mre"].append(mean_relative_error(y_train, p_tr))
        results["Gradient Boosting (L1)"]["test_mre"].append(mean_relative_error(y_test, p_te))
        results["Gradient Boosting (L1)"]["test_actual"].extend(y_test)
        results["Gradient Boosting (L1)"]["test_pred"].extend(p_te)

        # 5. Hybrid Survival Ensemble (GBR + KM Bayes)
        ens = HybridSurvivalEnsemble(alpha=0.7).fit(X_train, y_train)
        p_tr = ens.predict(X_train)
        p_te = ens.predict(X_test)
        results["Hybrid Survival Ensemble"]["train_mre"].append(mean_relative_error(y_train, p_tr))
        results["Hybrid Survival Ensemble"]["test_mre"].append(mean_relative_error(y_test, p_te))
        results["Hybrid Survival Ensemble"]["test_actual"].extend(y_test)
        results["Hybrid Survival Ensemble"]["test_pred"].extend(p_te)

    # Summarize Metrics
    summary_rows = []
    for model_name, data in results.items():
        actual = np.array(data["test_actual"])
        pred = np.array(data["test_pred"])
        mae = mean_absolute_error(actual, pred)
        medae = median_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        train_mre = np.mean(data["train_mre"])
        test_mre = np.mean(data["test_mre"])

        summary_rows.append({
            "Category": data["type"],
            "Model": model_name,
            "Train MRE (%)": round(train_mre, 1),
            "Test MRE (%)": round(test_mre, 1),
            "Test MAE (days)": round(mae, 2),
            "Test MedAE (days)": round(medae, 2),
            "Test RMSE (days)": round(rmse, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    metrics_path = os.path.join(out_results, "metrics_summary.csv")
    summary_df.to_csv(metrics_path, index=False)
    print("\n" + "=" * 72)
    print("EXPERIMENTAL EVALUATION SUMMARY (10-Fold Cross-Validation)")
    print("=" * 72)
    print(summary_df.to_string(index=False))

    # Save Predictions Comparison CSV
    pred_df = pd.DataFrame({
        "Actual_TTE": results["Linear Regression"]["test_actual"],
        "LR_Predicted": results["Linear Regression"]["test_pred"],
        "SVR_Predicted": results["Support Vector Regression"]["test_pred"],
        "NaiveBayes_Predicted": results["Modified Naive Bayes"]["test_pred"],
        "GBR_Predicted": results["Gradient Boosting (L1)"]["test_pred"],
        "HybridEnsemble_Predicted": results["Hybrid Survival Ensemble"]["test_pred"],
    })
    pred_path = os.path.join(out_results, "predictions_comparison.csv")
    pred_df.to_csv(pred_path, index=False)

    # Train and Save Full Models
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X)

    full_lr = LinearRegression().fit(X_full_scaled, y)
    full_svr = SVR(C=10.0, epsilon=0.1, kernel="rbf").fit(X_full_scaled, y)
    full_nb = ModifiedNaiveBayesModel(FEATURES).fit(X_full_scaled, y)
    full_gbr = GradientBoostingRegressor(loss="absolute_error", n_estimators=80, max_depth=3, random_state=42).fit(X, y)
    full_ens = HybridSurvivalEnsemble(alpha=0.7).fit(X, y)

    joblib.dump(scaler_full, os.path.join(out_models, "scaler.joblib"))
    joblib.dump(full_lr, os.path.join(out_models, "linear_regression.joblib"))
    joblib.dump(full_svr, os.path.join(out_models, "svr.joblib"))
    joblib.dump(full_nb, os.path.join(out_models, "modified_naive_bayes.joblib"))
    joblib.dump(full_gbr, os.path.join(out_models, "gradient_boosting.joblib"))
    joblib.dump(full_ens, os.path.join(out_models, "hybrid_ensemble.joblib"))
    print(f"\n[pipeline] Saved 5 trained models and scaler to {out_models}/")

    # Generate Visualizations
    generate_plots(df, results, full_gbr, out_results)
    print(f"[pipeline] Generated updated plots in {out_results}/")
    print("=" * 72)


def generate_plots(df, results, full_gbr, out_results):
    sns.set_theme(style="whitegrid")

    # 1. 2x2 Scatter Plots (Predicted TTE vs True TTE with y = x line)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_models = [
        ("Support Vector Regression", "Paper Baseline: SVR", axes[0, 0]),
        ("Modified Naive Bayes", "Paper Baseline: Modified Naive Bayes", axes[0, 1]),
        ("Gradient Boosting (L1)", "Proposed Extension: Gradient Boosting (L1)", axes[1, 0]),
        ("Hybrid Survival Ensemble", "Proposed Extension: Hybrid Survival Ensemble", axes[1, 1]),
    ]

    for m_key, title, ax in plot_models:
        y_true = np.array(results[m_key]["test_actual"])
        y_pred = np.array(results[m_key]["test_pred"])
        ax.scatter(y_true, y_pred, color="#e67e22", edgecolors="#d35400", alpha=0.7, s=32, label="Predictions")
        ax.plot([0, 40], [0, 40], "k--", linewidth=1.5, label="y = x (Perfect Fit)")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 32)
        ax.set_xlabel("True TTE (days)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted TTE (days)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_results, "predictions_scatter.png"), dpi=300)
    plt.close()

    # 2. Performance Comparison Bar Charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    model_keys = list(results.keys())
    short_names = ["Linear Reg", "SVR", "KM Bayes", "GBR (L1)", "Hybrid Ens"]
    palette = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    mae_vals = [mean_absolute_error(results[m]["test_actual"], results[m]["test_pred"]) for m in model_keys]
    rmse_vals = [np.sqrt(mean_squared_error(results[m]["test_actual"], results[m]["test_pred"])) for m in model_keys]
    mre_vals = [np.mean(results[m]["test_mre"]) for m in model_keys]

    # MAE
    axes[0].bar(short_names, mae_vals, color=palette, alpha=0.85, edgecolor="black")
    axes[0].set_title("Test Mean Absolute Error (MAE)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Days", fontsize=11)
    axes[0].set_xticks(range(len(short_names)))
    axes[0].set_xticklabels(short_names, rotation=20)
    for i, yval in enumerate(mae_vals):
        axes[0].text(i, yval + 0.3, f"{yval:.2f} d", ha="center", va="bottom", fontweight="bold")

    # RMSE
    axes[1].bar(short_names, rmse_vals, color=palette, alpha=0.85, edgecolor="black")
    axes[1].set_title("Test Root Mean Squared Error (RMSE)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Days", fontsize=11)
    axes[1].set_xticks(range(len(short_names)))
    axes[1].set_xticklabels(short_names, rotation=20)
    for i, yval in enumerate(rmse_vals):
        axes[1].text(i, yval + 0.5, f"{yval:.2f} d", ha="center", va="bottom", fontweight="bold")

    # MRE
    axes[2].bar(short_names, mre_vals, color=palette, alpha=0.85, edgecolor="black")
    axes[2].set_title("Test Mean Relative Error (MRE)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Percentage (%)", fontsize=11)
    axes[2].set_xticks(range(len(short_names)))
    axes[2].set_xticklabels(short_names, rotation=20)
    for i, yval in enumerate(mre_vals):
        axes[2].text(i, yval + 6.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_results, "model_comparison.png"), dpi=300)
    plt.close()

    # 3. Feature Importance Plot (from Gradient Boosting)
    importances = full_gbr.feature_importances_
    plt.figure(figsize=(8, 4.5))
    feat_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
    feat_series.plot(kind="barh", color="#16a085", edgecolor="black")
    plt.title("Feature Importance in Time-to-Event Prediction (Gradient Boosting)", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Importance Weight", fontweight="bold")
    for i, val in enumerate(feat_series):
        plt.text(val + 0.01, i, f"{val:.3f}", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_results, "feature_importance.png"), dpi=300)
    plt.close()

    # 4. Kaplan-Meier Survival Curve
    tte = df["tte"].values
    times = np.sort(np.unique(tte))
    km_prior = kaplan_meier_prior(tte)
    survival_probs = [km_prior[t] for t in times]

    plt.figure(figsize=(8, 5))
    plt.step(times, survival_probs, where="post", color="#8e44ad", linewidth=2.5, label="Kaplan-Meier Survival Function S(t)")
    plt.title("Kaplan-Meier Empirical Survival Probability S(t)", fontsize=13, fontweight="bold")
    plt.xlabel("Time to Next Anomaly t (Days)", fontsize=11, fontweight="bold")
    plt.ylabel("Survival Probability S(t)", fontsize=11, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_results, "kaplan_meier_survival_curve.png"), dpi=300)
    plt.close()

    # 5. TTE Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["tte"], bins=30, kde=True, color="#2980b9", edgecolor="black")
    plt.title("Distribution of Time-To-Event (Days)", fontsize=13, fontweight="bold")
    plt.xlabel("Time to Next Anomaly (Days)", fontsize=11, fontweight="bold")
    plt.ylabel("Frequency", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_results, "tte_distribution.png"), dpi=300)
    plt.close()

    # 6. Feature Correlation Heatmap
    corr_cols = ["tte", "perigee", "inclination", "ssn", "month"]
    corr_df = df[corr_cols].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_results, "feature_correlations.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_pipeline()
