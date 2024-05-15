import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
    median_absolute_error,
)


def run_regression(data):
    # Preprocess data
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Support Vector Regression": SVR(),
        "Ridge Regression": Ridge(),
    }

    results = []
    unique_metrics = []

    for name, reg in regressors.items():
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        result = {
            "Algorithm": name,
            "MAE": mae,
            "MSE": mse,
            "R-Squared": r2,
            "MedAE": medae,
        }
        results.append(result)

    unique_metrics = list(result.keys())
    unique_metrics.remove("Algorithm")  # Remove "Algorithm" key
    print(unique_metrics)
    unique_metrics = sorted(unique_metrics)

    return results, unique_metrics
