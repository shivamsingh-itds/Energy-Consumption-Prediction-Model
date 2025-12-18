import pandas as pd
from math import sqrt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.train_model import train_models

def evaluate_models():
    trained_models, X_test, y_test = train_models()

    results = []

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "R2 Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": sqrt(mean_squared_error(y_test, y_pred))
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    results_df = evaluate_models()
    print(results_df)
