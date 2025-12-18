import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data_processing import Data_processing   # ✅ corrected import

# --------------------------------------------------
# Base paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

def train_models():
    """
    Train multiple regression models and save them to disk
    """
    # Ensure models directory exists
    MODEL_DIR.mkdir(exist_ok=True)

    # Load processed data
    X_train, X_test, y_train, y_test = Data_processing()

    # Model dictionary
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Save model
        model_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)

    return trained_models, X_test, y_test


if __name__ == "__main__":
    train_models()
