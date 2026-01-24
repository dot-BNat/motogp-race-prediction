import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


FEATURE_COLS = ["RiderName", "GridPosition", "SprintFinish"]


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and create DNF flag + cleaned finish column."""
    df = pd.read_csv(csv_path)

    df["DNF_Flag"] = (df["MainRaceFinish"] == 23).astype(int)
    df["MainRaceFinishClean"] = df["MainRaceFinish"].replace(23, np.nan)

    # Drop rows where inputs are missing
    df = df.dropna(subset=["GridPosition", "SprintFinish"])

    return df


def build_preprocessor():
    """Create preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["RiderName"]),
            ("num", StandardScaler(), ["GridPosition", "SprintFinish"])
        ]
    )


def train_dnf_classifier(df: pd.DataFrame):
    """Train DNF classification model."""
    X = df[FEATURE_COLS]
    y = df["DNF_Flag"]

    preprocessor = build_preprocessor()

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"🏁 DNF Classifier Accuracy: {acc:.3f}")

    return model, X.dtypes.to_dict()


def train_finish_regressor(df: pd.DataFrame):
    """Train finishing position regressor (only non-DNF races)."""
    df_no_dnf = df[(df["DNF_Flag"] == 0) & (df["MainRaceFinishClean"].notna())]

    X = df_no_dnf[FEATURE_COLS]
    y = df_no_dnf["MainRaceFinishClean"]

    preprocessor = build_preprocessor()

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"🏎️ Regressor Performance: MAE={mae:.2f}, RMSE={rmse:.2f}")

    return model


def predict_result(
    dnf_model,
    finish_model,
    input_dtypes,
    rider_name: str,
    grid_position: int,
    sprint_finish: int,
    dnf_threshold: float = 0.5
):
    """Predict DNF probability and finishing position."""
    new_data = pd.DataFrame({
        "RiderName": [rider_name],
        "GridPosition": [grid_position],
        "SprintFinish": [sprint_finish]
    })

    # Match training dtypes
    new_data = new_data.astype(input_dtypes)

    dnf_prob = dnf_model.predict_proba(new_data)[0, 1]

    if dnf_prob > dnf_threshold:
        return {
            "DNF_Probability": dnf_prob,
            "Prediction": "Likely DNF"
        }

    finish_pred = finish_model.predict(new_data)[0]
    finish_pred = max(1, math.floor(finish_pred))

    return {
        "DNF_Probability": dnf_prob,
        "Prediction": f"Finish Position {finish_pred}",
        "FinishPosition": finish_pred
    }
