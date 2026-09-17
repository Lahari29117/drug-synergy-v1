import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load the tabular dataset.
data_path = Path("data/starter_synergy_data.csv")
data = pd.read_csv(data_path)

# 2. Define what the model receives (features) and predicts (target).
target_column = "synergy_score"

categorical_features = [
    "drug_a",
    "drug_b",
    "cell_line",
]

numeric_features = [
    "drug_a_molecular_weight",
    "drug_b_molecular_weight",
    "drug_a_logp",
    "drug_b_logp",
    "dose_a_um",
    "dose_b_um",
    "cell_line_growth_rate",
]

feature_columns = categorical_features + numeric_features

X = data[feature_columns]
y = data[target_column]

# 3. Keep 20% of rows aside for an honest first test.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# 4. Prepare the two feature types.
# Category names such as "DrugA" become machine-readable columns.
categorical_transformer = Pipeline(
    steps=[
        ("missing_values", SimpleImputer(strategy="most_frequent")),
        ("encode_categories", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Numerical values are filled with the median if a value is missing.
numeric_transformer = Pipeline(
    steps=[
        ("missing_values", SimpleImputer(strategy="median")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numeric", numeric_transformer, numeric_features),
    ]
)

# 5. Create the Random Forest regression model.
random_forest = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

# 6. Combine preprocessing and the ML model into one pipeline.
# This prevents the API from using different encoding rules later.
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest),
    ]
)

# 7. Train the pipeline using only training data.
pipeline.fit(X_train, y_train)

# 8. Test it using data the model did not see during training.
predictions = pipeline.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel evaluation on the test set")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# 9. Save the trained pipeline.
# It contains BOTH preprocessing and the trained Random Forest.
models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

joblib.dump(pipeline, models_folder / "synergy_pipeline.joblib")

# Save test predictions so we can inspect them later.
test_results = X_test.copy()
test_results["actual_synergy_score"] = y_test
test_results["predicted_synergy_score"] = predictions
test_results.to_csv(models_folder / "test_predictions.csv", index=False)

# Save project details and metrics in a readable file.
metadata = {
    "dataset": str(data_path),
    "rows_in_dataset": len(data),
    "training_rows": len(X_train),
    "test_rows": len(X_test),
    "target_column": target_column,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "metrics": {
        "mse": round(float(mse), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 3),
    },
}

with open(models_folder / "model_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("\nSaved files:")
print("- models/synergy_pipeline.joblib")
print("- models/test_predictions.csv")
print("- models/model_metadata.json")