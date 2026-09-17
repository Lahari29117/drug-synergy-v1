from pathlib import Path

import joblib
import pandas as pd

# Load the full saved pipeline:
# preprocessing rules + trained Random Forest.
model_path = Path("models/synergy_pipeline.joblib")
pipeline = joblib.load(model_path)

# One new DrugComb-style experiment to predict.
# These column names must exactly match the training data.
new_experiment = pd.DataFrame(
    [
        {
            "drug_a": "DrugA",
            "drug_b": "DrugB",
            "cell_line": "A549",
            "drug_a_molecular_weight": 350,
            "drug_b_molecular_weight": 420,
            "drug_a_logp": 1.2,
            "drug_b_logp": 2.1,
            "dose_a_um": 0.5,
            "dose_b_um": 0.5,
            "cell_line_growth_rate": 0.82,
        }
    ]
)

predicted_score = pipeline.predict(new_experiment)[0]

print("Input experiment:")
print(new_experiment.to_string(index=False))
print(f"\nPredicted synergy score: {predicted_score:.3f}")