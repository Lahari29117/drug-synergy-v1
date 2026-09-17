from pathlib import Path

import numpy as np
import pandas as pd

# Makes the same starter dataset each time you run this script.
rng = np.random.default_rng(42)

# Basic made-up drug descriptors for a prototype.
drugs = {
    "DrugA": {"molecular_weight": 350, "logp": 1.2},
    "DrugB": {"molecular_weight": 420, "logp": 2.1},
    "DrugC": {"molecular_weight": 280, "logp": 0.8},
    "DrugD": {"molecular_weight": 510, "logp": 3.0},
    "DrugE": {"molecular_weight": 390, "logp": 1.7},
}

# Made-up cell-line information for a prototype.
cell_lines = {
    "A549": 0.82,
    "MCF7": 0.64,
    "HCT116": 0.91,
    "PC3": 0.73,
}

# Some drug pairs are assigned stronger simulated synergy.
pair_bonus = {
    frozenset({"DrugA", "DrugB"}): 9.0,
    frozenset({"DrugA", "DrugD"}): 6.0,
    frozenset({"DrugB", "DrugE"}): 7.0,
    frozenset({"DrugC", "DrugD"}): 4.0,
}

rows = []

for _ in range(300):
    chosen_drugs = rng.choice(list(drugs.keys()), size=2, replace=False)
    drug_a, drug_b = sorted(chosen_drugs)

    cell_line = rng.choice(list(cell_lines.keys()))

    # Drug doses in micromolar (µM), spread from 0.01 to 10.
    dose_a_um = round(10 ** rng.uniform(-2, 1), 4)
    dose_b_um = round(10 ** rng.uniform(-2, 1), 4)

    a = drugs[drug_a]
    b = drugs[drug_b]
    growth_rate = cell_lines[cell_line]

    # This formula creates an artificial continuous synergy score.
    # It exists only so our first ML pipeline has patterns to learn.
    dose_balance = 3 - abs(np.log10(dose_a_um) - np.log10(dose_b_um)) * 2
    simulated_synergy = (
        pair_bonus.get(frozenset({drug_a, drug_b}), 0)
        + dose_balance
        + growth_rate * 5
        + (a["logp"] + b["logp"]) * 0.8
        - (a["molecular_weight"] + b["molecular_weight"]) * 0.004
        + rng.normal(0, 1.5)
    )

    rows.append(
        {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "cell_line": cell_line,
            "drug_a_molecular_weight": a["molecular_weight"],
            "drug_b_molecular_weight": b["molecular_weight"],
            "drug_a_logp": a["logp"],
            "drug_b_logp": b["logp"],
            "dose_a_um": dose_a_um,
            "dose_b_um": dose_b_um,
            "cell_line_growth_rate": growth_rate,
            "synergy_score": round(simulated_synergy, 3),
        }
    )

data = pd.DataFrame(rows)

output_path = Path("data/starter_synergy_data.csv")
output_path.parent.mkdir(exist_ok=True)
data.to_csv(output_path, index=False)

print(f"Created {output_path} with {len(data)} rows and {len(data.columns)} columns.")
print(data.head())
