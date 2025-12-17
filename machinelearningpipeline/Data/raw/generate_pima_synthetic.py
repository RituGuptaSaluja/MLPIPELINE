"""Generate a synthetic Pima Indians Diabetes dataset CSV.

This script creates `pima_diabetes_synthetic.csv` with 768 rows and the
standard Pima column names:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

The data is synthetic and intended for development and testing only.
"""
import csv
import random
import math
from pathlib import Path


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def generate_row(rng):
    # Generate features with plausible ranges/distributions
    pregnancies = rng.poisson(1) if hasattr(rng, 'poisson') else int(rng.expovariate(1/1.5))

    # Glucose: mean 120, std 30, clamp [40, 200]
    glucose = max(40, min(200, int(rng.gauss(120, 30))))

    # BloodPressure: mean 70, std 12, clamp [24, 120]
    blood_pressure = max(24, min(120, int(rng.gauss(70, 12))))

    # SkinThickness: 0-99, many zeros
    skin_thickness = int(max(0, min(99, rng.gauss(20, 15))))

    # Insulin: positive skew, many zeros
    insulin = int(max(0, min(999, rng.gauss(80, 90))))

    # BMI: mean 32, std 6, clamp [10, 70]
    bmi = round(max(10, min(70, rng.gauss(32, 6))), 1)

    # DiabetesPedigreeFunction: small positive float ~ [0.01,2.5]
    dpf = round(max(0.01, min(2.5, rng.gauss(0.47, 0.3))), 3)

    # Age: 21-90
    age = int(max(21, min(90, rng.gauss(33, 11))))

    # Create a simple logistic model for outcome probability
    # Heavier weights for glucose, bmi, age, pregnancies
    linear = (
        0.03 * glucose
        + 0.06 * bmi
        + 0.02 * age
        + 0.15 * pregnancies
        + 0.5 * dpf
        - 12.0
    )
    prob = sigmoid(linear)
    outcome = 1 if rng.random() < prob else 0

    return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, outcome]


def main(output_path=None, n_rows=768, seed=42):
    if output_path is None:
        output_path = Path(__file__).parent / "pima_diabetes_synthetic.csv"
    else:
        output_path = Path(output_path)

    rng = random.Random(seed)

    # attach simple poisson method for pregnancies generation
    try:
        import numpy as _np

        class RNGWrapper:
            def __init__(self, seed):
                self._np = _np.random.default_rng(seed)

            def poisson(self, lam):
                return int(self._np.poisson(lam))

            def gauss(self, mu, sigma):
                return float(self._np.normal(mu, sigma))

            def random(self):
                return float(self._np.random())

            def expovariate(self, lmbda):
                return float(self._np.exponential(1.0 / lmbda))

        rng = RNGWrapper(seed)
    except Exception:
        # fall back to built-in random.Random-based behavior
        pass

    header = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for _ in range(n_rows):
            writer.writerow(generate_row(rng))

    print(f"Wrote {n_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
