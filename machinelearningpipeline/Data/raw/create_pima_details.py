"""Create a details file mapping each row to a patient ID and include all columns.

This script ensures `pima_diabetes_synthetic.csv` exists (calling the generator
if necessary) and writes `pima_diabetes_synthetic_details.csv` with columns:

- Row: 1-indexed row number
- PatientID: PID0001, PID0002, ...
- ... all original columns from the CSV

Run: `python create_pima_details.py`
"""
from pathlib import Path
import csv
import runpy


ROOT = Path(__file__).parent
SRC_CSV = ROOT / "pima_diabetes_synthetic.csv"
DETAILS_CSV = ROOT / "pima_diabetes_synthetic_details.csv"


def ensure_source_csv():
    if SRC_CSV.exists():
        print(f"Found existing {SRC_CSV}")
        return
    print(f"{SRC_CSV} not found — generating synthetic data now")
    # run the generator script in this folder
    runpy.run_path(str(ROOT / "generate_pima_synthetic.py"), run_name="__main__")


def make_details():
    ensure_source_csv()
    with SRC_CSV.open("r", newline="") as inf, DETAILS_CSV.open("w", newline="") as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)
        header = next(reader)
        out_header = ["Row", "PatientID"] + header
        writer.writerow(out_header)
        for i, row in enumerate(reader, start=1):
            pid = f"PID{i:04d}"
            writer.writerow([i, pid] + row)

    print(f"Wrote details to {DETAILS_CSV}")


if __name__ == "__main__":
    make_details()
