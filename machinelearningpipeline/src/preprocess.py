import os
from pathlib import Path
import yaml
import pandas as pd


def load_preprocess_params():
    # Prefer params.yaml located one level above src (project root)
    params_path = Path(__file__).resolve().parent.parent / "params.yaml"
    if not params_path.exists():
        # fallback to CWD
        params_path = Path("params.yaml")

    cfg = yaml.safe_load(open(params_path))
    preprocess_cfg = cfg.get("preprocess", {})

    # allow either 'input'/'output' or 'input_path'/'output_path'
    inp = preprocess_cfg.get("input_path") or preprocess_cfg.get("input")
    out = preprocess_cfg.get("output_path") or preprocess_cfg.get("output")

    if inp is None or out is None:
        raise KeyError("preprocess params must define 'input' (or 'input_path') and 'output' (or 'output_path')")

    # resolve paths relative to params.yaml location
    base = params_path.parent
    input_path = (base / inp).resolve()
    output_path = (base / out).resolve()
    return str(input_path), str(output_path)


def preprocess(input_path, output_path):
    # Read CSV with header (default). If the file has no header, change as needed.
    data = pd.read_csv(input_path)

    outdir = os.path.dirname(output_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    inp, out = load_preprocess_params()
    preprocess(inp, out)