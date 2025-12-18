# src/data_ingestion.py
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_raw_data():
    path = BASE_DIR / "data" / "raw" / "train_energy_data.csv"
    return pd.read_csv(path)
