import pandas as pd

def load_telemetry():
    return pd.read_parquet("data/telemetry_clean.parquet")
