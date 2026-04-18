from datetime import datetime
from pathlib import Path

import pandas as pd


def prepare_data():
    project_root = Path(__file__).parent.parent
    raw_csv = project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_parquet = project_root / "data" / "raw" / "customer_features.parquet"

    # Read raw data
    df = pd.read_csv(raw_csv)

    # Feast requires an event_timestamp for point-in-time joins
    # In a real system this would be the time the feature values were recorded.
    # Here we simulate that these are fresh features recorded ~now.
    df["event_timestamp"] = pd.to_datetime(datetime.now())

    # The raw CSV has ' ' (space) for TotalCharges for new customers.
    # Convert to numeric, replace blanks with 0.0.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(" ", ""), errors="coerce"
    ).fillna(0.0)

    # Save to parquet. Parquet is the standard offline store format for Feast.
    df.to_parquet(output_parquet, index=False)
    print(f"✅ Prepared Feast offline data at {output_parquet}")
    print(f"   Shape: {df.shape}")


if __name__ == "__main__":
    prepare_data()
