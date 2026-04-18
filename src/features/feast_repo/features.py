from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# 1. Define Data Source
# This tells Feast where our offline feature data lives. We are using the parquet
# file we generated from the raw data.
customer_data = FileSource(
    path="../../../data/raw/customer_features.parquet",
    timestamp_field="event_timestamp",
)

# 2. Define Entities
# Entities map to the primary keys used in the online store to fetch features.
customer = Entity(
    name="customer",
    join_keys=["customerID"],
    value_type=ValueType.STRING,
    description="Customer identifier",
)

# 3. Define Feature Views
# Feature views logically group features. Our single feature view
# exposes the 19 raw features needed by the API's CustomerFeatures Pydantic schema.
customer_feature_view = FeatureView(
    name="customer_raw_features",
    entities=[customer],
    ttl=timedelta(days=3650),  # Features don't expire for demonstration purposes
    source=customer_data,
    schema=[
        Field(name="gender", dtype=String),
        Field(name="SeniorCitizen", dtype=Int64),
        Field(name="Partner", dtype=String),
        Field(name="Dependents", dtype=String),
        Field(name="tenure", dtype=Int64),
        Field(name="PhoneService", dtype=String),
        Field(name="MultipleLines", dtype=String),
        Field(name="InternetService", dtype=String),
        Field(name="OnlineSecurity", dtype=String),
        Field(name="OnlineBackup", dtype=String),
        Field(name="DeviceProtection", dtype=String),
        Field(name="TechSupport", dtype=String),
        Field(name="StreamingTV", dtype=String),
        Field(name="StreamingMovies", dtype=String),
        Field(name="Contract", dtype=String),
        Field(name="PaperlessBilling", dtype=String),
        Field(name="PaymentMethod", dtype=String),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
    ],
)
