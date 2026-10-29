"""Print every customer record in the Telco churn dataset."""

from pathlib import Path

import pandas as pd


data_path = Path(__file__).with_name("WA_Fn-UseC_-Telco-Customer-Churn.csv")
clients = pd.read_csv(data_path)

print(f"Total clients: {len(clients):,}")
print(clients.to_string(index=False))
