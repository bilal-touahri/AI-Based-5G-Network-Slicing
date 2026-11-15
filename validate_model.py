
import pandas as pd

df = pd.read_csv(
    "dataset/network-slicing.csv",
    low_memory=False
)

features = [
    "Use CaseType (Input 1)",
    "Technology Supported (Input 3)",
    "Packet Loss Rate (Reliability)",
    "Packet Delay Budget (Latency)"
]

# Garder uniquement les combinaisons distinctes
unique_data = df[
    features + ["Slice Type (Output)"]
].drop_duplicates()

print("Nombre total de lignes :", len(df))
print("Combinaisons uniques :", len(unique_data))

print("\nRepartition des combinaisons :")
print(unique_data["Slice Type (Output)"].value_counts())

print("\nToutes les combinaisons :")
print(unique_data.to_string(index=False))
