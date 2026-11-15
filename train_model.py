
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger le dataset
df = pd.read_csv(
    "dataset/network-slicing.csv",
    low_memory=False
)

print("Dataset charge :", df.shape)

# 2. Supprimer les doublons exacts
df = df.drop_duplicates()

# 3. Garder un echantillon equilibre
# 20 000 exemples pour chaque classe
df = (
    df.groupby("Slice Type (Output)", group_keys=False)
    .sample(n=20000, random_state=42)
)

print("Donnees utilisees :", len(df))

# 4. Choisir les variables
features = [
    "Use CaseType (Input 1)",
    "Technology Supported (Input 3)",
    "Packet Loss Rate (Reliability)",
    "Packet Delay Budget (Latency)"
]

X = df[features]
y = df["Slice Type (Output)"]

# 5. Preparer les colonnes textuelles
categorical = [
    "Use CaseType (Input 1)",
    "Technology Supported (Input 3)",
    "Packet Delay Budget (Latency)"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        )
    ],
    remainder="passthrough"
)

# 6. Creer le modele Random Forest
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Diviser les donnees
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Entrainement
print("\nEntrainement du modele IA...")
model.fit(X_train, y_train)

# 9. Evaluation
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nEntrainement termine !")
print("Accuracy :", round(accuracy * 100, 2), "%")

print("\nRapport de classification :")
print(classification_report(y_test, predictions))

# 10. Sauvegarder le modele
joblib.dump(model, "model_5g.pkl")

print("\nModele sauvegarde : model_5g.pkl")
