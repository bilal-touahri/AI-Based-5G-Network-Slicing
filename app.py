
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart 5G | AI Network Slicing",
    page_icon="📡",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# ==========================================
# 2. CHARGEMENT DU MODELE IA
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model_5g.pkl")


model = load_model()

# ==========================================
# 3. CHARGEMENT DES PROFILS CONNUS
# ==========================================

FEATURES = [
    "Use CaseType (Input 1)",
    "Technology Supported (Input 3)",
    "Packet Loss Rate (Reliability)",
    "Packet Delay Budget (Latency)"
]


@st.cache_data
def load_known_profiles():
    df = pd.read_csv(
        BASE_DIR / "dataset" / "network-slicing.csv",
        low_memory=False
    )

    return df[FEATURES].drop_duplicates()


known_profiles = load_known_profiles()

# ==========================================
# 4. SCENARIOS
# ==========================================

SCENARIOS = {
    "📱 Streaming / Smartphone": {
        "use_case": "Smartphone",
        "technology": "LTE/5G",
        "packet_loss": 0.01,
        "latency": "<50ms"
    },
    "🚗 Transport intelligent": {
        "use_case": "Smart Transportation",
        "technology": "IoT (LTE-M, NB-IoT)",
        "packet_loss": 0.01,
        "latency": "<50ms"
    },
    "🌐 Internet des objets": {
        "use_case": "IoT Devices",
        "technology": "IoT (LTE-M, NB-IoT)",
        "packet_loss": 0.01,
        "latency": "<300ms"
    }
}

DEFAULT = SCENARIOS["📱 Streaming / Smartphone"]

for key, value in DEFAULT.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================
# 5. INTERFACE PRINCIPALE
# ==========================================

st.title("📡 Smart 5G Network Slicing")

st.subheader(
    "Classification intelligente des services 5G"
)

st.caption(
    "Prototype pédagogique basé sur Random Forest"
)

st.divider()

# ==========================================
# 6. SCENARIOS INTERACTIFS
# ==========================================

st.subheader("🎮 Scénarios interactifs")

cols = st.columns(3)

for col, (name, values) in zip(cols, SCENARIOS.items()):
    with col:
        if st.button(name, use_container_width=True):
            for key, value in values.items():
                st.session_state[key] = value

            st.rerun()

st.divider()

# ==========================================
# 7. PARAMETRES DU SERVICE
# ==========================================

st.subheader("⚙️ Paramètres du service")

left, right = st.columns(2)

with left:
    st.selectbox(
        "Use Case",
        [
            "Smartphone",
            "AR/VR/Gaming",
            "Healthcare",
            "Smart Transportation",
            "Industry 4.0",
            "IoT Devices",
            "Smart City & Home",
            "Public Safety/E911"
        ],
        key="use_case"
    )

    st.selectbox(
        "Technologie",
        [
            "LTE/5G",
            "IoT (LTE-M, NB-IoT)"
        ],
        key="technology"
    )

with right:
    st.selectbox(
        "Packet Loss Rate",
        [0.01, 0.001, 0.000001],
        format_func=lambda x: f"{x:.6f}",
        key="packet_loss"
    )

    st.selectbox(
        "Packet Delay Budget",
        ["<10ms", "<50ms", "<300ms"],
        key="latency"
    )

st.divider()

# ==========================================
# 8. PREDICTION IA
# ==========================================

if st.button(
    "🚀 Lancer la prédiction IA",
    type="primary",
    use_container_width=True
):

    # Préparer les données d'entrée
    input_data = pd.DataFrame([{
        "Use CaseType (Input 1)":
            st.session_state.use_case,

        "Technology Supported (Input 3)":
            st.session_state.technology,

        "Packet Loss Rate (Reliability)":
            st.session_state.packet_loss,

        "Packet Delay Budget (Latency)":
            st.session_state.latency
    }])

    # ======================================
    # 9. DETECTION DES PROFILS INCONNUS
    # ======================================

    matches = (
        (
            known_profiles["Use CaseType (Input 1)"]
            == st.session_state.use_case
        )
        &
        (
            known_profiles["Technology Supported (Input 3)"]
            == st.session_state.technology
        )
        &
        (
            (
                known_profiles["Packet Loss Rate (Reliability)"]
                - st.session_state.packet_loss
            ).abs() < 1e-9
        )
        &
        (
            known_profiles["Packet Delay Budget (Latency)"]
            == st.session_state.latency
        )
    )

    profile_known = matches.any()

    if profile_known:
        st.info(
            "✅ Profil connu : cette combinaison "
            "existe dans le dataset."
        )
    else:
        st.warning(
            "⚠️ Profil inconnu : cette combinaison "
            "n'existe pas dans le dataset. "
            "La prédiction est exploratoire "
            "et non validée."
        )

    # ======================================
    # 10. RESULTAT DE LA PREDICTION
    # ======================================

    prediction = model.predict(input_data)[0]

    scores = model.predict_proba(input_data)[0]

    descriptions = {
        "eMBB": (
            "Enhanced Mobile Broadband : "
            "services nécessitant un haut débit."
        ),

        "URLLC": (
            "Ultra-Reliable Low-Latency Communications : "
            "services nécessitant une faible latence "
            "et une haute fiabilité."
        ),

        "mMTC": (
            "Massive Machine-Type Communications : "
            "connexion massive d'objets IoT."
        )
    }

    st.divider()

    st.subheader(
        "🎯 Résultat de l'intelligence artificielle"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Slice prédite",
            prediction
        )

    with col2:
        st.metric(
            "Score du modèle",
            f"{max(scores) * 100:.1f}%"
        )

    st.success(
        descriptions[prediction]
    )

    # ======================================
    # 11. GRAPHIQUE DES SCORES
    # ======================================

    result_df = pd.DataFrame({
        "Slice": model.classes_,
        "Score": scores
    }).set_index("Slice")

    st.subheader(
        "📊 Scores par type de slice"
    )

    st.bar_chart(
        result_df,
        y="Score"
    )

    st.caption(
        "Les scores représentent les sorties "
        "du modèle et ne garantissent pas "
        "la fiabilité sur un réseau réel."
    )


# ==========================================
# FEATURE IMPORTANCE - TOP 10
# ==========================================

import matplotlib.pyplot as plt

st.divider()
st.subheader("🧠 Feature Importance — Top 10")

st.write(
    "Les 10 caractéristiques les plus importantes "
    "pour le modèle Random Forest."
)

preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Variable": feature_names,
    "Importance": importances
})

top10 = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(10)
    .sort_values("Importance", ascending=True)
    .copy()
)

# Simplifier les noms
top10["Variable"] = (
    top10["Variable"]
    .str.replace("categorical__", "", regex=False)
    .str.replace("remainder__", "", regex=False)
    .str.replace(
        "Use CaseType (Input 1)_",
        "Service : ",
        regex=False
    )
    .str.replace(
        "Technology Supported (Input 3)_",
        "Techno : ",
        regex=False
    )
    .str.replace(
        "Packet Delay Budget (Latency)_",
        "Latence : ",
        regex=False
    )
    .str.replace(
        "Packet Loss Rate (Reliability)",
        "Packet Loss",
        regex=False
    )
)

# Graphique horizontal
fig, ax = plt.subplots(figsize=(11, 6))

ax.barh(
    top10["Variable"],
    top10["Importance"] * 100,
    color="#38BDF8"
)

ax.set_xlabel("Importance (%)")
ax.set_ylabel("")
ax.set_title("Top 10 des caractéristiques importantes")

# Afficher les pourcentages
for i, value in enumerate(top10["Importance"]):
    ax.text(
        value * 100 + 0.2,
        i,
        f"{value * 100:.1f}%",
        va="center",
        fontsize=9
    )

ax.set_xlim(
    0,
    max(top10["Importance"]) * 115
)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Importance globale des variables selon "
    "Random Forest. Le total du Top 10 "
    "n'est pas nécessairement égal à 100%."
)


# ==========================================
# 12. LIMITES SCIENTIFIQUES
# ==========================================

st.divider()

st.warning(
    "Limite scientifique : le dataset contient "
    "seulement 11 profils distincts pour les "
    "quatre variables utilisées. "
    "Ce prototype illustre la classification "
    "des slices mais ne contrôle pas "
    "un réseau 5G réel."
)
