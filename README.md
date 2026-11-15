# 📡 AI-Based 5G Network Slicing

## Description

Ce mini-projet propose une application interactive de classification des services 5G à l'aide d'un modèle de Machine Learning basé sur l'algorithme **Random Forest**.

L'application permet de prédire le type de Network Slice adapté aux caractéristiques d'un service :

- **eMBB (Enhanced Mobile Broadband)** : services nécessitant un débit élevé, comme le streaming vidéo et les applications multimédias.
- **URLLC (Ultra-Reliable Low-Latency Communications)** : services nécessitant une faible latence et une haute fiabilité, comme certains systèmes de transport intelligent.
- **mMTC (Massive Machine-Type Communications)** : services impliquant un grand nombre d'objets connectés, comme les capteurs IoT.

Le projet utilise **Python, Scikit-learn, Pandas, Matplotlib et Streamlit**.

Il s'agit d'un prototype pédagogique de classification des services 5G, et non d'un système réel de gestion des ressources réseau.

---

## 1. Prérequis

Avant de commencer, installer :

- **Python 3.10** (version utilisée pour développer le projet).
- **Git** pour cloner le dépôt.
- **Visual Studio Code** (facultatif).

## 2. Télécharger le projet

Ouvrir PowerShell ou un terminal et exécuter :

```bash
git clone https://github.com/bilal-touahri/AI-Based-5G-Network-Slicing.git
```

Accéder ensuite au dossier du projet :

```bash
cd AI-Based-5G-Network-Slicing
```

Il est également possible de télécharger le projet directement depuis GitHub en cliquant sur **Code → Download ZIP**, puis d'extraire l'archive.

## 3. Créer un environnement virtuel

Sous Windows, exécuter :

```powershell
python -m venv .venv
```

L'environnement virtuel permet d'installer les bibliothèques nécessaires au projet sans interférer avec les autres installations Python.

## 4. Installer les dépendances

Installer les bibliothèques à partir du fichier `requirements.txt` :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En cas de connexion Internet lente :

```powershell
.\.venv\Scripts\python.exe -m pip install --default-timeout=1000 --retries 10 -r requirements.txt
```

Les principales bibliothèques utilisées sont :

- **Streamlit** : création de l'interface web interactive.
- **Scikit-learn** : entraînement et utilisation du modèle Random Forest.
- **Pandas** : manipulation et traitement des données.
- **NumPy** : calcul numérique.
- **Matplotlib** : visualisation des résultats.
- **Joblib** : sauvegarde et chargement du modèle entraîné.

## 5. Structure du projet

```text
AI-Based-5G-Network-Slicing/
│
├── dataset/
│   └── network-slicing.csv
│
├── app.py
├── train_model.py
├── validate_model.py
├── model_5g.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

**Description des fichiers :**

| Fichier | Description |
|---|---|
| `app.py` | Application interactive développée avec Streamlit |
| `train_model.py` | Entraînement et évaluation du modèle Random Forest |
| `validate_model.py` | Analyse des profils distincts du dataset |
| `model_5g.pkl` | Modèle Machine Learning entraîné et sauvegardé |
| `network-slicing.csv` | Dataset utilisé pour entraîner le modèle |
| `requirements.txt` | Liste des dépendances Python |
| `.gitignore` | Exclusion des fichiers inutiles du dépôt Git |
| `README.md` | Documentation du projet |

## 6. Entraîner le modèle (si nécessaire)

Si le fichier `model_5g.pkl` est absent, exécuter :

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Ce script permet de :

1. Charger le dataset `network-slicing.csv`.
2. Préparer les données et sélectionner les caractéristiques.
3. Diviser les observations en ensembles d'entraînement et de test.
4. Entraîner un modèle Random Forest.
5. Évaluer les prédictions.
6. Sauvegarder le modèle dans `model_5g.pkl`.

Le dataset doit être présent dans le dossier `dataset`.

## 7. Lancer l'application Streamlit

Pour démarrer l'application :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

L'application s'ouvre généralement dans le navigateur à l'adresse suivante :

http://localhost:8501

Pour arrêter le serveur, appuyer sur `Ctrl + C` dans le terminal.

## 8. Fonctionnalités de l'application

### Classification intelligente des services 5G

L'utilisateur peut renseigner les caractéristiques d'un service :

- **Use Case** : type d'application ou de service.
- **Technology Supported** : technologie réseau associée.
- **Packet Loss Rate** : taux de perte des paquets.
- **Packet Delay Budget** : budget de latence.

Le modèle Random Forest prédit ensuite l'une des trois classes : **eMBB, URLLC ou mMTC**.

### Scénarios interactifs

L'application propose trois scénarios prédéfinis :

- **Smartphone / Streaming** : illustration de la classe eMBB.
- **Transport intelligent** : illustration de la classe URLLC.
- **Internet des objets (IoT)** : illustration de la classe mMTC.

Ces scénarios permettent de tester rapidement le fonctionnement du modèle.

### Visualisation des prédictions

L'application affiche :

- La Network Slice prédite.
- Le score associé à la prédiction.
- Un graphique comparatif des scores des trois classes.

Les scores sont issus du modèle Random Forest et ne garantissent pas sa fiabilité sur des données réelles inédites.

### Détection des profils inconnus

L'application vérifie si la combinaison des paramètres sélectionnés existe parmi les profils du dataset.

- **Profil connu** : la combinaison existe dans les données utilisées.
- **Profil inconnu** : la combinaison n'existe pas dans ces données ; la prédiction est alors exploratoire et non validée.

### Feature Importance

Un graphique **Top 10 Feature Importance** présente les dix caractéristiques encodées les plus importantes pour le modèle Random Forest.

Cette visualisation permet d'identifier les variables qui contribuent le plus à la classification globale, sans expliquer précisément chaque prédiction individuelle.

## 9. Dataset et résultats

Le dataset utilisé contient **466 739 observations** et **9 colonnes**.

La répartition des classes est la suivante :

| Network Slice | Nombre d'observations |
|---|---:|
| eMBB | 125 580 |
| URLLC | 209 300 |
| mMTC | 131 859 |
| **Total** | **466 739** |

Pour accélérer l'entraînement, un échantillon équilibré de 60 000 observations a été utilisé, comprenant 20 000 observations par classe.

Le modèle a obtenu une **accuracy de 100 % sur l'ensemble de test**, composé de 12 000 observations.

Cependant, les quatre variables sélectionnées ne produisent que **11 combinaisons distinctes** dans les données disponibles.

Cette forte répétition signifie que le résultat de 100 % ne permet pas de démontrer la capacité du modèle à généraliser à des profils entièrement nouveaux.

## 10. Limites scientifiques

Ce projet constitue une démonstration académique de la classification intelligente des services 5G.

Les principales limites sont :

- Le faible nombre de profils distincts malgré un volume important d'observations.
- La forte similarité entre les données d'entraînement et de test.
- L'absence d'une validation indépendante sur des données terrain.
- L'impossibilité de garantir les performances du modèle pour des combinaisons inédites.

Le modèle ne réalise **ni allocation réelle des ressources réseau, ni création ou déploiement effectif de Network Slices dans une infrastructure 5G**.

Les prédictions sont fournies exclusivement à titre pédagogique.

---

**Projet académique — AI-Based 5G Network Slicing**