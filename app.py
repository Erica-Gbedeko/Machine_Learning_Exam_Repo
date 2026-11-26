import glob
import os
import unicodedata
import warnings

import joblib
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Segmentation avec le modèle CAH", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Configuration des 4 jeux de données
#   dossier  : nom du dossier du dépôt GitHub qui contient les fichiers joblib
#   colonnes : colonnes de df, DANS LE MÊME ORDRE qu'à l'entraînement
#   defauts  : valeur pré-remplie (médiane) de chaque colonne
#   classes  : noms des classes (ceux de la cellule `noms_classes` du notebook)
#   encodees : colonnes textuelles encodées avec LabelEncoder (fichier encoders*.joblib)
#   exemples : lignes du jeu de données pour tester l'application en un clic
# ---------------------------------------------------------------------------
DATASETS = {
    "Insuffisance cardiaque": {
        "dossier": "heart_failure_clinical_records",
        "titre": "Segmentation des patients - modèle CAH",
        "description": "Saisissez les caractéristiques d'un patient pour connaître sa classe.",
        "colonnes": [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking",
            "time", "DEATH_EVENT",
        ],
        # Valeurs médianes typiques du jeu de données. Pour être exact, collez ici
        # le dictionnaire affiché par la cellule d'impression du notebook.
        "defauts": {
            "age": 60, "anaemia": 0, "creatinine_phosphokinase": 250, "diabetes": 0,
            "ejection_fraction": 38, "high_blood_pressure": 0, "platelets": 262000,
            "serum_creatinine": 1.1, "serum_sodium": 137, "sex": 1, "smoking": 0,
            "time": 115, "DEATH_EVENT": 0,
        },
        "classes": {
            0: "Patient avec CPK très élevé (profil atypique)",
            1: "Patient au profil standard",
        },
        "encodees": [],
        "exemples": [],
    },
    "Graines de blé": {
        "dossier": "Graine",
        "titre": "Segmentation des graines de blé - modèle CAH",
        "description": "Saisissez les caractéristiques d'une graine pour connaître sa classe.",
        "colonnes": [
            "area A", "perimeter", "compactness", "length of kernel",
            "width of kernel", "asymmetry coefficient", "length of kernel groove",
        ],
        "defauts": {
            "area A": 14.43, "perimeter": 14.37, "compactness": 0.8734,
            "length of kernel": 5.541, "width of kernel": 3.245,
            "asymmetry coefficient": 3.631, "length of kernel groove": 5.228,
        },
        "classes": {0: "petite graine", 1: "grosse graine"},
        "encodees": [],
        "exemples": [
            [15.26, 14.84, 0.8710, 5.763, 3.312, 2.221, 5.220],
            [14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956],
            [14.29, 14.09, 0.9050, 5.291, 3.337, 2.699, 4.825],
        ],
    },
    "Télécommunications": {
        "dossier": "Télécommunications",
        "titre": "Segmentation des clients - modèle CAH",
        "description": "Saisissez les caractéristiques d'un client pour connaître sa classe.",
        "colonnes": [
            "Call Failure", "Complains", "Subscription Length", "Charge Amount",
            "Seconds of Use", "Frequency of use", "Frequency of SMS",
            "Distinct Called Numbers", "Age Group", "Status", "Age", "Customer Value",
        ],
        "defauts": {},  # collez ici le dictionnaire des médianes (voir notebook)
        "classes": {},  # remplissez avec vos noms de classes, ex. {0: "...", 1: "..."}
        "encodees": [],
        "exemples": [],
    },
    "Produits (PriceRunner)": {
        "dossier": "pricerunner",
        "titre": "Segmentation des produits - modèle KMEANS ",
        "description": "Saisissez les caractéristiques d'un produit pour connaître sa classe.",
        "colonnes": [
            "Product ID", "Product Title", "Merchant ID", "Cluster ID",
            "Cluster Label", "Category ID", "Category Label",
        ],
        "defauts": {},
        "classes": {0: "Produit courant", 1: "Produit rare"},
        "encodees": ["Product Title", "Cluster Label", "Category Label"],
        "exemples": [],
    },
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def trouver_dossier(nom):
    """Trouve le dossier même si les accents sont encodés différemment (Mac/Linux)."""
    cible = unicodedata.normalize("NFC", nom)
    for element in os.listdir(BASE):
        chemin = os.path.join(BASE, element)
        if os.path.isdir(chemin) and unicodedata.normalize("NFC", element) == cible:
            return chemin
    return None


def trouver_fichier(dossier, motif):
    fichiers = sorted(glob.glob(os.path.join(dossier, motif)))
    return fichiers[0] if fichiers else None


@st.cache_resource
def charger(nom_dossier):
    dossier = trouver_dossier(nom_dossier)
    if dossier is None:
        return None, f"Le dossier « {nom_dossier} » est introuvable dans le dépôt."

    chemins = {
        "scaler": trouver_fichier(dossier, "*scaler*.joblib"),
        "pca": trouver_fichier(dossier, "*pca*.joblib"),
        "model": trouver_fichier(dossier, "*model*.joblib"),
        "encoders": trouver_fichier(dossier, "encoders*.joblib"),
    }
    manquants = [k for k in ("scaler", "pca", "model") if chemins[k] is None]
    if manquants:
        return None, (
            f"Fichiers joblib manquants dans « {nom_dossier} » : {', '.join(manquants)}."
        )

    objets = {
        "scaler": joblib.load(chemins["scaler"]),
        "pca": joblib.load(chemins["pca"]),
        "model": joblib.load(chemins["model"]),
        "encoders": joblib.load(chemins["encoders"]) if chemins["encoders"] else {},
    }
    return objets, None


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("Segmentation par classification hiérarchique (CAH)")

choix = st.selectbox("Jeu de données", list(DATASETS.keys()))
cfg = DATASETS[choix]

objets, erreur = charger(cfg["dossier"])
if erreur:
    st.error(erreur)
    st.stop()

scaler, pca, model, encoders = (
    objets["scaler"], objets["pca"], objets["model"], objets["encoders"],
)

st.subheader(cfg["titre"])
st.caption(cfg["description"])

colonnes = cfg["colonnes"]
cle = choix.replace(" ", "_")

# Valeurs initiales des champs (médianes), une seule fois par jeu de données
for col in colonnes:
    k = f"{cle}_{col}"
    if k in st.session_state:
        continue
    if col in cfg["encodees"] and col in encoders:
        st.session_state[k] = str(encoders[col].classes_[0])
    else:
        st.session_state[k] = float(cfg["defauts"].get(col, 0.0))


def charger_exemple():
    choisi = st.session_state.get(f"{cle}_exemple")
    if choisi and choisi.startswith("Exemple"):
        ligne = cfg["exemples"][int(choisi.split()[-1]) - 1]
        for col, val in zip(colonnes, ligne):
            st.session_state[f"{cle}_{col}"] = float(val)


# Exemples du jeu de données pour tester l'application en un clic
if cfg["exemples"]:
    st.selectbox(
        "Charger un exemple du jeu de données",
        ["—"] + [f"Exemple {i + 1}" for i in range(len(cfg["exemples"]))],
        key=f"{cle}_exemple",
        on_change=charger_exemple,
    )

# Un champ de saisie par variable, dans l'ordre des colonnes de df
colonnes_ui = st.columns(3)
for i, col in enumerate(colonnes):
    k = f"{cle}_{col}"
    with colonnes_ui[i % 3]:
        if col in cfg["encodees"] and col in encoders:
            options = [str(c) for c in encoders[col].classes_]
            if len(options) <= 500:
                st.selectbox(col, options, key=k)
            else:
                st.text_input(col, key=k)
        else:
            st.number_input(col, key=k, format="%.4f")

# Prédiction : la classe du centroïde le plus proche (NearestCentroid)
if st.button("Affecter à une classe", type="primary"):
    valeurs, ok = [], True
    for col in colonnes:
        v = st.session_state[f"{cle}_{col}"]
        if col in cfg["encodees"] and col in encoders:
            try:
                v = encoders[col].transform([str(v)])[0]
            except ValueError:
                st.error(
                    f"« {v} » ne figure pas dans les données d'entraînement "
                    f"de la colonne {col}."
                )
                ok = False
                break
        valeurs.append(float(v))

    if ok:
        x_new = scaler.transform(np.array(valeurs, dtype=float).reshape(1, -1))
        classe = int(model.predict(x_new)[0])
        nom = cfg["classes"].get(classe, f"Classe {classe}")
        pc1, pc2 = pca.transform(x_new)[0]
        st.success("C'est un " + nom)
        st.caption(f"Classe {classe} · PC1 = {pc1:.2f} · PC2 = {pc2:.2f}")
