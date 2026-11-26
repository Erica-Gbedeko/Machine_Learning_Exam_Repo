import glob
import os
import html
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Segmentation avec le modèle CAH",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------------------------
# STYLE DE L'INTERFACE : DESIGN CLUSTER BLEU / VIOLET
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #edf2ff 52%, #f8faff 100%);
    color: #172554;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #111827 0%, #312e81 52%, #0e7490 100%);
    padding: 2rem 2.2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 30px rgba(30, 41, 59, 0.16);
}
.hero h1 {
    color: #ffffff !important;
    margin: 0;
    padding: 0;
    border: 0;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -.7px;
}
.hero p {
    color: #dbeafe !important;
    margin: .55rem 0 0 0;
    font-size: 1rem;
}
h1, h2, h3 {
    color: #172554 !important;
    font-weight: 750 !important;
}
h1 { border: 0; }
div[data-testid="stCaptionContainer"] p { color: #64748b; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e1b4b 100%);
}
section[data-testid="stSidebar"] * { color: #f8fafc; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 7px;
    transition: .2s ease;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(129,140,248,.22);
    border-color: rgba(165,180,252,.5);
}

div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"],
div[data-testid="stTextInput"],
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.92);
    border: 1px solid #e2e8f0;
    border-radius: 13px;
    padding: 10px 12px 5px 12px;
    box-shadow: 0 4px 13px rgba(30,64,175,.045);
    transition: all .2s ease;
}
div[data-testid="stNumberInput"]:hover,
div[data-testid="stSelectbox"]:hover,
div[data-testid="stTextInput"]:hover {
    border-color: #818cf8;
    box-shadow: 0 5px 16px rgba(99,102,241,.10);
}
div[data-testid="stWidgetLabel"] p {
    color: noire;
    font-weight: 650;
}

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    border: 0;
    border-radius: 12px;
    min-height: 46px;
    font-weight: 700;
    transition: all .2s ease;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(105deg, #4f46e5, #6366f1 55%, #7c3aed);
    color: white;
    box-shadow: 0 6px 16px rgba(79,70,229,.22);
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 9px 20px rgba(79,70,229,.20);
}

.result-card {
    background: linear-gradient(120deg, rgba(79,70,229,.11), rgba(14,116,144,.09));
    border: 1px solid rgba(99,102,241,.22);
    border-left: 6px solid #4f46e5;
    border-radius: 16px;
    padding: 1.15rem 1.4rem;
    margin: 1rem 0 .7rem 0;
    box-shadow: 0 7px 20px rgba(30,41,59,.06);
}
.result-card .label {
    color: #64748b;
    font-size: .9rem;
    font-weight: 650;
    margin: 0;
}
.result-card .value {
    color: #312e81;
    font-size: 1.65rem;
    font-weight: 800;
    margin: .25rem 0 0 0;
    overflow-wrap: anywhere;
}
.metric-note {
    background: rgba(255,255,255,.75);
    border: 1px solid #e2e8f0;
    padding: .75rem 1rem;
    border-radius: 12px;
    color: #475569;
    margin-top: .5rem;
}
div[data-testid="stAlert"] { border-radius: 13px; }
hr { border-color: #dbe4f0; margin: 1.3rem 0; }

@media (max-width: 768px) {
    .block-container { padding: 1rem .8rem 2rem .8rem; }
    .hero { padding: 1.35rem; }
    .hero h1 { font-size: 1.5rem !important; }
    .result-card .value { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CONFIGURATION DES 4 JEUX DE DONNÉES
# Les colonnes et leur ordre restent ceux de ton premier app.py.
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
        "defauts": {},  # À compléter avec les médianes calculées dans le notebook.
        "classes": {},  # À compléter avec les noms de classes du notebook.
        "encodees": [],
        "exemples": [],
    },
    "Produits (PriceRunner)": {
        "dossier": "pricerunner",
        "titre": "Segmentation des produits - modèle KMEANS",
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
# UTILITAIRES ET CHARGEMENT DES FICHIERS JOBLIB
# ---------------------------------------------------------------------------
def trouver_dossier(nom):
    """Trouve le dossier même si les accents sont encodés différemment."""
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


def preparer_donnees(cfg, encoders, df):
    """
    Prépare les données CSV avec les mêmes encodeurs et le même ordre
    de colonnes que pour la prédiction simple.
    """
    colonnes = cfg["colonnes"]
    absentes = [col for col in colonnes if col not in df.columns]
    if absentes:
        return None, None, (
            "Colonnes manquantes dans le fichier CSV : " + ", ".join(absentes)
        )

    donnees = df[colonnes].copy()

    # Convertir les colonnes non encodées en nombres.
    for col in colonnes:
        if not (col in cfg["encodees"] and col in encoders):
            donnees[col] = pd.to_numeric(donnees[col], errors="coerce")

    # Ignorer les lignes incomplètes ou contenant une valeur numérique invalide.
    masque_valides = donnees.notna().all(axis=1)
    donnees_valides = donnees.loc[masque_valides].copy()

    if donnees_valides.empty:
        return None, None, "Aucune ligne complète et exploitable dans le fichier."

    # Encoder les variables textuelles avec les encodeurs du modèle.
    for col in colonnes:
        if col in cfg["encodees"] and col in encoders:
            valeurs = donnees_valides[col].astype(str).str.strip()
            connues = set(str(v) for v in encoders[col].classes_)
            inconnues = sorted(set(valeurs) - connues)
            if inconnues:
                return None, None, (
                    f"Valeurs inconnues dans la colonne « {col} » : "
                    + ", ".join(inconnues[:10])
                )
            donnees_valides[col] = encoders[col].transform(valeurs)

    X = donnees_valides[colonnes].to_numpy(dtype=float)
    return X, donnees_valides.index, None


# ---------------------------------------------------------------------------
# FONCTIONS DES DEUX MODES DE PRÉDICTION
# ---------------------------------------------------------------------------
def afficher_resultat_simple(nom, classe, pc1, pc2):
    st.markdown(
        f'<div class="result-card">'
        f'<p class="label">CLASSE ATTRIBUÉE</p>'
        f'<p class="value">{html.escape("C’est un " + str(nom))}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="metric-note">Classe : <b>{classe}</b>'
        f' &nbsp;·&nbsp; PC1 : <b>{pc1:.2f}</b>'
        f' &nbsp;·&nbsp; PC2 : <b>{pc2:.2f}</b></div>',
        unsafe_allow_html=True
    )


def onglet_prediction_simple(cfg, scaler, pca, model, encoders, cle):
    """Prédiction à partir des valeurs saisies dans le formulaire."""
    st.markdown("### 📝 Caractéristiques à renseigner")
    st.caption(
        "Remplissez les champs ci-dessous ou choisissez un exemple, "
        "puis lancez l'affectation à une classe."
    )

    colonnes = cfg["colonnes"]

    # Valeurs initiales des champs : médianes définies dans DATASETS.
    for col in colonnes:
        k = f"{cle}_{col}"
        if k not in st.session_state:
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

    if cfg["exemples"]:
        st.selectbox(
            "🧪 Charger un exemple du jeu de données",
            ["—"] + [f"Exemple {i + 1}" for i in range(len(cfg["exemples"]))],
            key=f"{cle}_exemple",
            on_change=charger_exemple,
        )

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

    st.markdown("---")
    if st.button(
        "🔮 Affecter à une classe",
        type="primary",
        use_container_width=True,
        key=f"{cle}_prediction_simple"
    ):
        valeurs = []
        ok = True

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
            x_new = scaler.transform(
                np.array(valeurs, dtype=float).reshape(1, -1)
            )
            classe = int(model.predict(x_new)[0])
            nom = cfg["classes"].get(classe, f"Classe {classe}")
            pc1, pc2 = pca.transform(x_new)[0]
            afficher_resultat_simple(nom, classe, pc1, pc2)


def onglet_prediction_multiple(cfg, scaler, pca, model, encoders, cle):
    """Import CSV, prédiction de toutes les lignes exploitables et export."""
    st.markdown("### 📂 Importer un jeu de données")
    st.write(
        "Chargez un fichier CSV contenant plusieurs lignes. "
        "L'application appliquera le modèle sélectionné à chaque ligne exploitable."
    )

    colonnes = cfg["colonnes"]

    st.markdown("**Colonnes attendues dans le fichier :**")
    st.code(", ".join(colonnes), language="text")

    # Télécharger un modèle CSV vide avec les bons en-têtes.
    modele_csv = pd.DataFrame(columns=colonnes).to_csv(index=False).encode("utf-8")
    st.download_button(
        "📄 Télécharger un modèle de fichier CSV",
        data=modele_csv,
        file_name=f"modele_{cle}.csv",
        mime="text/csv",
        key=f"{cle}_modele_csv"
    )

    fichier = st.file_uploader(
        "Choisir un fichier CSV",
        type=["csv"],
        key=f"{cle}_fichier_csv"
    )

    if fichier is None:
        st.info("Importez un fichier CSV pour commencer les prédictions multiples.")
        return

    try:
        df = pd.read_csv(fichier, sep=None, engine="python")
    except Exception as e:
        st.error(f"Impossible de lire ce fichier CSV : {e}")
        return

    st.markdown("#### Aperçu du fichier importé")
    st.caption(f"{len(df)} ligne(s) chargée(s).")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button(
        "🚀 Lancer les prédictions sur le fichier",
        type="primary",
        use_container_width=True,
        key=f"{cle}_lancer_multiple"
    ):
        X, index_valides, erreur = preparer_donnees(cfg, encoders, df)

        if erreur:
            st.error(erreur)
        else:
            try:
                X_scaled = scaler.transform(X)
                classes = model.predict(X_scaled)
                coordonnees = pca.transform(X_scaled)

                resultat = df.loc[index_valides].copy()
                resultat["Classe prédite"] = [
                    cfg["classes"].get(int(c), f"Classe {int(c)}")
                    for c in classes
                ]
                resultat["Numéro de classe"] = classes.astype(int)
                resultat["PC1"] = coordonnees[:, 0]
                resultat["PC2"] = coordonnees[:, 1]

                nb_ignorees = len(df) - len(resultat)
                st.session_state[f"{cle}_resultat_multiple"] = resultat
                st.session_state[f"{cle}_nb_ignorees"] = nb_ignorees
                st.session_state[f"{cle}_nom_fichier"] = fichier.name

            except Exception as e:
                st.error(f"Une erreur est survenue pendant la prédiction : {e}")

    resultat = st.session_state.get(f"{cle}_resultat_multiple")
    nom_fichier = st.session_state.get(f"{cle}_nom_fichier")

    if resultat is not None and nom_fichier == fichier.name:
        st.markdown("---")
        st.markdown("### ✅ Résultats de la segmentation")

        nb_ignorees = st.session_state.get(f"{cle}_nb_ignorees", 0)
        col1, col2 = st.columns(2)
        col1.metric("Lignes segmentées", len(resultat))
        col2.metric("Lignes ignorées", nb_ignorees)

        st.dataframe(resultat, use_container_width=True)

        csv_resultat = resultat.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Télécharger les résultats (CSV)",
            data=csv_resultat,
            file_name=f"resultats_segmentation_{cle}.csv",
            mime="text/csv",
            key=f"{cle}_telecharger_resultat"
        )


# ---------------------------------------------------------------------------
# INTERFACE PRINCIPALE : NAVIGATION + BANDEAU + DEUX ONGLETS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 Erica APP")
    st.caption("Espace de segmentation")
    st.divider()

    choix = st.radio(
        "CHOISIR UN MODÈLE DE DONNÉES",
        list(DATASETS.keys()),
        key="dataset_choisi"
    )

    st.divider()
    st.caption("Quatre modèles de segmentation accessibles depuis une seule application.")
    st.caption("Modèles entraînés et sauvegardés au format .joblib.")

st.markdown("""
<div class="hero">
    <h1>Bienvenue dans mon application de segmentation</h1>
    <p>Explorez vos données et obtenez des prédictions avec les modèles de clustering.</p>
</div>
""", unsafe_allow_html=True)

cfg = DATASETS[choix]

st.subheader(cfg["titre"])
st.caption(cfg["description"])

objets, erreur = charger(cfg["dossier"])
if erreur:
    st.error(erreur)
    st.stop()

scaler = objets["scaler"]
pca = objets["pca"]
model = objets["model"]
encoders = objets["encoders"]

cle = choix.replace(" ", "_").replace("(", "").replace(")", "")

st.markdown("---")

tab_simple, tab_multiple = st.tabs([
    "✍️ Prédiction simple",
    "📁 Prédiction multiple (CSV)"
])

with tab_simple:
    onglet_prediction_simple(cfg, scaler, pca, model, encoders, cle)

with tab_multiple:
    onglet_prediction_multiple(cfg, scaler, pca, model, encoders, cle)
