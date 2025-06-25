# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv # Nécessaire si vous utilisez un .env en local, mais inoffensif ici
from supabase import create_client, Client # Importation pour Supabase

# --- Définir les noms de fichiers du modèle et du préprocesseur ---
# Ces noms doivent correspondre EXACTEMENT aux fichiers que vous avez uploadés/sauvegardés dans Colab (/content/)
MODEL_FILENAME = 'random_forest_model.pkl' # <-- VÉRIFIE ET ADAPTE CE NOM
PREPROCESSOR_FILENAME = 'column_transformer_preprocessor.pkl' # <-- VÉRIFIE ET ADAPTE CE NOM

# --- Charger les variables d'environnement ---
# En Colab, os.environ est déjà rempli par la Cellule 2, donc load_dotenv() ne fait rien ici.
# Mais c'est bon de le laisser pour un déploiement local ou avec un .env.
load_dotenv()

# --- Récupérer les informations de connexion ---
supabase_url = os.getenv("swivaiakgzzzwdumnbam")
supabase_key = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN3aXZhaWFrZ3p6endkdW1uYmFtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg2MDM4OTksImV4cCI6MjA2NDE3OTg5OX0.PzLxdl1spnEosej2XNiJQnQ2rxOv4VT5TF5xcU6ms7Y")

# --- Initialiser la connexion à Supabase au démarrage de l'API ---
supabase: Client = None # Initialiser la variable

if not supabase_url or not supabase_key:
    print("❌ AVERTISSEMENT : SUPABASE_URL ou SUPABASE_KEY non définis. La connexion Supabase ne sera pas établie.")
    # L'API démarrera mais les fonctionnalités Supabase ne fonctionneront pas.
else:
    try:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Client Supabase initialisé.")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation du client Supabase : {e}")
        supabase = None # S'assurer que supabase est None en cas d'erreur

# --- Charger le modèle et le préprocesseur au démarrage ---
loaded_model = None
loaded_preprocessor = None

try:
    with open(MODEL_FILENAME, 'rb') as file:
        loaded_model = pickle.load(file)
    with open(PREPROCESSOR_FILENAME, 'rb') as file:
        loaded_preprocessor = pickle.load(file)
    print("✅ Modèle et préprocesseur chargés au démarrage de l'API.")
except FileNotFoundError as e:
    print(f"❌ Erreur : Fichier modèle ou préprocesseur introuvable ({e}). Assurez-vous que les fichiers PKL sont uploadés et que les noms dans le code sont corrects.")
    # L'API démarrera mais ne pourra pas faire de prédictions
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle ou du préprocesseur : {e}")
    loaded_model = None
    loaded_preprocessor = None


# --- Créer l'application FastAPI ---
app = FastAPI(
    title="API de Prédiction du Statut Post-Entretien Frigo",
    description="API pour prédire le statut d'un frigo après un entretien basé sur ses caractéristiques.",
    version="1.0.0",
)



# --- Configuration CORS ---
origins = [
    "http://localhost",
    "http://localhost:8080", # Si votre front-end Lovable tourne localement sur le port 8080
    "https://gulfmaintain-insight-hub.lovable.app", # <--- C'est la ligne CRUCIALE à ajouter/modifier
    # Si vous êtes encore en phase de test intensif et que vous avez des doutes sur l'URL exacte,
    # vous pouvez temporairement laisser le "*" comme dernière option, mais retirez-le en production.
    # "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- Définir le schéma des données d'entrée attendues (Pydantic) ---
# Cette classe DOIT être définie AVANT qu'elle ne soit utilisée dans l'endpoint.
class FrigoData(BaseModel):
    # Ces noms DOIVENT correspondre aux clés attendues dans le JSON de la requête POST
    Taux_remplissage_pct: float
    Temperature_C: float
    Lineaire_val: int
    Tension_V: float
    Intensite_avant_entretien_A: float
    Technicien_GFI: str
    Division: str
    Secteur: str
    Partenaire: str
    Ville: str
    Quartier: str
    Type_Frigo: str
    AF_NF: str
    Branding: str
    Securite: str # Securité ( Disjoncteur / Régulateur )
    Eclairage: str # Eclairage (O / N )
    Purge_circuit_eaux: str
    Soufflage_parties_actives: str
    Date: str # Date comme string attendue en entrée

# --- Définir un endpoint (une route) pour la prédiction ---
@app.post("/predict/", summary="Prédire le statut post-entretien d'un frigo")
async def predict_status(data: FrigoData):
    """
    Accepte les données d'un frigo après un entretien, fait la prédiction
    et potentiellement sauvegarde le résultat dans Supabase.

    Le corps de la requête doit être un objet JSON correspondant au schéma FrigoData.
    """
    # Vérifier si le modèle et le préprocesseur ont été chargés avec succès
    if loaded_model is None or loaded_preprocessor is None:
         raise HTTPException(status_code=500, detail="Erreur interne : Modèle ou préprocesseur non chargé.")

    # --- Mapper les données entrantes (du schéma Pydantic) aux noms de colonnes du DataFrame original ---
    # Le préprocesseur s'attend aux noms de colonnes exacts qui étaient dans X.
    try:
        data_dict = data.model_dump() # Utilise model_dump() (pour Pydantic v2+) ou dict() (pour v1)
        # Crée un dictionnaire avec les noms de colonnes originaux attendus par le préprocesseur
        # CES NOMS DE CLÉS DOIVENT CORRESPONDRE EXACTEMENT AUX NOMS DES COLONNES DANS TON DATAFRAME X ORIGINEL
        mapped_data = {
            'Taux_remplissage (en %)': data_dict['Taux_remplissage_pct'],
            'Température (en °C)': data_dict['Temperature_C'],
            'Linéaire (1 / 0)': data_dict['Lineaire_val'],
            'Tension (V)': data_dict['Tension_V'],
            'Intensité avant entretien (en A)': data_dict['Intensite_avant_entretien_A'],
            'Technicien_GFI': data_dict['Technicien_GFI'],
            'Division': data_dict['Division'],
            'Secteur': data_dict['Secteur'],
            'Partenaire': data_dict['Partenaire'],
            'Ville': data_dict['Ville'],
            'Quartier': data_dict['Quartier'],
            'Type_Frigo': data_dict['Type_Frigo'],
            'AF / NF': data_dict['AF_NF'],
            'Branding': data_dict['Branding'],
            'Securité ( Disjoncteur / Régulateur )': data_dict['Securite'],
            'Eclairage (O / N )': data_dict['Eclairage'],
            "Purge du circuit d'évaluation des eaux": data_dict['Purge_circuit_eaux'],
            "Soufflage des parties actives à l'air": data_dict['Soufflage_parties_actives'],
            'Date': data_dict['Date']
        }
        new_data_df = pd.DataFrame([mapped_data])

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Erreur de mappage des données d'entrée : Clé '{e}' manquante ou incorrecte. Assurez-vous que le corps JSON envoyé correspond au schéma FrigoData et que le mappage dans le code est correct.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la préparation des données pour le prétraitement : {e}")

    # --- Appliquer le préprocesseur ---
    try:
        processed_data = loaded_preprocessor.transform(new_data_df)
    except ValueError as e:
         raise HTTPException(status_code=400, detail=f"Erreur lors du prétraitement des données. Vérifiez les colonnes, types ou valeurs des données d'entrée : {e}")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Erreur interne lors du prétraitement : {e}")


    # --- Faire la prédiction ---
    try:
        prediction = loaded_model.predict(processed_data)
        probabilities = loaded_model.predict_proba(processed_data)
        classes_order = loaded_model.classes_
        prob_dict = {str(k): float(v) for k, v in zip(classes_order, probabilities[0])}


    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction par le modèle : {e}")

    # --- Optionnel: Sauvegarder le résultat dans Supabase ---
    if supabase: # Si le client Supabase est initialisé
        try:
            # Prépare les données à insérer dans ta table Supabase
            # Remplace 'votre_table_des_predictions' par le nom réel de ta table
            # et assure-toi que les clés correspondent aux noms de colonnes de cette table
            prediction_result_to_save = {
                "predicted_status": str(prediction[0]),
                # Sauvegarder les probabilités dans une colonne JSONB si possible
                "probabilities": prob_dict,
                # Ajoutez ici d'autres données que vous avez reçues et que vous voulez lier à la prédiction
                # par exemple, un identifiant unique pour cette entrée d'entretien
                # "entretien_id": data.entretien_id # Si vous avez ajouté un champ entretien_id au schéma
                # Vous pourriez aussi vouloir sauvegarder les valeurs clés qui ont mené à la prédiction
                "intensite_avant": data_dict['Intensite_avant_entretien_A'],
                "temperature": data_dict['Temperature_C'],
                "type_frigo": data_dict['Type_Frigo']
                 # ... ajoutez d'autres champs pertinents
            }

            # Exécute l'insertion
            # Utilise la syntaxe correcte pour l'insertion avec le client Supabase
            # La méthode `insert` prend un dictionnaire ou une liste de dictionnaires
            response = supabase.table('votre_table_des_predictions').insert(prediction_result_to_save).execute()

            # Vérifie si l'insertion a réussi. response.data contient les données insérées si succès.
            if response.data:
                print("✅ Résultat de la prédiction sauvegardé dans Supabase.")
            else:
                 # response.error contient les détails de l'erreur Supabase
                 print(f"❌ Erreur lors de la sauvegarde dans Supabase : {response.error}")
                 # Tu pourrais choisir de lever une HTTPException ici si la sauvegarde est obligatoire
                 # raise HTTPException(status_code=500, detail=f"Impossible de sauvegarder la prédiction: {response.error}")

        except Exception as e:
             print(f"❌ Erreur inattendue lors de la tentative de sauvegarde dans Supabase : {e}")
             # Journaliser cette erreur

    # --- Renvoyer le résultat (code existant) ---
    return {
        "predicted_status": str(prediction[0]),
        "probabilities": prob_dict
    }

# --- Ajouter un endpoint racine pour vérifier que l'API fonctionne ---
@app.get("/", summary="Vérifier l'état de l'API")
async def read_root():
    return {"message": "API de Prédiction du Statut Post-Entretien Frigo est opérationnelle!"}

# Note: Pour exécuter cette API dans Colab, vous utiliserez nest_asyncio et uvicorn.Server
# comme montré dans l'étape de test ci-dessous.
