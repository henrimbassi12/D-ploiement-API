# Utilisez une image Python 3.11 légère comme base.
FROM python:3.11-slim

# Définissez le répertoire de travail dans le conteneur Docker
WORKDIR /app

# Copiez le fichier requirements.txt dans le répertoire de travail
COPY requirements.txt .

# Installez les dépendances Python listées dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiez tous les autres fichiers du répertoire actuel
COPY . .

# Commande pour lancer l'application avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
