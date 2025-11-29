# 🏆 WC Analytics — FIFA World Cup Match Prediction  
Machine Learning · Flask App · Interactive Dashboard

WC Analytics est une application web permettant de prédire l’issue d’un match de football de Coupe du Monde grâce à un modèle Machine Learning basé sur :

- ELO Ratings  
- Forme récente des équipes  
- Caractéristiques statistiques  
- Encodage One-Hot + Pipeline scikit-learn  

L’interface inclut :

- Sélection dynamique des équipes  
- Probabilités détaillées (victoire / nul / défaite)  
- Graphiques interactifs (Chart.js)  
- Barres animées pour les probabilités  


---

## 🚀 Technologies utilisées

### Backend
- Python 3.10  
- Flask  
- Pandas  
- NumPy  
- Scikit-learn 1.6.1 (important pour la compatibilité pickle)

### Frontend
- HTML5  
- CSS3  
- JavaScript  
- Jinja2  
- Chart.js  

---


---

## 🛠️ Installation et exécution

### 1️⃣ Installer Python 3.10 (obligatoire)
Ubuntu 24/25 ne fournit plus Python 3.10 par défaut.  
Installe-le via le dépôt Deadsnakes :

bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-distutils -y


2️⃣ Exécuter le script d’installation

Depuis le dossier du projet :

chmod +x setup.sh
./setup.sh

Ce script :

crée un environnement virtuel Python 3.10

installe Flask, pandas, numpy, sklearn 1.6.1

prépare l’environnement pour charger le modèle ML

3️⃣ Lancer l’application

./run.sh


Puis ouvrir :

http://127.0.0.1:5000