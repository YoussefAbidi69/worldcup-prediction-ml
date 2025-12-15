import pickle
import json
from sklearn.dummy import DummyClassifier
from wc_model import WorldCupPredictor

# Charger la liste des équipes
with open('../teams_list.json', 'r', encoding='utf-8') as f:
    teams = json.load(f)

# Construire name_map (clé = lowercase input -> canonical name)
name_map = {t.strip().lower(): t for t in teams}

# Dummy classifier: toujours prédira 'draw' (class 2) but provide probabilities
clf = DummyClassifier(strategy='uniform')
# For fitting a DummyClassifier we need some data
X = [[0]] * 3
y = [0,1,2]
clf.fit(X, y)

# last_elo and last_form: default values
last_elo = {t: 1500 for t in teams}
last_form = {t: 0.5 for t in teams}

predictor = WorldCupPredictor(pipe_rf=clf, last_elo=last_elo, last_form=last_form, name_map=name_map)

with open('../wc_web_predictor.pkl', 'wb') as f:
    pickle.dump(predictor, f)

print('wc_web_predictor.pkl créé')
