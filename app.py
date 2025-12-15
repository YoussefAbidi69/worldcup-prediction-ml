from flask import Flask, request, render_template
import pickle
import pandas as pd
import json
import numpy as np   # pour la moyenne des probas
import signal
import sys
import joblib

import pickle


def normalize_team(team_name):
    if not team_name:
        return None
    return old_predictor.name_map.get(team_name.strip().lower(), team_name.strip())


TEAM_TO_FLAG = {
    "Algeria": "dz",
    "Angola": "ao",
    "Argentina": "ar",
    "Australia": "au",
    "Austria": "at",
    "Belgium": "be",
    "Bolivia": "bo",
    "Bosnia and Herzegovina": "ba",
    "Brazil": "br",
    "Bulgaria": "bg",
    "Cameroon": "cm",
    "Canada": "ca",
    "Chile": "cl",
    "China": "cn",
    "Chinese Taipei": "tw",
    "Colombia": "co",
    "Costa Rica": "cr",
    "Croatia": "hr",
    "Cuba": "cu",
    "Czech Republic": "cz",
    "Czechoslovakia": "cz",
    "Denmark": "dk",
    "Dutch East Indies": "id",
    "East Germany": "de",
    "Ecuador": "ec",
    "Egypt": "eg",
    "El Salvador": "sv",
    "England": "gb-eng",
    "Equatorial Guinea": "gq",
    "France": "fr",
    "Germany": "de",
    "Ghana": "gh",
    "Greece": "gr",
    "Haiti": "ht",
    "Honduras": "hn",
    "Hungary": "hu",
    "Iceland": "is",
    "Iran": "ir",
    "Iraq": "iq",
    "Israel": "il",
    "Italy": "it",
    "Ivory Coast": "ci",
    "Jamaica": "jm",
    "Japan": "jp",
    "Kuwait": "kw",
    "Mexico": "mx",
    "Morocco": "ma",
    "Netherlands": "nl",
    "New Zealand": "nz",
    "Nigeria": "ng",
    "North Korea": "kp",
    "Northern Ireland": "gb-nir",
    "Norway": "no",
    "Panama": "pa",
    "Paraguay": "py",
    "Peru": "pe",
    "Poland": "pl",
    "Portugal": "pt",
    "Qatar": "qa",
    "Republic of Ireland": "ie",
    "Romania": "ro",
    "Russia": "ru",
    "Saudi Arabia": "sa",
    "Scotland": "gb-sct",
    "Senegal": "sn",
    "Serbia": "rs",
    "Serbia and Montenegro": "rs",
    "Slovakia": "sk",
    "Slovenia": "si",
    "South Africa": "za",
    "South Korea": "kr",
    "Soviet Union": "su",
    "Spain": "es",
    "Sweden": "se",
    "Switzerland": "ch",
    "Thailand": "th",
    "Togo": "tg",
    "Trinidad and Tobago": "tt",
    "Tunisia": "tn",
    "Turkey": "tr",
    "Ukraine": "ua",
    "United Arab Emirates": "ae",
    "United States": "us",
    "Uruguay": "uy",
    "Wales": "gb-wls",
    "West Germany": "de",
    "Yugoslavia": "yu",
    "Zaire": "cd",
}

# Charger MLValod de façon non bloquante :
# - On démarre un thread qui importe MLValod en arrière-plan.
# - Avant que le chargement soit terminé, la route /score retourne un message de chargement
import threading
import importlib

# état du loader
_predire_score_final = None
_predire_load_error = None
_predire_loading = False

def _fallback_predire(home_team, away_team, *args, **kwargs):
    return {
        'score': '1 - 1',
        'resultat': 'Draw',
        'note': 'Modèle non chargé (chargement en cours ou erreur).'
    }

def _load_mlvalod():
    global _predire_score_final, _predire_load_error, _predire_loading
    try:
        _predire_loading = True
        mod = importlib.import_module('MLValod')
        # récupérer la fonction prévue
        _predire_score_final = getattr(mod, 'predire_score_final')
    except Exception as e:
        _predire_load_error = str(e)
        _predire_score_final = None
    finally:
        _predire_loading = False

# Remarque: on n'appelle PAS automatiquement _load_mlvalod() au démarrage
# pour éviter de lancer l'entraînement lourd au moment de l'import.
# Vous pouvez démarrer le chargement manuellement (script séparé) ou
# appeler une route d'administration pour lancer _load_mlvalod().


# ================================
# Classe WorldCupPredictor
# ================================
class WorldCupPredictor:
    def __init__(self, model_xgb, feature_columns, last_elo, last_form, name_map, matches_df):
        self.model = model_xgb
        self.feature_columns = feature_columns
        self.last_elo = last_elo
        self.last_form = last_form
        self.name_map = name_map
        self.matches_df = matches_df

    def _prepare_X(self, df):
        df = pd.get_dummies(df, drop_first=True)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_columns]

    def predict(self, home_input, away_input):
        home = self.name_map.get(home_input.strip().lower())
        away = self.name_map.get(away_input.strip().lower())


        if home is None or away is None:
            return None, "❌ Une des équipes n'existe pas"

        h_elo = self.last_elo.get(home, 1500)
        a_elo = self.last_elo.get(away, 1500)
        h_form = self.last_form.get(home, 0.5)
        a_form = self.last_form.get(away, 0.5)

        ex1 = pd.DataFrame([{
            "home_team_name": home,
            "away_team_name": away,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "home_recent_form": h_form,
            "away_recent_form": a_form,
            "elo_diff": h_elo - a_elo,
            "form_diff": h_form - a_form,
            "goal_diff_avg": 0,
            "is_world_cup": 1,
            "home_advantage": 1
        }])

        X1 = self._prepare_X(ex1)
        proba1 = self.model.predict_proba(X1)[0]

        ex2 = pd.DataFrame([{
            "home_team_name": away,
            "away_team_name": home,
            "home_elo": a_elo,
            "away_elo": h_elo,
            "home_recent_form": a_form,
            "away_recent_form": h_form,
            "elo_diff": a_elo - h_elo,
            "form_diff": a_form - h_form,
            "goal_diff_avg": 0,
            "is_world_cup": 1,
            "home_advantage": 1
        }])

        X2 = self._prepare_X(ex2)
        proba2 = self.model.predict_proba(X2)[0]

        p_draw = (proba1[0] + proba2[0]) / 2
        p_home = (proba1[1] + proba2[2]) / 2
        p_away = (proba1[2] + proba2[1]) / 2

        proba_sym = np.array([p_draw, p_home, p_away])
        pred_class = int(np.argmax(proba_sym))

        text = (
            f"Match nul entre {home} et {away}."
            if pred_class == 0 else
            f"Victoire de {home}."
            if pred_class == 1 else
            f"Victoire de {away}."
        )

        return text, {
            "proba": proba_sym.tolist(),
            "h_elo": h_elo,
            "a_elo": a_elo,
            "h_form": h_form,
            "a_form": a_form,
        }


# ================================
# Chargement données & Flask app
# ================================
app = Flask(__name__)

# Liste des équipes pour les listes déroulantes
with open("teams_list.json", "r", encoding="utf-8") as f:
    TEAMS = json.load(f)

# Historique des matchs
MATCHES_DF = pd.read_csv("matches_history.csv")
MATCHES_DF["date"] = pd.to_datetime(MATCHES_DF["date"], errors="coerce")

# ================================
# 🔥 CHARGEMENT DU MODÈLE XGBOOST
# ================================
model_xgb = joblib.load("xgboost_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Charger ELO / forme / name_map depuis ancien predictor
with open("wc_web_predictor.pkl", "rb") as f:
    old_predictor = pickle.load(f)

predictor = WorldCupPredictor(
    model_xgb=model_xgb,
    feature_columns=feature_columns,
    last_elo=old_predictor.last_elo,
    last_form=old_predictor.last_form,
    name_map=old_predictor.name_map,
    matches_df=MATCHES_DF
)

print("🔥 MODELE ACTIF :", type(predictor.model))


# Objet prédicteur complet (pipe_rf + ELO + forme + name_map)


# Charger le prédicteur de score séparé (pour les prédictions de score uniquement)
score_predictor_data = None
try:
    with open("score_predictor.pkl", "rb") as f:
        score_predictor_data = pickle.load(f)
except Exception:
    # Fallback: utiliser les données du predictor principal
    score_predictor_data = {
        'last_elo': predictor.last_elo,
        'last_form': predictor.last_form,
        'name_map': predictor.name_map,
        'team_stats': {},
        'global_avg_home_score': 1.5,
        'global_avg_away_score': 1.0,
    }



with open("worldcup_model_TN.pkl", "rb") as f:
    model_TN = pickle.load(f)

model_clf_TN  = model_TN["model_clf"]
model_home_TN = model_TN["model_home"]
model_away_TN = model_TN["model_away"]
X_columns_TN  = model_TN["X_columns"]

le_home_TN  = model_TN.get("le_home")
le_away_TN  = model_TN.get("le_away")



# Historique des matchs (matches_history.csv généré depuis matches.csv)



# ================================
# Statistiques globales du dataset
# ================================
TOTAL_MATCHES = int(MATCHES_DF.shape[0])

teams_home = set(MATCHES_DF["home_team_name"].dropna().unique())
teams_away = set(MATCHES_DF["away_team_name"].dropna().unique())
TOTAL_TEAMS = len(teams_home | teams_away)

year_series = MATCHES_DF["date"].dt.year.dropna()
if not year_series.empty:
    DATA_START_YEAR = int(year_series.min())
    DATA_END_YEAR = int(year_series.max())
else:
    DATA_START_YEAR = None
    DATA_END_YEAR = None



# ================================
# Fonctions utilitaires pour l'historique
# ================================
def compute_h2h(team_a, team_b, n=5):
    """Renvoie les n derniers matchs entre team_a et team_b."""
    df = MATCHES_DF.copy()
    mask = (
        ((df["home_team_name"] == team_a) & (df["away_team_name"] == team_b)) |
        ((df["home_team_name"] == team_b) & (df["away_team_name"] == team_a))
    )
    sub = df.loc[mask].sort_values("date", ascending=False).head(n)

    results = []
    for _, row in sub.iterrows():
        home = row["home_team_name"]
        away = row["away_team_name"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        if hs > as_:
            winner = home
        elif hs < as_:
            winner = away
        else:
            winner = "Nul"

        results.append({
            "date": row["date"].strftime("%Y-%m-%d") if pd.notnull(row["date"]) else "",
            "home": home,
            "away": away,
            "score": f"{hs} - {as_}",
            "winner": winner
        })
    return results


def compute_h2h_summary(team_a, team_b):
    """Statistiques globales H2H entre team_a et team_b."""
    df = MATCHES_DF.copy()
    mask = (
        ((df["home_team_name"] == team_a) & (df["away_team_name"] == team_b)) |
        ((df["home_team_name"] == team_b) & (df["away_team_name"] == team_a))
    )
    sub = df.loc[mask]

    if sub.empty:
        return None

    total = len(sub)
    wins_a = wins_b = draws = 0
    goals_a = goals_b = 0

    for _, row in sub.iterrows():
        home = row["home_team_name"]
        away = row["away_team_name"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        if home == team_a:
            goals_a += hs
            goals_b += as_
            if hs > as_:
                wins_a += 1
            elif hs < as_:
                wins_b += 1
            else:
                draws += 1
        else:  # home == team_b
            goals_b += hs
            goals_a += as_
            if hs > as_:
                wins_b += 1
            elif hs < as_:
                wins_a += 1
            else:
                draws += 1

    return {
        "team_a": team_a,
        "team_b": team_b,
        "total": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "goals_a": goals_a,
        "goals_b": goals_b,
    }


def compute_last_matches(team, n=5):
    """Renvoie les n derniers matchs pour une équipe donnée."""
    df = MATCHES_DF.copy()
    mask = (df["home_team_name"] == team) | (df["away_team_name"] == team)
    sub = df.loc[mask].sort_values("date", ascending=False).head(n)

    results = []
    for _, row in sub.iterrows():
        home = row["home_team_name"]
        away = row["away_team_name"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        if home == team:
            opponent = away
            goals_for = hs
            goals_against = as_
        else:
            opponent = home
            goals_for = as_
            goals_against = hs

        if goals_for > goals_against:
            result = "Victoire"
        elif goals_for < goals_against:
            result = "Défaite"
        else:
            result = "Nul"

        results.append({
            "date": row["date"].strftime("%Y-%m-%d") if pd.notnull(row["date"]) else "",
            "opponent": opponent,
            "score": f"{goals_for} - {goals_against}",
            "result": result
        })
    return results


# ================================
# Pourcentages victoires / nuls
# ================================
home_wins = (MATCHES_DF["home_score"] > MATCHES_DF["away_score"]).sum()
away_wins = (MATCHES_DF["away_score"] > MATCHES_DF["home_score"]).sum()
draws = (MATCHES_DF["home_score"] == MATCHES_DF["away_score"]).sum()

if TOTAL_MATCHES > 0:
    P_HOME = round(home_wins / TOTAL_MATCHES * 100, 1)
    P_DRAW = round(draws / TOTAL_MATCHES * 100, 1)
    P_AWAY = round(away_wins / TOTAL_MATCHES * 100, 1)
else:
    P_HOME = P_DRAW = P_AWAY = 0


# ================================
# ROUTES FLASK
# ================================

# --- Accueil simple ---
@app.route("/")
def home():
    stats = {
    "total_matches": TOTAL_MATCHES,
    "total_teams": TOTAL_TEAMS,
    "start_year": DATA_START_YEAR,
    "end_year": DATA_END_YEAR,
    "p_home": P_HOME,
    "p_draw": P_DRAW,
    "p_away": P_AWAY,
    }
    return render_template(
        "home.html",
        current_page="home",
        stats=stats
    )



def predict_TN_model(home_team, away_team):

    match_data = {col: 0 for col in X_columns_TN}

    match_data["home_team_encoded"] = (
        le_home_TN.transform([home_team])[0]
        if home_team in le_home_TN.classes_ else 0
    )
    match_data["away_team_encoded"] = (
        le_away_TN.transform([away_team])[0]
        if away_team in le_away_TN.classes_ else 0
    )

    match_data["home_xg"] = 1.3
    match_data["away_xg"] = 1.1
    match_data["Attendance"] = 45000

    X_match = pd.DataFrame([match_data])[X_columns_TN]

    pred_class = model_clf_TN.predict(X_match)[0]
    home_goals = int(round(model_home_TN.predict(X_match)[0]))
    away_goals = int(round(model_away_TN.predict(X_match)[0]))

    return (
        ["Victoire domicile", "Match nul", "Victoire extérieur"][pred_class],
        f"{home_goals} - {away_goals}"
    )


# --- Page Prédictions ---
@app.route("/predictions", methods=["GET", "POST"])
def predictions():
    home_team = ""
    away_team = ""
    prediction_text = None
    error = None

    proba_home = None
    proba_draw = None
    proba_away = None

    winner_side = None           # "home" | "away" | "draw"
    confidence = None            # probabilité max en %
    confidence_label = None      # "Faible", "Moyenne", "Forte"

    home_elo = None
    away_elo = None
    home_form = None
    away_form = None

    tn_result = None
    tn_score = None
    home_flag = None
    away_flag = None


    h2h_matches = []
    h2h_summary = None
    last_home_matches = []
    last_away_matches = []
    analysis_text = None

    if request.method == "POST":
        home_team = request.form.get("home_team", "").strip()
        away_team = request.form.get("away_team", "").strip()

        if home_team and away_team:
            home_norm = predictor.name_map.get(home_team.lower(), home_team)
            away_norm = predictor.name_map.get(away_team.lower(), away_team)
            text, details = predictor.predict(home_team, away_team)

            if text is None:
                error = details
            else:
                tn_result, tn_score = predict_TN_model(home_norm, away_norm)





            home_flag = TEAM_TO_FLAG.get(home_norm)
            away_flag = TEAM_TO_FLAG.get(away_norm)


            if text is None:
                error = details  # message d'erreur (équipe inconnue)
            else:
                prediction_text = text

                # Probabilités symétriques [nul, domicile, extérieur]
                proba = details.get("proba", [0.0, 0.0, 0.0])
                proba_draw = round(proba[0] * 100, 1)
                proba_home = round(proba[1] * 100, 1)
                proba_away = round(proba[2] * 100, 1)

                # Winner side + confiance
                if proba_home >= proba_away and proba_home >= proba_draw:
                    winner_side = "home"
                    max_p = proba_home
                elif proba_away >= proba_home and proba_away >= proba_draw:
                    winner_side = "away"
                    max_p = proba_away
                else:
                    winner_side = "draw"
                    max_p = proba_draw

                confidence = round(max_p, 1)
                if confidence < 40:
                    confidence_label = "Faible confiance"
                elif confidence < 60:
                    confidence_label = "Confiance moyenne"
                else:
                    confidence_label = "Forte confiance"

                # Détails ELO & forme
                home_elo = round(details.get("h_elo", 0), 1)
                away_elo = round(details.get("a_elo", 0), 1)
                home_form = round(details.get("h_form", 0) * 100, 1)
                away_form = round(details.get("a_form", 0) * 100, 1)

                # Historique H2H + derniers matchs
                home_norm = normalize_team(home_team)
                away_norm = normalize_team(away_team)

                h2h_matches = compute_h2h(home_norm, away_norm, n=5)
                h2h_summary = compute_h2h_summary(home_norm, away_norm)
                last_home_matches = compute_last_matches(home_norm, n=5)
                last_away_matches = compute_last_matches(away_norm, n=5)

                # Analyse textuelle du modèle
                parts = []

                # ELO
                if home_elo is not None and away_elo is not None:
                    diff_elo = home_elo - away_elo
                    if abs(diff_elo) < 30:
                        parts.append("Les niveaux ELO des deux équipes sont très proches.")
                    elif diff_elo > 0:
                        parts.append(f"{home_team} possède un avantage ELO d'environ {abs(diff_elo):.0f} points.")
                    else:
                        parts.append(f"{away_team} possède un avantage ELO d'environ {abs(diff_elo):.0f} points.")

                # Forme récente
                if home_form is not None and away_form is not None:
                    diff_form = home_form - away_form
                    if abs(diff_form) < 5:
                        parts.append("La forme récente est équilibrée.")
                    elif diff_form > 0:
                        parts.append(f"{home_team} est légèrement mieux en forme ({home_form:.1f}% contre {away_form:.1f}%).")
                    else:
                        parts.append(f"{away_team} est légèrement mieux en forme ({away_form:.1f}% contre {home_form:.1f}%).")

                # Pronostic du modèle
                if winner_side == "home":
                    parts.append(f"Le modèle penche pour une victoire de {home_team}.")
                elif winner_side == "away":
                    parts.append(f"Le modèle penche pour une victoire de {away_team}.")
                else:
                    parts.append("Le modèle considère le match nul comme scénario le plus probable.")

                # Niveau de confiance
                if confidence is not None:
                    if confidence < 40:
                        parts.append("L'incertitude reste élevée, plusieurs scénarios sont plausibles.")
                    elif confidence < 60:
                        parts.append("La confiance est modérée, le match reste ouvert.")
                    else:
                        parts.append("La confiance du modèle est élevée sur ce pronostic.")

                analysis_text = " ".join(parts) if parts else None

    return render_template(
        "predictions.html",
        current_page="predictions",
        teams=TEAMS,
        home_team=home_team,
        away_team=away_team,
        prediction_text=prediction_text,
        error=error,
        proba_home=proba_home,
        proba_draw=proba_draw,
        proba_away=proba_away,
        winner_side=winner_side,
        confidence=confidence,
        confidence_label=confidence_label,
        home_elo=home_elo,
        away_elo=away_elo,
        home_form=home_form,
        away_form=away_form,
        h2h_matches=h2h_matches,
        h2h_summary=h2h_summary,
        last_home_matches=last_home_matches,
        last_away_matches=last_away_matches,
        analysis_text=analysis_text,

        tn_result=tn_result,
        tn_score=tn_score,
        home_flag=home_flag,
        away_flag=away_flag,
    )


# --- Page Score (utilise la fonction de MLValod) ---
@app.route("/score", methods=["GET", "POST"])
def score():
    home_team = ""
    away_team = ""
    result = None
    error = None

    if request.method == "POST":
        home_team = request.form.get("home_team", "").strip()
        away_team = request.form.get("away_team", "").strip()

        if not home_team or not away_team:
            error = "Veuillez renseigner les deux équipes."
        else:
            # Cas 1 : chargement en cours
            if _predire_loading:
                error = "Le modèle est en cours de chargement. Réessayez dans quelques secondes."
            # Cas 2 : erreur d'import
            elif _predire_load_error is not None and _predire_score_final is None:
                error = f"Erreur lors du chargement du module MLValod: {_predire_load_error}"
            else:
                func = _predire_score_final or _fallback_predire
                try:
                    tmp = func(home_team, away_team)
                    # si tmp contient une clé 'note' indiquant absence de modèle,
                    # afficher la note mais ne pas considérer la prédiction comme fiable
                    if isinstance(tmp, dict) and tmp.get('note'):
                        score_str = tmp.get('score', '1 - 1')
                        score_parts = score_str.split(' - ') if ' - ' in score_str else score_str.split('-')
                        result = {
                            'score': score_str,
                            'home_score': int(score_parts[0].strip()) if len(score_parts) > 0 else 1,
                            'away_score': int(score_parts[1].strip()) if len(score_parts) > 1 else 1,
                            'note': tmp.get('note')
                        }
                    else:
                        # résultat normal attendu (score, resultat)
                        if isinstance(tmp, dict) and 'score' in tmp:
                            score_str = tmp.get('score', '1 - 1')
                            score_parts = score_str.split(' - ') if ' - ' in score_str else score_str.split('-')
                            tmp['home_score'] = int(score_parts[0].strip()) if len(score_parts) > 0 else 1
                            tmp['away_score'] = int(score_parts[1].strip()) if len(score_parts) > 1 else 1
                        result = tmp
                except Exception as e:
                    error = str(e)

    # Fonction améliorée de prédiction de score basée sur l'historique et ELO
    def predict_score_logic(home, away):
        """Prédiction de score améliorée utilisant historique, ELO et forme récente"""
        try:
            # Utiliser score_predictor_data si disponible, sinon fallback sur predictor
            if score_predictor_data:
                score_data = score_predictor_data
            else:
                # Fallback sur predictor principal
                score_data = {
                    'last_elo': predictor.last_elo,
                    'last_form': predictor.last_form,
                    'name_map': predictor.name_map,
                    'team_stats': {},
                    'global_avg_home_score': 1.5,
                    'global_avg_away_score': 1.0,
                }
            
            # Normaliser les noms d'équipes
            home_key = home.strip().lower()
            away_key = away.strip().lower()
            
            # Récupérer les noms normalisés depuis le score_predictor
            home_normalized = score_data['name_map'].get(home_key, home)
            away_normalized = score_data['name_map'].get(away_key, away)
            
            # Récupérer ELO et forme depuis le score_predictor
            h_elo = score_data['last_elo'].get(home_normalized, 1500)
            a_elo = score_data['last_elo'].get(away_normalized, 1500)
            h_form = score_data['last_form'].get(home_normalized, 0.5)
            a_form = score_data['last_form'].get(away_normalized, 0.5)
            
            # Calculer la différence ELO (impact sur les buts)
            elo_diff = h_elo - a_elo
            # Facteur ELO amélioré: impact plus significatif
            # Différence de 200 points ELO = ~15% d'avantage
            elo_factor = 1 + (elo_diff / 600)  # Facteur plus sensible (au lieu de 1000)
            # Limiter le facteur entre 0.7 et 1.5 pour éviter des extrêmes
            elo_factor = max(0.7, min(1.5, elo_factor))
            
            # Moyennes historiques pour chaque équipe
            # Utiliser les stats du score_predictor si disponibles, sinon calculer depuis MATCHES_DF
            if score_data.get('team_stats') and home_normalized in score_data['team_stats']:
                home_stats = score_data['team_stats'][home_normalized]
                home_avg_scored = home_stats['avg_scored_home']
                home_avg_conceded = home_stats['avg_conceded_home']
            else:
                home_games_as_home = MATCHES_DF[MATCHES_DF['home_team_name'] == home_normalized]
                home_avg_scored = home_games_as_home['home_score'].mean() if not home_games_as_home.empty else score_data.get('global_avg_home_score', 1.5)
                home_avg_conceded = home_games_as_home['away_score'].mean() if not home_games_as_home.empty else score_data.get('global_avg_away_score', 1.0)
            
            if score_data.get('team_stats') and away_normalized in score_data['team_stats']:
                away_stats = score_data['team_stats'][away_normalized]
                away_avg_scored = away_stats['avg_scored_away']
                away_avg_conceded = away_stats['avg_conceded_away']
            else:
                away_games_as_away = MATCHES_DF[MATCHES_DF['away_team_name'] == away_normalized]
                away_avg_scored = away_games_as_away['away_score'].mean() if not away_games_as_away.empty else score_data.get('global_avg_away_score', 1.0)
                away_avg_conceded = away_games_as_away['home_score'].mean() if not away_games_as_away.empty else score_data.get('global_avg_home_score', 1.5)
            
            # Forme récente : moyenne des derniers 5 matches
            def recent_avg_scored(team):
                sub = MATCHES_DF[(MATCHES_DF['home_team_name'] == team) | (MATCHES_DF['away_team_name'] == team)]
                sub = sub.sort_values('date', ascending=False).head(5)
                if sub.empty:
                    return 1.5
                total = 0
                for _, r in sub.iterrows():
                    if r['home_team_name'] == team:
                        total += float(r['home_score'])
                    else:
                        total += float(r['away_score'])
                return total / len(sub) if len(sub) > 0 else 1.5
            
            def recent_avg_conceded(team):
                sub = MATCHES_DF[(MATCHES_DF['home_team_name'] == team) | (MATCHES_DF['away_team_name'] == team)]
                sub = sub.sort_values('date', ascending=False).head(5)
                if sub.empty:
                    return 1.2
                total = 0
                for _, r in sub.iterrows():
                    if r['home_team_name'] == team:
                        total += float(r['away_score'])
                    else:
                        total += float(r['home_score'])
                return total / len(sub) if len(sub) > 0 else 1.2
            
            h_recent_scored = recent_avg_scored(home_normalized)
            a_recent_scored = recent_avg_scored(away_normalized)
            h_recent_conceded = recent_avg_conceded(home_normalized)
            a_recent_conceded = recent_avg_conceded(away_normalized)
            
            # Prédiction pour l'équipe à domicile
            # Combinaison améliorée avec pondérations plus réalistes
            # Base: moyenne historique (50%) + forme récente (30%) + ELO (15%) + défense adverse (5%)
            base_home = (
                home_avg_scored * 0.5 +
                h_recent_scored * 0.3 +
                (home_avg_scored * elo_factor) * 0.15 +
                (a_recent_conceded) * 0.05
            )
            
            # Ajustement basé sur la forme (plage 0.9 à 1.3 au lieu de 0.8 à 1.2)
            form_multiplier_home = 0.9 + h_form * 0.4
            pred_home = base_home * form_multiplier_home
            
            # Avantage à domicile: +0.3 but en moyenne
            pred_home += 0.3
            
            # Prédiction pour l'équipe à l'extérieur
            base_away = (
                away_avg_scored * 0.5 +
                a_recent_scored * 0.3 +
                (away_avg_scored / elo_factor) * 0.15 +
                (h_recent_conceded) * 0.05
            )
            
            # Ajustement basé sur la forme
            form_multiplier_away = 0.9 + a_form * 0.4
            pred_away = base_away * form_multiplier_away
            
            # S'assurer que les valeurs sont réalistes (minimum 0.5, maximum 5.0)
            pred_home = max(0.5, min(5.0, pred_home))
            pred_away = max(0.5, min(5.0, pred_away))
            
            # Arrondir intelligemment (arrondi vers le haut si >= 0.7, sinon vers le bas)
            pred_home = int(pred_home + 0.3) if pred_home >= 0.7 else max(1, int(pred_home))
            pred_away = int(pred_away + 0.3) if pred_away >= 0.7 else max(1, int(pred_away))
            
            # S'assurer qu'au moins un but est marqué par équipe
            pred_home = max(1, pred_home)
            pred_away = max(1, pred_away)
            
            return {
                'score': f"{pred_home} - {pred_away}",
                'home_score': pred_home,
                'away_score': pred_away,
                'resultat': 'Home Win' if pred_home > pred_away else ('Away Win' if pred_away > pred_home else 'Draw'),
                'note': 'Prédiction réussite...'
            }
        except Exception as e:
            # En cas d'erreur, retourner une prédiction basique mais pas toujours 1-1
            import random
            base_home = random.randint(1, 2)
            base_away = random.randint(0, 2)
            return {
                'score': f"{base_home} - {base_away}",
                'home_score': base_home,
                'away_score': base_away,
                'resultat': 'Draw',
                'note': f'Prédiction de base (erreur: {str(e)})'
            }

    # Initialiser les variables pour les statistiques
    stats_data = None
    h2h_data = None
    team_comparison = None
    
    # Toujours utiliser la prédiction logique améliorée (sauf si erreur de formulaire)
    if request.method == 'POST' and not error and home_team and away_team:
        logic = predict_score_logic(home_team, away_team)
        result = logic
        
        # Calculer des statistiques détaillées
        try:
            # Normaliser les noms
            home_key = home_team.strip().lower()
            away_key = away_team.strip().lower()
            home_normalized = predictor.name_map.get(home_key, home_team)
            away_normalized = predictor.name_map.get(away_key, away_team)
            
            # Stats ELO et forme
            h_elo = predictor.last_elo.get(home_normalized, 1500)
            a_elo = predictor.last_elo.get(away_normalized, 1500)
            h_form = predictor.last_form.get(home_normalized, 0.5)
            a_form = predictor.last_form.get(away_normalized, 0.5)
            
            # Calculer les statistiques des équipes
            def get_team_stats(team):
                team_matches = MATCHES_DF[
                    (MATCHES_DF['home_team_name'] == team) | 
                    (MATCHES_DF['away_team_name'] == team)
                ].sort_values('date', ascending=False)
                
                last_5 = team_matches.head(5)
                if last_5.empty:
                    return {
                        'wins': 0, 'draws': 0, 'losses': 0,
                        'goals_for': 0, 'goals_against': 0,
                        'avg_goals_for': 1.5, 'avg_goals_against': 1.2,
                        'win_rate': 0
                    }
                
                wins = draws = losses = 0
                goals_for = goals_against = 0
                
                for _, row in last_5.iterrows():
                    is_home = row['home_team_name'] == team
                    if is_home:
                        gf = float(row['home_score'])
                        ga = float(row['away_score'])
                    else:
                        gf = float(row['away_score'])
                        ga = float(row['home_score'])
                    
                    goals_for += gf
                    goals_against += ga
                    
                    if gf > ga:
                        wins += 1
                    elif gf < ga:
                        losses += 1
                    else:
                        draws += 1
                
                total = len(last_5)
                return {
                    'wins': wins,
                    'draws': draws,
                    'losses': losses,
                    'goals_for': goals_for,
                    'goals_against': goals_against,
                    'avg_goals_for': goals_for / total if total > 0 else 0,
                    'avg_goals_against': goals_against / total if total > 0 else 0,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'goal_diff': goals_for - goals_against
                }
            
            home_stats = get_team_stats(home_normalized)
            away_stats = get_team_stats(away_normalized)
            
            # Head-to-head
            h2h_matches = compute_h2h(home_normalized, away_normalized, n=10)
            h2h_summary = compute_h2h_summary(home_normalized, away_normalized)
            
            # Derniers matchs
            last_home_matches = compute_last_matches(home_normalized, n=5)
            last_away_matches = compute_last_matches(away_normalized, n=5)
            
            # Calculer quelques scores probables (basés sur les moyennes)
            def calculate_score_probabilities(home_avg, away_avg):
                """Calcule les probabilités de différents scores"""
                # Scores les plus probables basés sur les moyennes
                scores = []
                for h in range(max(0, int(home_avg - 1)), int(home_avg + 2) + 1):
                    for a in range(max(0, int(away_avg - 1)), int(away_avg + 2) + 1):
                        # Probabilité approximative (Poisson simplifié)
                        prob = min(100, max(5, 
                            (50 * (1 - abs(h - home_avg) / (home_avg + 1))) *
                            (50 * (1 - abs(a - away_avg) / (away_avg + 1))) / 100
                        ))
                        scores.append({
                            'score': f"{h} - {a}",
                            'probability': round(prob, 1)
                        })
                # Trier par probabilité et prendre les top 5
                scores.sort(key=lambda x: x['probability'], reverse=True)
                return scores[:6]
            
            score_probs = calculate_score_probabilities(
                home_stats['avg_goals_for'],
                away_stats['avg_goals_for']
            )
            
            stats_data = {
                'home': {
                    'elo': round(h_elo, 0),
                    'form': round(h_form * 100, 1),
                    'stats': home_stats,
                    'name': home_normalized
                },
                'away': {
                    'elo': round(a_elo, 0),
                    'form': round(a_form * 100, 1),
                    'stats': away_stats,
                    'name': away_normalized
                },
                'score_probabilities': score_probs
            }
            
            h2h_data = {
                'matches': h2h_matches,
                'summary': h2h_summary
            }
            
            team_comparison = {
                'home_last_matches': last_home_matches,
                'away_last_matches': last_away_matches
            }
            
        except Exception as e:
            # En cas d'erreur, continuer sans les stats détaillées
            pass

    return render_template(
        "score.html",
        current_page="score",
        teams=TEAMS,
        home_team=home_team,
        away_team=away_team,
        result=result,
        error=error,
        stats_data=stats_data,
        h2h_data=h2h_data,
        team_comparison=team_comparison,
    )


# Route d'administration pour lancer le chargement du module MLValod manuellement
@app.route('/admin/load_score_model')
def admin_load_score_model():
    global _predire_loading
    if _predire_loading:
        return "Chargement déjà en cours", 202
    # lancer le loader en thread
    threading.Thread(target=_load_mlvalod, daemon=True).start()
    return "Chargement démarré", 202


# --- Pages Stats / À propos (simples pour l’instant) ---
@app.route("/stats")
def stats():
    return render_template(
        "home.html",   # tu peux créer un stats.html plus tard
        current_page="stats"
    )


@app.route("/about")
def about():
    return render_template(
        "home.html",   # tu peux créer un about.html plus tard
        current_page="about"
    )


# ================================
# Lancement app
# ================================
def signal_handler(sig, frame):
    """Gère proprement l'arrêt du serveur avec Ctrl+C"""
    print('\n\nArrêt du serveur Flask...')
    sys.exit(0)

if __name__ == "__main__":
    # Enregistrer le gestionnaire de signal pour Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        print('\n\nArrêt du serveur Flask...')
        sys.exit(0)
