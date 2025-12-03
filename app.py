from flask import Flask, request, render_template
import pickle
import pandas as pd
import json
import numpy as np   # pour la moyenne des probas


# ================================
# Classe WorldCupPredictor
# ================================
class WorldCupPredictor:
    def __init__(self, pipe_rf, last_elo, last_form, name_map):
        """
        pipe_rf   : pipeline RandomForest entraîné (scikit-learn)
        last_elo  : dict {team_name: elo}
        last_form : dict {team_name: forme récente}
        name_map  : dict pour corriger / normaliser les noms d'équipes
        """
        self.pipe_rf = pipe_rf
        self.last_elo = last_elo
        self.last_form = last_form
        self.name_map = name_map

    def predict(self, home_input, away_input):
        """
        Prédiction SYMÉTRIQUE :
        - on prédit une fois avec (home, away)
        - une deuxième fois avec (away, home)
        - on remet tout dans le référentiel (home, away) original
        - on moyenne les probabilités des deux sens
        """

        # 1) Normaliser le texte saisi
        h_key = home_input.strip().lower()
        a_key = away_input.strip().lower()

        home = self.name_map.get(h_key)
        away = self.name_map.get(a_key)

        if home is None or away is None:
            return None, "❌ Une des équipes n'existe pas dans le dataset."

        # 2) Récupérer ELO et forme
        h_elo = self.last_elo.get(home, 1500)
        a_elo = self.last_elo.get(away, 1500)
        h_form = self.last_form.get(home, 0.5)
        a_form = self.last_form.get(away, 0.5)

        # ==========================
        # 3) PREMIÈRE PRÉDICTION
        #    home = home, away = away
        # ==========================
        ex1 = pd.DataFrame({
            "home_team_name": [home],
            "away_team_name": [away],
            "home_elo": [h_elo],
            "away_elo": [a_elo],
            "home_recent_form": [h_form],
            "away_recent_form": [a_form],
            "elo_diff": [h_elo - a_elo],
            "form_diff": [h_form - a_form],
            "goal_diff_avg": [0],   # inconnu pour futur match → 0
            "is_world_cup": [1],
            "home_advantage": [1],
        })

        proba1 = self.pipe_rf.predict_proba(ex1)[0]  # [P(nul), P(home), P(away)]

        # ==========================
        # 4) DEUXIÈME PRÉDICTION
        #    home = away, away = home
        # ==========================
        ex2 = pd.DataFrame({
            "home_team_name": [away],
            "away_team_name": [home],
            "home_elo": [a_elo],
            "away_elo": [h_elo],
            "home_recent_form": [a_form],
            "away_recent_form": [h_form],
            "elo_diff": [a_elo - h_elo],
            "form_diff": [a_form - h_form],
            "goal_diff_avg": [0],
            "is_world_cup": [1],
            "home_advantage": [1],
        })

        proba2 = self.pipe_rf.predict_proba(ex2)[0]  # [P(nul), P(home2), P(away2)]

        # ==========================
        # 5) REMISE DANS LE RÉFÉRENTIEL ORIGINAL (home, away)
        # ==========================
        p_draw = (proba1[0] + proba2[0]) / 2
        p_home = (proba1[1] + proba2[2]) / 2
        p_away = (proba1[2] + proba2[1]) / 2

        proba_sym = np.array([p_draw, p_home, p_away])

        # 6) Classe finale (0 = nul, 1 = home, 2 = away)
        pred_class = int(np.argmax(proba_sym))

        # 7) Construire un texte lisible
        if pred_class == 0:
            text = f"Match nul entre {home} et {away}."
        elif pred_class == 1:
            text = f"Victoire de {home}."
        else:
            text = f"Victoire de {away}."

        details = {
            "home": home,
            "away": away,
            "pred_class": pred_class,
            "proba": proba_sym.tolist(),  # [P_nul, P_home, P_away]
            "h_elo": float(h_elo),
            "a_elo": float(a_elo),
            "h_form": float(h_form),
            "a_form": float(a_form),
        }
        return text, details


# ================================
# Chargement données & Flask app
# ================================
app = Flask(__name__)

# Liste des équipes pour les listes déroulantes
with open("teams_list.json", "r", encoding="utf-8") as f:
    TEAMS = json.load(f)

# Objet prédicteur complet (pipe_rf + ELO + forme + name_map)
with open("wc_web_predictor.pkl", "rb") as f:
    predictor: WorldCupPredictor = pickle.load(f)

# Historique des matchs (matches_history.csv généré depuis matches.csv)
MATCHES_DF = pd.read_csv("matches_history.csv")
MATCHES_DF["date"] = pd.to_datetime(MATCHES_DF["date"], errors="coerce")


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

    h2h_matches = []
    h2h_summary = None
    last_home_matches = []
    last_away_matches = []
    analysis_text = None

    if request.method == "POST":
        home_team = request.form.get("home_team", "").strip()
        away_team = request.form.get("away_team", "").strip()

        if home_team and away_team:
            text, details = predictor.predict(home_team, away_team)

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
                h2h_matches = compute_h2h(home_team, away_team, n=5)
                h2h_summary = compute_h2h_summary(home_team, away_team)
                last_home_matches = compute_last_matches(home_team, n=5)
                last_away_matches = compute_last_matches(away_team, n=5)

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
    )


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
if __name__ == "__main__":
    app.run(debug=True)
