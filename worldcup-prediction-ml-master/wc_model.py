import pandas as pd

class WorldCupPredictor:
    def __init__(self, pipe_rf, last_elo, last_form, name_map):
        self.pipe_rf = pipe_rf
        self.last_elo = last_elo
        self.last_form = last_form
        self.name_map = name_map

    def predict(self, home_input, away_input):
        h_key = home_input.strip().lower()
        a_key = away_input.strip().lower()

        home = self.name_map.get(h_key)
        away = self.name_map.get(a_key)

        if home is None or away is None:
            return None, "❌ Une des équipes n'existe pas dans le dataset."

        h_elo = self.last_elo.get(home, 1500)
        a_elo = self.last_elo.get(away, 1500)
        h_form = self.last_form.get(home, 0.5)
        a_form = self.last_form.get(away, 0.5)

        ex = pd.DataFrame({
            'home_team_name': [home],
            'away_team_name': [away],
            'home_elo': [h_elo],
            'away_elo': [a_elo],
            'home_recent_form': [h_form],
            'away_recent_form': [a_form],
            'elo_diff': [h_elo - a_elo],
            'form_diff': [h_form - a_form],
            'goal_diff_avg': [0],
            'is_world_cup': [1],
            'home_advantage': [1]
        })

        pred_class = self.pipe_rf.predict(ex)[0]
        proba = self.pipe_rf.predict_proba(ex)[0]

        if int(pred_class) == 0:
            text = f"Match nul entre {home} et {away}."
        elif int(pred_class) == 1:
            text = f"Victoire de {home}."
        else:
            text = f"Victoire de {away}."

        details = {
            "home": home,
            "away": away,
            "pred_class": int(pred_class),
            "proba": proba.tolist(),
            "h_elo": float(h_elo),
            "a_elo": float(a_elo),
            "h_form": float(h_form),
            "a_form": float(a_form),
        }

        return text, details
