"""
Script pour créer le fichier score_predictor.pkl
Fichier séparé pour les prédictions de score (ne touche pas à wc_web_predictor.pkl)
"""
import pickle
import json
import pandas as pd
import numpy as np
import sys
import os

print("📊 Création du prédicteur de score (score_predictor.pkl)...")

# 1. Charger la liste des équipes
print("   → Chargement de teams_list.json...")
with open('teams_list.json', 'r', encoding='utf-8') as f:
    teams = json.load(f)

# 2. Construire name_map (normalisation des noms)
name_map = {t.strip().lower(): t for t in teams}
print(f"   → {len(teams)} équipes chargées")

# 3. Charger les données historiques
print("   → Chargement de matches_history.csv...")
try:
    matches_df = pd.read_csv('matches_history.csv')
    matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
    matches_df = matches_df.sort_values('date')
    print(f"   → {len(matches_df)} matchs historiques chargés")
except FileNotFoundError:
    print("   ⚠️  matches_history.csv non trouvé, utilisation de valeurs par défaut")
    matches_df = pd.DataFrame()

# 4. Calculer les valeurs ELO pour chaque équipe
print("   → Calcul des ratings ELO...")
last_elo = {}
K = 32  # Facteur K pour le calcul ELO

# Initialiser tous les ELO à 1500
for team in teams:
    last_elo[team] = 1500.0

# Calculer ELO en parcourant les matchs historiques
if not matches_df.empty:
    for idx, row in matches_df.iterrows():
        home_team = row.get('home_team_name', '')
        away_team = row.get('away_team_name', '')
        home_score = row.get('home_score', 0)
        away_score = row.get('away_score', 0)
        
        if pd.isna(home_team) or pd.isna(away_team) or home_team == '' or away_team == '':
            continue
            
        # Normaliser les noms
        home_key = home_team.strip().lower()
        away_key = away_team.strip().lower()
        home = name_map.get(home_key, home_team)
        away = name_map.get(away_key, away_team)
        
        if home not in last_elo:
            last_elo[home] = 1500.0
        if away not in last_elo:
            last_elo[away] = 1500.0
        
        # Calculer le résultat attendu
        elo_home = last_elo[home]
        elo_away = last_elo[away]
        
        expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
        expected_away = 1 - expected_home
        
        # Déterminer le résultat réel
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score < away_score:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5
        
        # Mettre à jour les ELO
        last_elo[home] = elo_home + K * (actual_home - expected_home)
        last_elo[away] = elo_away + K * (actual_away - expected_away)
    
    print(f"   → ELO calculés pour {len(last_elo)} équipes")
else:
    # Valeurs par défaut si pas de données
    for team in teams:
        last_elo[team] = 1500.0
    print("   → ELO par défaut (1500) pour toutes les équipes")

# 5. Calculer la forme récente (derniers 5 matchs)
print("   → Calcul de la forme récente...")
last_form = {}

if not matches_df.empty:
    for team in teams:
        # Trouver les derniers 5 matchs de cette équipe
        team_matches = matches_df[
            (matches_df['home_team_name'] == team) | 
            (matches_df['away_team_name'] == team)
        ].sort_values('date', ascending=False).head(5)
        
        if team_matches.empty:
            last_form[team] = 0.5  # Forme neutre par défaut
        else:
            wins = 0
            draws = 0
            losses = 0
            
            for _, match in team_matches.iterrows():
                home_team = match.get('home_team_name', '')
                away_team = match.get('away_team_name', '')
                home_score = match.get('home_score', 0)
                away_score = match.get('away_score', 0)
                
                if home_team == team:
                    if home_score > away_score:
                        wins += 1
                    elif home_score < away_score:
                        losses += 1
                    else:
                        draws += 1
                else:  # away_team == team
                    if away_score > home_score:
                        wins += 1
                    elif away_score < home_score:
                        losses += 1
                    else:
                        draws += 1
            
            # Forme = (wins * 1.0 + draws * 0.5) / total_matches
            total = len(team_matches)
            if total > 0:
                form = (wins * 1.0 + draws * 0.5) / total
            else:
                form = 0.5
            
            last_form[team] = form
    
    print(f"   → Forme calculée pour {len(last_form)} équipes")
else:
    # Valeurs par défaut
    for team in teams:
        last_form[team] = 0.5
    print("   → Forme par défaut (0.5) pour toutes les équipes")

# 6. Calculer les statistiques historiques moyennes pour chaque équipe
print("   → Calcul des statistiques historiques...")
team_stats = {}

if not matches_df.empty:
    for team in teams:
        # Matchs à domicile
        home_matches = matches_df[matches_df['home_team_name'] == team]
        # Matchs à l'extérieur
        away_matches = matches_df[matches_df['away_team_name'] == team]
        
        # Moyennes de buts marqués
        avg_scored_home = home_matches['home_score'].mean() if not home_matches.empty else 1.5
        avg_scored_away = away_matches['away_score'].mean() if not away_matches.empty else 1.0
        
        # Moyennes de buts encaissés
        avg_conceded_home = home_matches['away_score'].mean() if not home_matches.empty else 1.2
        avg_conceded_away = away_matches['home_score'].mean() if not away_matches.empty else 1.5
        
        team_stats[team] = {
            'avg_scored_home': float(avg_scored_home) if not pd.isna(avg_scored_home) else 1.5,
            'avg_scored_away': float(avg_scored_away) if not pd.isna(avg_scored_away) else 1.0,
            'avg_conceded_home': float(avg_conceded_home) if not pd.isna(avg_conceded_home) else 1.2,
            'avg_conceded_away': float(avg_conceded_away) if not pd.isna(avg_conceded_away) else 1.5,
        }
    
    print(f"   → Statistiques calculées pour {len(team_stats)} équipes")
else:
    # Valeurs par défaut
    for team in teams:
        team_stats[team] = {
            'avg_scored_home': 1.5,
            'avg_scored_away': 1.0,
            'avg_conceded_home': 1.2,
            'avg_conceded_away': 1.5,
        }
    print("   → Statistiques par défaut pour toutes les équipes")

# 7. Créer un dictionnaire avec toutes les données nécessaires pour la prédiction de score
score_predictor_data = {
    'last_elo': last_elo,
    'last_form': last_form,
    'name_map': name_map,
    'team_stats': team_stats,
    'global_avg_home_score': float(matches_df['home_score'].mean()) if not matches_df.empty else 1.5,
    'global_avg_away_score': float(matches_df['away_score'].mean()) if not matches_df.empty else 1.0,
}

# 8. Sauvegarder dans le fichier .pkl
print("   → Sauvegarde dans score_predictor.pkl...")
with open('score_predictor.pkl', 'wb') as f:
    pickle.dump(score_predictor_data, f)

print("\n✅ Fichier score_predictor.pkl créé avec succès!")
print(f"   - {len(teams)} équipes")
print(f"   - ELO calculés: {len(last_elo)} équipes")
print(f"   - Forme calculée: {len(last_form)} équipes")
print(f"   - Statistiques: {len(team_stats)} équipes")
print(f"   - ELO moyen: {np.mean(list(last_elo.values())):.1f}")
print(f"   - Forme moyenne: {np.mean(list(last_form.values())):.2f}")
print("\n⚠️  Note: Le fichier wc_web_predictor.pkl n'a PAS été modifié")

