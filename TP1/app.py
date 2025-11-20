import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

# Load data
data_game = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'games.csv'))

# Load champion info
with open(os.path.join(os.path.dirname(__file__), 'dataset', 'champion_info_2.json'), 'r') as f:
    champion_data = json.load(f)

# Create a mapping of champion ID to name
champ_id_to_name = {}
for champ_key, champ_info in champion_data['data'].items():
    champ_id = int(champ_info['id'])
    champ_name = champ_info['name']
    champ_id_to_name[champ_id] = champ_name

# ===== FIRST INHIBITOR ANALYSIS =====

first_inhibitor = sns.catplot(x="firstInhibitor", y="winner", data=data_game, kind="bar", palette="Set2", hue="firstInhibitor")
plt.title('Impact du premier inhibiteur sur la victoire', fontsize=16, fontweight='bold')
plt.show(block=False)

# ===== FIRST BARON ANALYSIS =====

first_baron = sns.catplot(x="firstBaron", y="winner", data=data_game, kind="bar", palette="Set3", hue="firstBaron")
plt.title('Impact du premier baron sur la victoire', fontsize=16, fontweight='bold')
plt.show(block=False)

# ===== CHAMPION WINRATE ANALYSIS =====

champion_games = []

for idx, row in data_game.iterrows():
    winner = row['winner']
    
    # Team 1 champions
    for i in range(1, 6):
        champ_id = row[f't1_champ{i}id']
        if pd.notna(champ_id) and champ_id != -1:
            champion_games.append({
                'champion_id': int(champ_id),
                'won': 1 if winner == 1 else 0
            })
    
    # Team 2 champions
    for i in range(1, 6):
        champ_id = row[f't2_champ{i}id']
        if pd.notna(champ_id) and champ_id != -1:
            champion_games.append({
                'champion_id': int(champ_id),
                'won': 1 if winner == 2 else 0
            })

champ_df = pd.DataFrame(champion_games)

# Calculate winrate for each champion
champ_stats = champ_df.groupby('champion_id').agg(
    games=('won', 'count'),
    wins=('won', 'sum')
).reset_index()

champ_stats['winrate'] = (champ_stats['wins'] / champ_stats['games'] * 100)

# Filter champions with at least 30 games for statistical relevance
champ_stats_filtered = champ_stats[champ_stats['games'] >= 30].copy()

# Add champion names
champ_stats_filtered['champion_name'] = champ_stats_filtered['champion_id'].map(champ_id_to_name)

# Sort by winrate and get top 15
top_winrate = champ_stats_filtered.nlargest(15, 'winrate')

# Plot top winrate champions
plt.figure(figsize=(12, 6))
sns.barplot(data=top_winrate, x='winrate', y='champion_name', palette='viridis', hue='winrate')
plt.title('Top 15 Champions by Winrate (min 30 games)', fontsize=14, fontweight='bold')
plt.xlabel('Winrate (%)', fontsize=12)
plt.ylabel('Champion', fontsize=12)
plt.xlim(0, 100)
for i, (idx, row) in enumerate(top_winrate.iterrows()):
    plt.text(row['winrate'] + 1, i, f"{row['winrate']:.1f}% ({row['games']} games)", 
             va='center', fontsize=9)
plt.tight_layout()
plt.show(block=False)

# ===== FIRST BLOOD ANALYSIS =====

first_blood_data = []

for idx, row in data_game.iterrows():
    fb_team = row['firstBlood']
    
    if pd.notna(fb_team) and fb_team != 0:
        if fb_team == 1:
            for i in range(1, 6):
                champ_id = row[f't1_champ{i}id']
                if pd.notna(champ_id) and champ_id != -1:
                    first_blood_data.append({'champion_id': int(champ_id)})
        elif fb_team == 2:
            for i in range(1, 6):
                champ_id = row[f't2_champ{i}id']
                if pd.notna(champ_id) and champ_id != -1:
                    first_blood_data.append({'champion_id': int(champ_id)})

fb_df = pd.DataFrame(first_blood_data)

# Count first bloods by champion
fb_stats = fb_df['champion_id'].value_counts().reset_index()
fb_stats.columns = ['champion_id', 'fb_count']

# Add champion names
fb_stats['champion_name'] = fb_stats['champion_id'].map(champ_id_to_name)

# Get top 10
top_fb = fb_stats.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_fb, x='fb_count', y='champion_name', palette='rocket', hue='fb_count')
plt.title('Top 10 Champions in Teams with First Blood', fontsize=14, fontweight='bold')
plt.xlabel('Number of Games with First Blood', fontsize=12)
plt.ylabel('Champion', fontsize=12)
for i, (idx, row) in enumerate(top_fb.iterrows()):
    plt.text(row['fb_count'] + 2, i, f"{row['fb_count']}", va='center', fontsize=10)
plt.tight_layout()
plt.show(block=False)


# ===== HEATMAP FIRST OBJECTIVES ANALYSIS =====

plt.figure(figsize=(10, 8))
sns.heatmap(data_game[["firstInhibitor","firstBaron","firstRiftHerald","winner"]].corr(),annot = True)
plt.title('Correlation between First Inhibitor, First Baron, First Rift Herald et Winner', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show(block=False)

plt.pause(0.1)
input("Press Enter to close all plots...")