import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
data_game = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'games.csv'))

# Features selected
features = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 
            'firstDragon', 'firstRiftHerald', 't1_towerKills', 't1_inhibitorKills',
            't1_baronKills', 't1_dragonKills', 't2_towerKills', 't2_inhibitorKills',
            't2_baronKills', 't2_dragonKills']

ml_data = data_game[features + ['winner']].dropna()

X = ml_data[features]
y = ml_data['winner']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features (optional for Naive Bayes, but can help)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== NAIVE BAYES CLASSIFIER =====

print("\n" + "="*50)
print("NAIVE BAYES ANALYSIS")
print("="*50)

# Create and train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred_nb = nb_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"\nNaive Bayes Accuracy: {accuracy_nb:.4f}")

# Classification Report
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb, target_names=['Team 1', 'Team 2']))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Team 1', 'Team 2'], 
            yticklabels=['Team 1', 'Team 2'])
plt.title('Confusion Matrix - Naive Bayes', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Winner', fontsize=12)
plt.ylabel('Actual Winner', fontsize=12)
plt.tight_layout()
plt.show(block=False)

# Feature probabilities visualization
print("\n" + "="*50)
print("CLASS PROBABILITIES")
print("="*50)
print(f"Prior probability Team 1: {nb_classifier.class_prior_[0]:.4f}")
print(f"Prior probability Team 2: {nb_classifier.class_prior_[1]:.4f}")

# Plot feature means per class
feature_means = pd.DataFrame({
    'Feature': features,
    'Team 1 Mean': nb_classifier.theta_[0],
    'Team 2 Mean': nb_classifier.theta_[1]
})

plt.figure(figsize=(12, 6))
x = np.arange(len(features))
width = 0.35

plt.bar(x - width/2, feature_means['Team 1 Mean'], width, label='Team 1', alpha=0.8, color='skyblue')
plt.bar(x + width/2, feature_means['Team 2 Mean'], width, label='Team 2', alpha=0.8, color='salmon')

plt.xlabel('Features', fontsize=12)
plt.ylabel('Mean Value', fontsize=12)
plt.title('Feature Means by Class (Naive Bayes)', fontsize=14, fontweight='bold')
plt.xticks(x, features, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show(block=False)

plt.pause(0.1)
input("Press Enter to close all plots...")