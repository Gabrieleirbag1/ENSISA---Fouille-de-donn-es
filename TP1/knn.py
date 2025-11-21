import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
# ===== MACHINE LEARNING ANALYSIS =====

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

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== K-NEAREST NEIGHBORS (KNN) =====

print("\n" + "="*50)
print("K-NEAREST NEIGHBORS ANALYSIS")
print("="*50)

# Test different values of k
k_values = [3, 5, 7, 9, 11]
knn_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    knn_accuracies.append(accuracy)
    print(f"\nKNN with k={k}: Accuracy = {accuracy:.4f}")

# Use best k value
best_k = k_values[np.argmax(knn_accuracies)]
print(f"\nBest k value: {best_k} with accuracy: {max(knn_accuracies):.4f}")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn = knn_best.predict(X_test_scaled)

print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn))

# Confusion Matrix for KNN
plt.figure(figsize=(8, 6))
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Team 1', 'Team 2'], 
            yticklabels=['Team 1', 'Team 2'])
plt.title(f'Confusion Matrix - KNN (k={best_k})', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Winner', fontsize=12)
plt.ylabel('Actual Winner', fontsize=12)
plt.tight_layout()
plt.show(block=False)

plt.pause(0.1)
input("Press Enter to close all plots...")