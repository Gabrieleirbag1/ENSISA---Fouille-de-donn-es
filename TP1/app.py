import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

data_game = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'games.csv'))

first_inhibitor = sns.catplot(x = "firstInhibitor", y = "winner", data = data_game, kind = "bar", palette="Set2", hue="firstInhibitor")
plt.title('Impact du premier inhibiteur sur la victoire')
plt.show(block=False)

first_baron = sns.catplot(x = "firstBaron",y = "winner", data = data_game,kind = "bar", palette="Set3", hue="firstBaron")
plt.title('Impact du premier baron sur la victoire')
plt.show(block=False)

# Keep the script running so windows don't close immediately
plt.pause(0.1)
input("Press Enter to close all plots...")