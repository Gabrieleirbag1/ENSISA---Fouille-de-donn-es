import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# question 1.15
df = pd.read_csv('http://germain-forestier.info/dataset/cockroft.csv', index_col='ID')
print(df.head(10))
plt.scatter(df["age"],df["poids"])
plt.ylabel('poids')
plt.xlabel('age')
plt.plot()
plt.show()

df["clairance cockroft"] = ((140 - df["age"]) * df["poids"] * 1.04) / df["creatininemie"]
plt.scatter(df["clairance"], df["clairance cockroft"])
plt.xlabel("Clairance calculée par la formule de Cockroft")
plt.ylabel("Clairance mesurée sur les urines de 24h")
plt.title("Clairance de Cockcroft")
plt.plot()
plt.show()

