import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Week1_Data_Preprocessing/cleaned_titanic.csv')
sns.set_theme(style="whitegrid")

# Plot 1: Survival Rate by Sex
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='Sex', y='Survived', errorbar=None, palette='Blues_d', hue='Sex', legend=False)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Probability')
plt.savefig('Week2_EDA/survival_by_gender.png')
plt.close()

# Plot 2: Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Survived', 'Pclass', 'Age_Scaled', 'Fare_Scaled']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.savefig('Week2_EDA/correlation_heatmap.png')
plt.close()

print("Saved visual reports in Week2_EDA/ directory")
