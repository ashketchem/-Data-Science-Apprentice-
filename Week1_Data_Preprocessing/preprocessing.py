import pandas as pd
from sklearn.preprocessing import StandardScaler

print("--- WEEK 1: Data Acquisition & Pre-processing ---")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Imputing Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Outlier Handling via IQR
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= (Q1 - 1.5 * IQR)) & (df['Fare'] <= (Q3 + 1.5 * IQR))]

# Scaling
scaler = StandardScaler()
df['Fare_Scaled'] = scaler.fit_transform(df[['Fare']])
df['Age_Scaled'] = scaler.fit_transform(df[['Age']])

df.to_csv('Week1_Data_Preprocessing/cleaned_titanic.csv', index=False)
print("Saved preprocessed dataset to Week1_Data_Preprocessing/cleaned_titanic.csv")
