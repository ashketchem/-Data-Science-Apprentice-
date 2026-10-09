import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Week1_Data_Preprocessing/cleaned_titanic.csv')
df['Sex_Code'] = df['Sex'].map({'male': 0, 'female': 1})
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

features = ['Pclass', 'Sex_Code', 'Age_Scaled', 'Fare_Scaled', 'FamilySize', 'IsAlone']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation
baseline = RandomForestClassifier(random_state=42)
cv_score = cross_val_score(baseline, X, y, cv=5).mean()
print("Mean Cross-Validation:", round(cv_score, 4))

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Tuned Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("
Classification Report:
", classification_report(y_test, y_pred))
