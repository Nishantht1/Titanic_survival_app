import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split

# Load Titanic dataset (local or from URL)
df = pd.read_csv("titanic.csv")

# Preprocess the data (same steps you did before)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df = df.drop(columns=['Cabin', 'Name', 'Ticket', 'Embarked'])

# Train/test split
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'titanic_model.pkl')
