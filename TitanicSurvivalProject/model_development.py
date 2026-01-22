import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("titanic.csv")   # Ensure dataset is in same folder
df.head()

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

df = df[features + [target]]

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

with open("titanic_survival_model.pkl", "wb") as file:
    pickle.dump((model, scaler), file)

with open("titanic_survival_model.pkl", "rb") as file:
    loaded_model, loaded_scaler = pickle.load(file)

sample = [[3, 0, 25, 7.25, 2]]   # Example passenger
sample_scaled = loaded_scaler.transform(sample)

prediction = loaded_model.predict(sample_scaled)
print("Survived" if prediction[0] == 1 else "Did Not Survive")
