import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "titanic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "titanic_survival_model.pkl")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

# ===============================
# SELECT FEATURES
# ===============================
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']]

# ===============================
# HANDLE MISSING VALUES
# ===============================
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ===============================
# ENCODE CATEGORICAL DATA
# ===============================
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ===============================
# SPLIT DATA
# ===============================
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# SCALE FEATURES
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# TRAIN MODEL
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# EVALUATE
# ===============================
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy:.2f}")

# ===============================
# SAVE MODEL + SCALER
# ===============================
with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, scaler), f)

print("Model saved successfully!")
print("File:", MODEL_PATH)
