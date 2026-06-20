import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from config import DATA_PATH, MODEL_PATH
import os


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["label", "timestamp"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Model Evaluation:\n")
    print(classification_report(y_test, y_pred))

    dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(dir):
        os.makedirs(dir)

    joblib.dump(model, MODEL_PATH)
    print("✅ Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()