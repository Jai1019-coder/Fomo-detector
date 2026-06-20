import time
import joblib
import random
import pandas as pd

from config import MODEL_PATH, LOG_PATH
from utils import generate_sample, get_label_name

# Load trained model
model = joblib.load(MODEL_PATH)


def run_simulation(duration=300):
    print("🚀 Running simulation for 5 minutes...\n")

    logs = []
    start = time.time()

    while time.time() - start < duration:

        # Simulate real state
        real_label = random.choice([0, 1, 2, 3])

        # Generate sample
        sample = generate_sample(real_label)

        # ✅ FIX: column names must EXACTLY match training
        sample_df = pd.DataFrame([sample], columns=[
            "alpha",
            "beta",
            "theta",
            "gamma",
            "attention",
            "scroll_speed"   # 🔥 FIXED NAME
        ])

        # Predict
        pred = model.predict(sample_df)[0]
        print(sample_df)
        real_name = get_label_name(real_label)
        pred_name = get_label_name(pred)

        print(f"Real: {real_name} | Predicted: {pred_name}")

        if pred == 1:
            print("⚠ ALERT: FOMO DETECTED!\n")

        logs.append(sample + [real_label, pred])

        time.sleep(1)

    # Save logs
    df = pd.DataFrame(logs, columns=[
        "alpha",
        "beta",
        "theta",
        "gamma",
        "attention",
        "scroll_speed",
        "real_label",
        "predicted"
    ])

    df.to_csv(LOG_PATH, index=False)

    print("\n✅ Simulation completed!")
    print("📁 Log saved at:", LOG_PATH)


if __name__ == "__main__":
    run_simulation()