import time
# pyrefly: ignore [missing-import]
import joblib
import json
import random
import pandas as pd

from src.config import MODEL_PATH, LOG_PATH
from src.utils import generate_sample, get_label_name

from collections.abc import AsyncIterable, Iterable
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# class Data(BaseModel):
#     alpha:float
#     beta:float
#     theta:float
#     gamma:float
#     attention:float
#     scroll_speed:float
#     prediction:int


# Load trained model
model = joblib.load(MODEL_PATH)

def generate_stream(duration=300):
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
        real_name = get_label_name(real_label)
        pred_name = get_label_name(pred)

        data = {
            "alpha": sample[0],
            "beta": sample[1],
            "theta": sample[2],
            "gamma": sample[3],
            "attention": sample[4],
            "scroll_speed": sample[5],
            "prediction": int(pred)
        }
        yield json.dumps(data) + "\n"

        print(f"Real: {real_name} | Predicted: {pred_name}")

        if pred == 1:
            print("⚠ ALERT: FOMO DETECTED!\n")

        logs.append(sample + [real_label, pred])

        time.sleep(1)

    # Save logs
    # df = pd.DataFrame(logs, columns=[
    #     "alpha",
    #     "beta",
    #     "theta",
    #     "gamma",
    #     "attention",
    #     "scroll_speed",
    #     "real_label",
    #     "predicted"
    # ])

    # df.to_csv(LOG_PATH, index=False)

    # print("\n✅ Simulation completed!")
    # print("📁 Log saved at:", LOG_PATH)

@app.get("/data_stream")
def data_stream():
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)