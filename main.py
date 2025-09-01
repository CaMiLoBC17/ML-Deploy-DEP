import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from datetime import datetime, date

scaler = joblib.load("scaler.joblib")
best_model = joblib.load("best_model.joblib")

history = []

app = FastAPI()

@app.get("/")
async def welcome():
    return {"message": "Your API is working..."}

@app.get("/predictions")
async def get_predictions():
    return history

@app.get("/predictions/{id}")
async def get_prediction(id: int):
    for row in history:
        if row["index_x"] == id:
            return row
    raise HTTPException(status_code=404, detail="Prediction not found")

@app.post("/predict")
async def create_upload_file(file: UploadFile):
    file_content = await file.read()
    X_new = pd.read_csv(BytesIO(file_content))
    X_scaled = scaler.transform(X_new)
    y_new = best_model.predict(X_scaled)
    y_new_df = pd.DataFrame(y_new, columns=["prediction"])
    current_date = date.today()
    predicted = []

    for i in list(y_new_df["prediction"]):
        row = {"date": current_date, "index_x": len(history), "prediction": i}
        history.append(row)
        predicted.append(row)

    return predicted

@app.put("/predict/{id}")
async def update_prediction(id: int, file: UploadFile):
    file_content = await file.read()
    X_new = pd.read_csv(BytesIO(file_content))
    X_scaled = scaler.transform(X_new)
    y_new = best_model.predict(X_scaled)
    if len(y_new) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one record to update")
    for row in history:
        if row["index_x"] == id:
            row["date"] = date.today()
            row["prediction"] = y_new[0]
            return row
    raise HTTPException(status_code=404, detail="Prediction not found")

@app.delete("/delete/{id}")
async def delete_prediction(id: int):
    for i, row in enumerate(history):
        if row["index_x"] == id:
            deleted = history.pop(i)
            return {"deleted": deleted}
    raise HTTPException(status_code=404, detail="Prediction not found")
