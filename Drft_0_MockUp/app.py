# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # dev เท่านั้น (โปรดระวังตอนขึ้นโปรดักชัน)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




model = joblib.load("model.pkl")  # โมเดลที่เทรนไว้

class InputData(BaseModel):
    x1: float
    x2: float

@app.post("/predict")
def predict(data: InputData):
    y = model.predict([[data.x1, data.x2]])[0]
    return {"prediction": str(y)}
