import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


CHAMP_MODEL_PATH = '../lab_4/Model_champion/prod_champion.joblib'

app = FastAPI()


class ObesityInput(BaseModel):
    Age: int
    Height: float
    Weight: float
    family_history_with_overweight: int
    FAVC: int
    FCVC: int
    NCP: float
    CAEC: int
    SMOKE: float
    CH2O: float
    SCC: float
    FAF: float
    TUE: float
    CALC: int
    Gender_Female: int
    Gender_Male: int
    MTRANS_Automobile: int
    MTRANS_Bike: int
    MTRANS_Motorbike: int
    MTRANS_Public_Transportation: int
    MTRANS_Walking: int

OBESITY_LEVEL_BY_ID = {0: "Insufficient weight", 1: "Normal weight", 2: "Overweight level I", 3: "Overweight level II",
                        4: "Obesity type I", 5: "Obesity type II", 6: "Obesity type III"}

@app.get("/")
async def root():
    return "PJA-ASI-14C-GR2 API"


@app.post("/predict")
def predict(input: ObesityInput):
    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, CHAMP_MODEL_PATH)
    model = joblib.load(model_path)
    df = pd.DataFrame(input.__dict__, index=[0]).astype(np.float32)
    y = model.predict(df)
    return {'prediction': OBESITY_LEVEL_BY_ID[y.tolist()[0]]}

if __name__ == '__main__':
    autogluon_path = os.path.abspath("src/api/AutogluonModels")
    print(autogluon_path)