from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('./src/notebooks/KNeighbors_best_model.joblib')

class CustomerData(BaseModel):
    customerid: str
    surname: str
    creditscore: int
    geography: str
    gender: str
    age: int
    tenure: int
    balance: float
    numofproducts: int
    hascrcard: int
    isactivemember: int
    estimatedsalary: float

class PredictionOutput(BaseModel):
    churn_prediction: int

@app.post("/predict_churn", response_model=PredictionOutput)
async def predict_churn(customer: CustomerData):
    try:
        churn_prediction = model.predict(customer)[0]        
        return PredictionOutput(churn_prediction=int(churn_prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
