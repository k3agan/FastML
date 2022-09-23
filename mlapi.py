from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle


app = FastAPI()

with open('rfmodel.pkl', 'rb') as f:
    model = pickle.load(f)


class ScoringItem(BaseModel):
    YearsAtCompany: float
    EmployeeSatisfaction: float
    Position: str
    Salary: int


@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat)}
