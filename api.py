from fastapi import FastAPI
from pydantic import BaseModel, Field
from new import run_sarima_forecast

app = FastAPI()


class ForecastRequest(BaseModel):
    drug_name: str
    days: int


@app.post("/forecast")
def forecast(req: ForecastRequest):

    return run_sarima_forecast(req.drug_name, req.days)