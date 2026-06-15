import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\sedna\Downloads\projects\forcasting\Sample_Pharmaceutical_Drug_Sales.csv"

def run_sarima_forecast(drug_name: str, days: int):

    df = pd.read_csv(DATA_PATH)

    drug_df = df[df["Drug Name"] == drug_name]

    if drug_df.empty:
        return {"error": "Drug not found"}

    drug_df["Sale Date"] = pd.to_datetime(drug_df["Sale Date"])
    drug_df = drug_df.sort_values("Sale Date")

    daily_demand = (
        drug_df
        .groupby("Sale Date")["Units Sold"]
        .sum()
        .asfreq("D")
        .fillna(0)
    )

    if len(daily_demand) < 60:
        return {"error": "Not enough historical data"}

    model = SARIMAX(
        daily_demand,
        order=(1,1,1),
        seasonal_order=(1,1,1,7)
    )

    fit = model.fit()

    forecast_result = fit.get_forecast(steps=days)

    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    result = []

    for i in range(len(forecast)):
        result.append({
            "date": str(forecast.index[i].date()),
            "y": float(forecast.iloc[i]),              # ✅ main forecast value
            "yhat_lower": float(conf_int.iloc[i, 0]),  # lower bound
            "yhat_upper": float(conf_int.iloc[i, 1])   # upper bound
        })

    return {
        "drug": drug_name,
        "total_forecast": float(forecast.sum()),
        "forecast": result
    }