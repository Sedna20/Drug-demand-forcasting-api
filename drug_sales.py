import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv(
    r"C:\Users\sedna\Downloads\projects\forcasting\Sample_Pharmaceutical_Drug_Sales.csv"
)

print(df.head())

drug_name = input("Enter Drug Name: ")
drug_df = df[
    df['Drug Name'] == drug_name
]
if len(drug_df) == 0:
    print("Drug not found in dataset")
    exit()

drug_df['Sale Date'] = pd.to_datetime(
    drug_df['Sale Date']
)
drug_df = drug_df.sort_values(
    'Sale Date'
)


daily_demand = (
    drug_df
    .groupby('Sale Date')['Units Sold']
    .sum()
)
daily_demand = daily_demand.asfreq('D')
daily_demand = daily_demand.fillna(0)
print("\nDaily Demand:")
print(daily_demand.head())
plt.figure(figsize=(12,6))
plt.plot(daily_demand)
plt.title(
    f"Daily Demand - {drug_name}"
)
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.show()


if len(daily_demand) < 60:
    print(
        "Not enough historical data"
    )
    exit()

train = daily_demand[:-30]
test = daily_demand[-30:]

print("\nTraining SARIMA...")



sarima_model = SARIMAX(
    train,
    order=(1,1,1),
    seasonal_order=(1,1,1,7)
)
sarima_fit = sarima_model.fit()
sarima_pred = sarima_fit.forecast(
    steps=len(test)
)

sarima_mae = mean_absolute_error(
    test,
    sarima_pred
)
sarima_rmse = np.sqrt(
    mean_squared_error(
        test,
        sarima_pred
    )
)
print("\nSARIMA Results")
print("MAE :", round(sarima_mae,2))
print("RMSE :", round(sarima_rmse,2))

print("Creating final model...")

final_model = SARIMAX(
    daily_demand,
    order=(1,1,1),
    seasonal_order=(1,1,1,7)
)

print("Fitting final model...")
final_fit = final_model.fit()
print("Final model fitted successfully")

forecast = final_fit.forecast(
    steps=30
)
print("Forecast generated")
print("\nNext 30 Days Forecast")
print(forecast)
pcm_required= forecast.sum()
print("PCM REQUIRED NEXT MONTH")
print(round(pcm_required))

avg_daily_demand = daily_demand.mean()
avg_lead_time = 30

reorder_level = (
    avg_daily_demand *
    avg_lead_time
)
print("\nReorder Level:")
print(round(reorder_level))


plt.figure(figsize=(12,6))
plt.plot(
    forecast.index,
    forecast,
    label='Forecast Next Month'
)
plt.legend()
plt.title(
    f"{drug_name} Forecast Next Month"
)
plt.xlabel("Date")
plt.ylabel("Forecast Units")
plt.show()