import io
import numpy as np
import pandas as pd
import requests

print(f"1. Pandas Version: {pd.__version__}")

# Fetching the data while bypassing SSL verification
URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
response = requests.get(URL, verify=False)
response.raise_for_status()
csv_content = response.text

# DataFrame for given URL
df = pd.read_csv(io.StringIO(csv_content))

print("\n# 2. DataFrame:")
df.info()

print(f"\n3. Unique Fuel Types: {len(df['fuel_type'].unique())}")

print(f"\n4. How many columns have missing values: {df.isnull().any().sum()}")

print(f"\n5. Maximum fuel efficiency of cars from Asia: {df[df['origin'].str.lower() == 'asia']['fuel_efficiency_mpg'].max()}")

print(f"\n6. Median Horsepower: {df['horsepower'].median()}")
freq = df['horsepower'].mode()
print(f"   Most Frequent: {freq}")
df['horsepower'] = df['horsepower'].fillna(freq)
print(f"\n   New Median Horsepower: {df['horsepower'].median()}")

asian_cars = df[df['origin'].str.lower() == 'asia']
X = asian_cars[['vehicle_weight', 'model_year']].head(n=7).values
XTX = np.transpose(X).dot(X)
XTX_invers = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = (XTX_invers.dot(np.transpose(X))).dot(y)
print(f"\n7. Sum of weights: {w.sum()}")