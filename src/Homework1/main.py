import io
import requests
import pandas as pd

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

print(f"\n5. Maximum fuel efficiency of cars from Asia: {df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()}")

med = df['horsepower'].median()
print(f"\n6. Median Horsepower: {med}")
print(f"   Most Frequent: {df['horsepower'].mode()}")
df['horsepower'] = df['horsepower'].fillna(med)
print(f"\n   New Median Horsepower: {df['horsepower'].median()}")
