import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('src/Homework2/car_fuel_efficiency_2026.csv')

print(len(df))
print(df.head())

plt.figure(figsize=(6, 4))

sns.histplot(df.fuel_efficiency_mpg, bins=40, color='black', alpha=1)
plt.xlabel('MPG')
plt.title('Fuel Efficieny')

# plt.show()

print(f"min: {df.fuel_efficiency_mpg.min()} max: {df.fuel_efficiency_mpg.max()}")

BASE = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']

def prepare_data(data):
	df_data = data[BASE]
	print(df_data.isnull().any())
	# df_data = df_data.fillna(0)
	return df_data.values
	
training_data = prepare_data(df)
print(training_data[:5])