import pandas as pd
import numpy as np

from RegressionUtils import *

BASE = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df = pd.read_csv('data/car_fuel_efficiency_2026.csv')

print("# Exploring Data")
RegressionUtils.explore_data(df, 'fuel_efficiency_mpg')

print(f"# 1. DataFrame: {df[BASE].isnull().any()}")

median = df['horsepower'].median()
print(f" # 2. MEDIAN: {median} Horsepower")

mean = df['horsepower'].mean()
print(f" # 2. MEAN: {mean} Horsepower")

df_train, df_val, df_test = RegressionUtils.split_data(df[BASE])

y_train, y_val, y_test = RegressionUtils.prepare_target(df_train, df_val, df_test, 'fuel_efficiency_mpg')

X_train, X_val, X_test = RegressionUtils.prepare_data(df_train, df_val, df_test, 'fuel_efficiency_mpg', 0)

w0, w = RegressionUtils.train_linear_regression(X_train, y_train)
y_pred = w0 + X_val.dot(w)
print(f"# 3. RMSE: {round(RegressionUtils.rmse(y_val, y_pred), 3)}")

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
	w0, w = RegressionUtils.train_linear_regression_r(X_train, y_train, r)
	y_pred = w0 + X_val.dot(w)
	print(f"# 4. R: {r} RMSE: {round(RegressionUtils.rmse(y_val, y_pred), 4)}")

def rmse_variable_seed(seed):
	df_train, df_val, df_test = RegressionUtils.split_data(df[BASE], seed)
	y_train, y_val, y_test = RegressionUtils.prepare_target(df_train, df_val, df_test, 'fuel_efficiency_mpg')
	X_train, X_val, X_test = RegressionUtils.prepare_data(df_train, df_val, df_test, 'fuel_efficiency_mpg', 0)
	
	w0, w = RegressionUtils.train_linear_regression(X_train, y_train)
	y_pred = w0 + X_val.dot(w)
	return RegressionUtils.rmse(y_val, y_pred)
	
result = list(map(rmse_variable_seed, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(f"# 5. STD: {round(np.std(result), 3)}")

df_train, df_val, df_test = RegressionUtils.split_data(df[BASE], 9)
y_train, y_val, y_test = RegressionUtils.prepare_target(df_train, df_val, df_test, 'fuel_efficiency_mpg')

X_train, X_val, X_test = RegressionUtils.prepare_data(df_train, df_val, df_test, 'fuel_efficiency_mpg', 0)

X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

w0, w = RegressionUtils.train_linear_regression_r(X_full, y_full, 0.001)
y_pred = w0 + X_test.dot(w)
print(f"# 6. RMSE: {round(RegressionUtils.rmse(y_test, y_pred), 3)}")
