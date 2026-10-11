import pandas as pd
import numpy as np

from ClassificationUtils import *

df = pd.read_csv('data/course_lead_scoring_2026.csv')

ClassificationUtils.explore_data(df)

CATEGORIAL = [
	'lead_source',
	'industry',
	'employment_status',
	'location'	
]

NUMERICAL = [
	'annual_income',
	'number_of_courses_viewed',
	'interaction_count',
	'lead_score']

for c in list(df.dtypes.index):
	if c in CATEGORIAL:
		df[c] = df[c].fillna('NA')
	else:
		df[c] = df[c].fillna(0.0)

print(f"\n# 1. Most Frequent industry observation {df.industry.mode()}")

print("\n# 2. Correlation of Features")
print(df['interaction_count'].corr(df['lead_score']))
print(df['number_of_courses_viewed'].corr(df['lead_score']))
print(df['number_of_courses_viewed'].corr(df['interaction_count']))
print(df['annual_income'].corr(df['interaction_count']))

df_train, df_val, df_test = ClassificationUtils.split_data(df)
y_train, y_val, y_test = ClassificationUtils.prepare_target(df_train, df_val, df_test, 'converted')
df_train, df_val, df_test = ClassificationUtils.prepare_data(df_train, df_val, df_test, 'converted')

print("\n# 3. Mutual Information Score")
for c in list(df_train.dtypes[CATEGORIAL].index):
	print(f"between '{c}' and 'converted' { ClassificationUtils.calculate_mi(df_train, c, y_train) }")

accuracy = ClassificationUtils.calculate_accuracy(df_train, df_val, y_train, y_val, 1.0)

print(f"\n# 4. Accuracy of Model { accuracy }")

print(f"\n# 5. Accuracy of Model")
for c in list(df_train.dtypes.index):
	train_copy = df_train.copy()
	val_copy = df_val.copy()
	
	del train_copy[c]
	del val_copy[c]
	
	acc = ClassificationUtils.calculate_accuracy(train_copy, val_copy, y_train, y_val, 1.0)
	print(f"without '{c}' { acc } and diff { accuracy - acc }")

print("\n# 6. Accuracy with changing C Parameter")
for c in [0.000001, 0.00001, 0.0001, 0.001]:
	acc = ClassificationUtils.calculate_accuracy(df_train, df_val, y_train, y_val, c)
	print(f"Accuracy of Model with '{c}' { acc } ")
