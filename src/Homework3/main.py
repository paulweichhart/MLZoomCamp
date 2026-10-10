import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv('data/course_lead_scoring_2026.csv')

print(df.dtypes)
print(df.head())

print(df.isnull().any())
	
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

print(df.industry.mode())