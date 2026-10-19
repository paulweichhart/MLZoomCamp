import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

class ClassificationUtils:
	
	@staticmethod
	def explore_data(df):
		print(df.dtypes)
		print(df.head())
		
		print(df.isnull().any())
		
	@staticmethod
	def split_data(df):
		df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
		df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
		
		df_train = df_train.reset_index(drop=True)
		df_val = df_val.reset_index(drop=True)
		df_test = df_test.reset_index(drop=True)
		
		return df_full_train, df_train, df_val, df_test
		
	@staticmethod
	def prepare_target(df_train, df_val, df_test, column):
		y_train = df_train[column].values
		y_val = df_val[column].values
		y_test = df_test[column].values
		return y_train, y_val, y_test
		
	@staticmethod
	def prepare_data(df_train, df_val, df_test, column):
		del df_train[column]
		del df_val[column]
		del df_test[column]
		
		return df_train, df_val, df_test
		
	@staticmethod
	def calculate_mi(df, category, target):
		return round(mutual_info_score(df[category], target), 2)
		
	@staticmethod
	def ohe_data(df, dv, fit):		
		train_dict = df.to_dict(orient='records')
		if fit:
			return dv.fit_transform(train_dict)
		else:
			return dv.transform(train_dict)
		
	@staticmethod
	def train_model(X_train, y_train, X_val, y_val, c):
		model = LogisticRegression(solver='liblinear',C=1.0, max_iter=1000)
		model.fit(X_train, y_train)
		
		y_pred = model.predict_proba(X_val)[:, 1]
		converted_decision = (y_pred >= 0.5)
		
		return y_pred, (y_val == converted_decision).mean()
		
	@staticmethod
	def calculate_accuracy(df_train, df_val, y_train, y_val, c):
		dv = DictVectorizer(sparse=False)
		
		X_train = ClassificationUtils.ohe_data(df_train, dv, True)
		X_val = ClassificationUtils.ohe_data(df_val, dv, False)
		
		return ClassificationUtils.train_model(X_train, y_train, X_val, y_val, c)
	
	@staticmethod	
	def train(df_train, y_train, C):
		dicts = df_train.to_dict(orient='records')
	
		dv = DictVectorizer(sparse=False)
		X_train = dv.fit_transform(dicts)
	
		model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
		model.fit(X_train, y_train)
	
		return dv, model
		
	@staticmethod
	def predict(df, dv, model):
		dicts = df.to_dict(orient='records')
	
		X = dv.transform(dicts)
		return model.predict_proba(X)[:, 1]