import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from Homework3.ClassificationUtils import *

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
	df[c] = df[c].fillna('NA') if c in CATEGORIAL else df[c].fillna(0.0)
	
df_train, df_val, df_test = ClassificationUtils.split_data(df)
y_train, y_val, y_test = ClassificationUtils.prepare_target(df_train, df_val, df_test, 'converted')
df_train, df_val, df_test = ClassificationUtils.prepare_data(df_train, df_val, df_test, 'converted')

print("\n # 1. ROC AUC feature importance: ")
for c in NUMERICAL:
	auc_val = round(roc_auc_score(y_train, df_train[c]), 3)
	if auc_val < 0.5:
		print(f"-{c} { auc_val }")
	else:
		print(f"{c} { auc_val }")
		
y_pred, acc = ClassificationUtils.calculate_accuracy(df_train, df_val, y_train, y_val, 1.0)
print(f" \n# 2 AUC of this model: { round(roc_auc_score(y_val, y_pred), 3) }")

scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
	actual_positive = (y_val == 1)
	actual_negative = (y_val == 0)
	
	predict_positive = (y_pred >= t)
	predict_negative = (y_pred < t)

	tp = (predict_positive & actual_positive).sum()
	tn = (predict_negative & actual_negative).sum()

	fp = (predict_positive & actual_negative).sum()
	fn = (predict_negative & actual_positive).sum()
	
	scores.append((t, tp, fp, fn, tn))

columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()
plt.savefig('src/Homework4/curve.png')

num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal_pred = np.linspace(0, 1, len(y_val))

print(f" \n# 3. IDEAL { accuracy_score(y_ideal, y_ideal_pred >= (1 - y_val.mean())) } at Threshold ({ 1 - y_val.mean()})")

f1 = np.zeros((2, 100))
	
for t in thresholds:
	actual_positive = (y_val == 1)
	actual_negative = (y_val == 0)
	
	predict_positive = (y_pred >= t)
	predict_negative = (y_pred < t)

	tp = (predict_positive & actual_positive).sum()
	tn = (predict_negative & actual_negative).sum()

	fp = (predict_positive & actual_negative).sum()
	fn = (predict_negative & actual_positive).sum()
	
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	result = 2 * (precision * recall) / (precision + recall)
	print(f" {t} { round(result, 5)} ")
	np.append(f1, [result, t])
	
