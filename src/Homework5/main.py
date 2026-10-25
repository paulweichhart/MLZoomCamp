import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
	DictVectorizer(),
	LogisticRegression(solver='liblinear')
)

datapoint = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

with open('src/Homework5/pipeline.bin', 'rb') as f_in:
	pipeline = pickle.load(f_in)
	
	result = round(pipeline.predict_proba(datapoint)[0, 1],3)
	print(result)
	
	
