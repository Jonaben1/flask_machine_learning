import pandas as pd
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('gender_classifier.csv')

X = df.drop(['GENDER'], axis=1)
y = df.GENDER

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=5)

model = RandomForestClassifier()

model.fit(X_train.values, y_train.values)

joblib.dump(model, 'rf_model.pkl')
