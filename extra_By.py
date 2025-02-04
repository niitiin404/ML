import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart (1).csv")

x = data.drop(columns="target")
y = data["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

model = GaussianNB().fit(x_train, y_train)

y_pred = model.predict(x_test)

acc = model.score(x_test, y_test) * 100
print(f"Model Accuracy: {acc:.2f}%")


filtered_data = x_test.query("age > 50 and chol > 0 and sex == 0")

probs = model.predict_proba(filtered_data)

filtered_data["Predicted Probabilities"] = [p[1] * 100 for p in probs] 
filtered_data["Heart Diagnosis"] = ["Heart Disease Present" if p[1] > 0.5 else "Heart Disease Absent" for p in probs]

filtered_data
