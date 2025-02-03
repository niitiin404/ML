import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("training_data.csv")

# Encode categorical features using LabelEncoder
df = df.apply(LabelEncoder().fit_transform)

X = df.drop(columns="PlayTennis")  # Assuming "PlayTennis" is the target column
y = df["PlayTennis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = CategoricalNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Compare actual vs predicted values
print("\nActual vs Predicted:")
print(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}))
