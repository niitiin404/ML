import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset from training.csv
df = pd.read_csv("training.csv")

# Encode categorical features using LabelEncoder
df = df.apply(LabelEncoder().fit_transform)

# Split features and target variable
X = df.drop(columns="PlayTennis")  # Assuming "PlayTennis" is the target column
y = df["PlayTennis"]

# Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes model
model = CategoricalNB().fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Compare actual vs predicted values
print("\nActual vs Predicted:")
print(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}))
