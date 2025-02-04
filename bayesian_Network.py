import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("heart (1).csv")

# Split features and target
x = data.drop(columns="target")
y = data["target"]

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

# Train the Naïve Bayes model
model = GaussianNB().fit(x_train, y_train)

# Get predictions
y_pred = model.predict(x_test)

# Compute accuracy
acc = model.score(x_test, y_test)

# Compute probability estimates
probs = model.predict_proba(x_test[:2])  # Taking first two samples for table representation

# Print formatted output
print("|  t   |  phi(t)  |")
print("===================")
for i, p in enumerate(probs):
    print(f"| t({i}) |  {p[1]:.4f} |")
    print("-------------------")
print(f"\nAccuracy: {acc:.2f}")
