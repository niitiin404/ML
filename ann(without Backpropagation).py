from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.4, random_state=42)

model = MLPClassifier(max_iter=1000)

model.fit(X_train, y_train)

for i, loss in enumerate(model.loss_curve_):
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

accuracy = model.score(X_test, y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
