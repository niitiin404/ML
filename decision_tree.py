import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=0)

model = DecisionTreeClassifier().fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

sample = [[5.1, 3.5, 1.4, 0.2]]
predicted_class = model.predict(sample)[0]
print(f"Predicted class: {data.target_names[predicted_class]}")

plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True)
plt.show()

