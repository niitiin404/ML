
import pandas as pd

def candandate_eliminate(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    special_h = None
    general_h = [["?"] * len(X[0])]

    for i, label in enumerate(y):
        if label == "Yes":
            if special_h is None:
                special_h = X[i].copy()
            else:
                special_h = [h if h == x else "?" for h, x in zip(special_h, X[i])]
                print(special_h)
        else:
            for j in range(len(X[i])):
                if special_h[j] != X[i][j] and special_h[j] != "?":
                    general_h[0][j] = special_h[j]

    return special_h, general_h

# Replace "training_data.csv" with the path to your CSV file
data = pd.read_csv("training_data.csv")

specific_hypothesis, general_hypotheses = candandate_eliminate(data)

print("Specific Hypothesis:")
print(specific_hypothesis)
print("\nGeneral Hypotheses:")
print(general_hypotheses)
