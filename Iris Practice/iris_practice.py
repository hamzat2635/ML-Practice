"""Start exploring the Iris flower dataset."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description="Explore Iris data and train a species predictor.")
parser.add_argument("--no-plot", action="store_true", help="Show results in the terminal without opening a chart.")
parser.add_argument(
    "--predict",
    nargs=4,
    type=float,
    metavar=("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH"),
    help="Predict a flower from four measurements in centimetres, in this order.",
)
args = parser.parse_args()
if args.predict is not None and any(value <= 0 for value in args.predict):
    parser.error("All four flower measurements must be positive numbers.")

print("[1/4] Loading the Iris data...", flush=True)
data_path = Path(__file__).with_name("Iris.csv")
flowers = pd.read_csv(data_path)

print(f"Flowers: {len(flowers)}")
print("\nFirst five rows:")
print(flowers.head().to_string(index=False))

print("\nFlowers per species:")
print(flowers["Species"].value_counts())

measurements = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

print("\nAverage measurements by species (cm):")
print(flowers.groupby("Species")[measurements].mean().round(2))

# X contains the clues. y contains the answer we want to predict.
# Id is just a row label, so it is not included in X.
X = flowers[measurements]
y = flowers["Species"]

print("\n[2/4] Splitting flowers into learning, validation, and test groups...", flush=True)
# Keep 30 flowers aside for the final test (20% of 150).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Split the remaining 120 into 90 for learning and 30 for validation.
# That makes the temporary split 60% / 20% / 20% of all flowers.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

print(f"\nFlowers used for learning: {len(X_fit)}")
print(f"Flowers used for validation: {len(X_val)}")
print(f"Flowers kept for the final test: {len(X_test)}")

# Compare several model choices on the same validation flowers.
# KNN and logistic regression use scaled measurements. A tree does not need scaling.
print("\n[3/4] Comparing models on the validation group...", flush=True)
candidates = [
    (f"KNN (k={k})", make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=k)
    ))
    for k in [1, 3, 5, 7, 9]
]
candidates.extend([
    ("Logistic regression", make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    )),
    ("Decision tree (depth=3)", DecisionTreeClassifier(
        max_depth=3, random_state=42
    )),
])

validation_scores = []
print("\nValidation results:")
for name, candidate in candidates:
    candidate.fit(X_fit, y_fit)
    val_predictions = candidate.predict(X_val)
    score = accuracy_score(y_val, val_predictions)
    validation_scores.append((score, name, candidate))
    print(f"{name}: {score:.1%} correct")

# If scores tie, keep the first model in the list.
best_score, best_name, model = max(validation_scores, key=lambda result: result[0])
print(f"\nChosen model: {best_name} (validation accuracy: {best_score:.1%})")

# With the model chosen, learn again from all 120 training + validation flowers.
# The test set is still separate from fitting and model selection.
print("\n[4/4] Training on 120 flowers and checking the final 30...", flush=True)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
correct = (predictions == y_test).sum()

print(f"\nFinal training flowers: {len(X_train)}")
print(f"Final test flowers: {len(X_test)}")
print(f"Correct test predictions: {correct} out of {len(X_test)}")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.1%}")

print("\nFirst five test predictions:")
print(pd.DataFrame({
    "Actual": y_test.to_numpy()[:5],
    "Predicted": predictions[:5],
}).to_string(index=False))

test_results = X_test.copy()
test_results.insert(0, "Id", flowers.loc[X_test.index, "Id"])
test_results["Actual"] = y_test
test_results["Predicted"] = predictions
mistakes = test_results[test_results["Actual"] != test_results["Predicted"]]

print("\nFlowers the model got wrong:")
print(mistakes.to_string(index=False))

if args.predict is not None:
    new_flower = pd.DataFrame([args.predict], columns=measurements)
    predicted_species = model.predict(new_flower)[0]
    print("\nYour flower measurements (cm):")
    print(new_flower.to_string(index=False))
    print(f"Predicted species: {predicted_species}")

print("\nDone.", flush=True)

if not args.no_plot:
    import matplotlib.pyplot as plt

    for species, group in flowers.groupby("Species"):
        plt.scatter(
            group["PetalLengthCm"],
            group["PetalWidthCm"],
            label=species,
        )

    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title("Iris flowers by petal measurements")
    plt.legend()
    plt.tight_layout()
    plt.show()
