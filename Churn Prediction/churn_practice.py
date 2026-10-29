# =========================================================
# TELCO CUSTOMER CHURN - MACHINE LEARNING PRACTICE PROJECT
# =========================================================


# =========================
# 1. Import Libraries
# =========================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    fbeta_score
)


# =========================
# 2. Load Dataset
# =========================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# =========================
# 3. Initial Dataset Inspection
# =========================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()


# =========================
# 4. Investigate TotalCharges
# =========================

print("\nEmpty TotalCharges:")

print(
    df[df["TotalCharges"].str.strip() == ""]
)

empty_total = df[
    df["TotalCharges"].str.strip() == ""
]

print(
    empty_total[
        [
            "customerID",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn"
        ]
    ]
)


# =========================
# 5. Convert TotalCharges to Numeric
# =========================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges before fix:")
print(df["TotalCharges"].isnull().sum())


# =========================
# 6. Fix Missing TotalCharges
# =========================

# Customers with tenure = 0 have not accumulated
# historical TotalCharges yet.

df.loc[
    (df["TotalCharges"].isnull()) &
    (df["tenure"] == 0),
    "TotalCharges"
] = 0

print(
    "\nMissing TotalCharges after fix:",
    df["TotalCharges"].isnull().sum()
)


# =========================
# 7. Analyze Target Variable: Churn
# =========================

print("\nChurn counts:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print(
    df["Churn"].value_counts(
        normalize=True
    ) * 100
)

print("\nUnique Churn values:")
print(df["Churn"].unique())


# =========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================


# =========================
# 8. Contract vs Churn
# =========================

print("\nContract types:")
print(df["Contract"].value_counts())

print("\nContract vs Churn:")
print(
    pd.crosstab(
        df["Contract"],
        df["Churn"]
    )
)

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nContract churn percentages:")
print(contract_churn)


# Contract Churn Visualization

contract_churn["Yes"].plot(
    kind="bar"
)

plt.title(
    "Churn Rate by Contract Type"
)

plt.xlabel(
    "Contract Type"
)

plt.ylabel(
    "Churn Percentage"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# =========================
# 9. Tenure vs Churn
# =========================

print("\nTenure statistics by Churn:")

print(
    df.groupby(
        "Churn"
    )["tenure"].agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
)


# =========================
# 10. Create Tenure Groups
# =========================

df["tenure_group"] = pd.cut(
    df["tenure"],

    bins=[
        -1,
        12,
        24,
        36,
        48,
        60,
        72
    ],

    labels=[
        "0-12",
        "13-24",
        "25-36",
        "37-48",
        "49-60",
        "61-72"
    ]
)


# =========================
# 11. Tenure Group vs Churn
# =========================

tenure_churn = pd.crosstab(
    df["tenure_group"],
    df["Churn"],
    normalize="index"
) * 100

print(
    "\nChurn percentage by tenure group:"
)

print(tenure_churn)


# Tenure Visualization

tenure_churn["Yes"].plot(
    kind="bar"
)

plt.title(
    "Churn Rate by Customer Tenure"
)

plt.xlabel(
    "Tenure (Months)"
)

plt.ylabel(
    "Churn Percentage"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# =========================
# 12. Monthly Charges vs Churn
# =========================

print(
    "\nMonthly Charges statistics by Churn:"
)

print(
    df.groupby(
        "Churn"
    )["MonthlyCharges"].agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
)


# Monthly Charges Boxplot

df.boxplot(
    column="MonthlyCharges",
    by="Churn"
)

plt.title(
    "Monthly Charges by Churn"
)

plt.suptitle("")

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Monthly Charges"
)

plt.tight_layout()

plt.show()


# =========================
# 13. Check Duplicate Rows
# =========================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# =========================
# 14. Check Missing Values
# =========================

print("\nMissing values by column:")
print(df.isnull().sum())


# =========================================================
# MACHINE LEARNING PREPARATION
# =========================================================


# =========================
# 15. Create Features (X)
#    and Target (y)
# =========================

X = df.drop(
    columns=[
        "Churn",
        "customerID",
        "tenure_group"
    ]
)


# Convert:
#
# No  -> 0
# Yes -> 1

y = df["Churn"].map(
    {
        "No": 0,
        "Yes": 1
    }
)


print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget values:")
print(y.value_counts())


# =========================
# 16. Train / Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:")
print(X_train.shape)

print("\nTraining target:")
print(y_train.value_counts())


print("\nTesting data:")
print(X_test.shape)

print("\nTesting target:")
print(y_test.value_counts())


# =========================
# 17. Find Categorical and
#     Numerical Columns
# =========================

categorical_columns = X.select_dtypes(
    include=[
        "object",
        "string"
    ]
).columns


numeric_columns = X.select_dtypes(
    include=[
        "number"
    ]
).columns


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumeric columns:")
print(numeric_columns)


# =========================
# 18. Create Preprocessor
# =========================

preprocessor = ColumnTransformer(
    transformers=[

        # Convert text categories
        # into numerical columns

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),

        # Scale numerical variables

        (
            "num",
            StandardScaler(),
            numeric_columns
        )
    ]
)


# =========================
# 19. Make a Validation Set
# =========================

# Keep the test set untouched while choosing a threshold.
# The validation set is 25% of the training set, or 20% of all rows.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

print("\nModel-fitting data:", X_fit.shape)
print("Validation data:", X_val.shape)


# =========================
# 20. Fit on Model-Fitting Data
# =========================

X_fit_ready = preprocessor.fit_transform(X_fit)
X_val_ready = preprocessor.transform(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(X_fit_ready, y_fit)

# Probability that each validation customer will churn.
val_churn_probability = model.predict_proba(X_val_ready)[:, 1]


# =========================
# 21. Compare Thresholds on Validation Data
# =========================

# F2 gives recall more weight than precision. This fits the goal
# of catching more customers who would leave.
threshold_results = []

for threshold in [i / 100 for i in range(10, 91, 5)]:
    val_pred = (val_churn_probability >= threshold).astype(int)
    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall": recall_score(y_val, val_pred, zero_division=0),
        "f2": fbeta_score(y_val, val_pred, beta=2, zero_division=0)
    })

threshold_table = pd.DataFrame(threshold_results)

print("\nValidation results by threshold:")
print(threshold_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

best_row = max(threshold_results, key=lambda row: row["f2"])
best_threshold = best_row["threshold"]

print(f"\nChosen threshold (highest validation F2): {best_threshold:.2f}")
print(f"Validation precision: {best_row['precision']:.3f}")
print(f"Validation recall: {best_row['recall']:.3f}")


# =========================
# 22. Plot the Tradeoff
# =========================

plt.figure()
plt.plot(threshold_table["threshold"], threshold_table["precision"],
         marker="o", label="Precision")
plt.plot(threshold_table["threshold"], threshold_table["recall"],
         marker="o", label="Recall")
plt.axvline(best_threshold, color="gray", linestyle="--",
            label=f"Chosen threshold: {best_threshold:.2f}")
plt.xlabel("Churn probability threshold")
plt.ylabel("Score on validation data")
plt.title("Precision and Recall by Threshold")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("threshold_tradeoff.png", dpi=150)
plt.show()


# =========================
# 23. Refit on All Training Data
# =========================

# The threshold is fixed. Now use all training rows to fit the
# preprocessor and model before checking the held-out test set.
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)
model.fit(X_train_ready, y_train)

test_churn_probability = model.predict_proba(X_test_ready)[:, 1]
y_pred_default = (test_churn_probability >= 0.50).astype(int)
y_pred_tuned = (test_churn_probability >= best_threshold).astype(int)


# =========================
# 24. Compare on the Test Set
# =========================

baseline_accuracy = y_test.value_counts(normalize=True).max()
print(f"\nAlways-predict-stay baseline accuracy: {baseline_accuracy:.3f}")

for name, predictions in [
    ("Default threshold (0.50)", y_pred_default),
    (f"Chosen threshold ({best_threshold:.2f})", y_pred_tuned)
]:
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Churn precision: {precision_score(y_test, predictions):.3f}")
    print(f"Churn recall: {recall_score(y_test, predictions):.3f}")
    print(f"Churn F2: {fbeta_score(y_test, predictions, beta=2):.3f}")
    print("Confusion matrix [[stay correct, false alarm],"
          " [missed churn, churn correct]]:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(
        y_test, predictions, target_names=["Stay", "Churn"]
    ))
