"""Clean the Titanic datasets while keeping the original files unchanged."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


folder = Path(__file__).parent
train = pd.read_csv(folder / "train.csv")
test = pd.read_csv(folder / "test.csv")

print("Missing values before cleaning (training data):")
print(train.isna().sum()[train.isna().sum() > 0])

# Cabin is missing for most passengers, so we leave it out of this first model.
train_clean = train.drop(columns=["Cabin"]).copy()
test_clean = test.drop(columns=["Cabin"]).copy()

# Learn replacement values from the training dataset.
age_median = train["Age"].median()
fare_median = train["Fare"].median()
embarked_mode = train["Embarked"].mode()[0]

# Fill missing values instead of deleting passengers.
train_clean["Age"] = train_clean["Age"].fillna(age_median)
test_clean["Age"] = test_clean["Age"].fillna(age_median)
test_clean["Fare"] = test_clean["Fare"].fillna(fare_median)
train_clean["Embarked"] = train_clean["Embarked"].fillna(embarked_mode)
test_clean["Embarked"] = test_clean["Embarked"].fillna(embarked_mode)

train_output = folder / "train_cleaned.csv"
test_output = folder / "test_cleaned.csv"
train_clean.to_csv(train_output, index=False)
test_clean.to_csv(test_output, index=False)

print("\nReplacement values learned from training data:")
print(f"Age median: {age_median}")
print(f"Fare median: {fare_median}")
print(f"Most common embarkation port: {embarked_mode}")

print("\nMissing values after cleaning (training data):")
print(train_clean.isna().sum()[train_clean.isna().sum() > 0])

print("\nMissing values after cleaning (test data):")
print(test_clean.isna().sum()[test_clean.isna().sum() > 0])

print(f"\nSaved {len(train_clean)} training passengers to {train_output.name}")
print(f"Saved {len(test_clean)} test passengers to {test_output.name}")


# Explore survival patterns before training a model.
overall_survival = train_clean["Survived"].mean()
print(f"\nOverall survival rate: {overall_survival:.1%}")

survival_by_sex = train_clean.groupby("Sex")["Survived"].agg(
    Passengers="count",
    Survived="sum",
    SurvivalRate="mean",
)
survival_by_sex["SurvivalRate"] = survival_by_sex["SurvivalRate"].map(
    lambda value: f"{value:.1%}"
)
print("\nSurvival by sex:")
print(survival_by_sex)

survival_by_class = train_clean.groupby("Pclass")["Survived"].agg(
    Passengers="count",
    Survived="sum",
    SurvivalRate="mean",
)
survival_by_class["SurvivalRate"] = survival_by_class["SurvivalRate"].map(
    lambda value: f"{value:.1%}"
)
print("\nSurvival by passenger class (1 = highest class):")
print(survival_by_class)

# Use only passengers whose real age is known for this age comparison.
known_ages = train.dropna(subset=["Age"]).copy()
known_ages["AgeGroup"] = pd.cut(
    known_ages["Age"],
    bins=[0, 12, 17, 29, 49, float("inf")],
    labels=["Child (0-12)", "Teen (13-17)", "Young adult (18-29)",
            "Adult (30-49)", "Older adult (50+)"]
)
survival_by_age = known_ages.groupby(
    "AgeGroup", observed=True
)["Survived"].agg(
    Passengers="count",
    Survived="sum",
    SurvivalRate="mean",
)
survival_by_age["SurvivalRate"] = survival_by_age["SurvivalRate"].map(
    lambda value: f"{value:.1%}"
)
print("\nSurvival by age group (known ages only):")
print(survival_by_age)


# Build extra clues from the original passenger information.
def add_model_features(passengers):
    """Create useful passenger clues without changing the original data."""
    featured = passengers.copy()
    featured["Title"] = featured["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
    featured["Title"] = featured["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    })
    common_titles = ["Mr", "Mrs", "Miss", "Master"]
    featured.loc[~featured["Title"].isin(common_titles), "Title"] = "Rare"

    featured["FamilySize"] = featured["SibSp"] + featured["Parch"] + 1
    featured["IsAlone"] = (featured["FamilySize"] == 1).astype(int)
    featured["AgeWasMissing"] = featured["Age"].isna().astype(int)
    featured["CabinKnown"] = featured["Cabin"].notna().astype(int)
    featured["FarePerPerson"] = featured["Fare"] / featured["FamilySize"]
    return featured


train_for_model = add_model_features(train)

# Build a model that predicts Survived from passenger information.
numeric_features = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone",
    "AgeWasMissing", "CabinKnown", "FarePerPerson",
]
category_features = ["Pclass", "Sex", "Embarked", "Title"]
model_features = numeric_features + category_features

# Use the original selected columns here. Missing values are filled inside each
# cross-validation fold, so a fold never learns replacement values from its
# own validation passengers.
X = train_for_model[model_features]
y = train_for_model["Survived"]

print("\nEngineered model features:")
print("Title, FamilySize, IsAlone, AgeWasMissing, CabinKnown, FarePerPerson")

# Keep 20% untouched for the final test.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nDataset split:")
print(f"Cross-validation training passengers: {len(X_train)}")
print(f"Final test passengers: {len(X_test)}")


def make_preprocessor():
    """Fill missing values, convert categories, and scale numbers."""
    number_steps = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    category_steps = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer([
        ("numbers", number_steps, numeric_features),
        ("categories", category_steps, category_features),
    ])


candidates = [
    ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
    ("Logistic regression", LogisticRegression(max_iter=1000)),
    ("Decision tree (depth=4)", DecisionTreeClassifier(
        max_depth=4,
        random_state=42,
    )),
    ("Random forest", RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
    ("XGBoost", XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )),
]

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cross_validation_results = []
print("\nFive-fold cross-validation comparison:")

for model_name, algorithm in candidates:
    candidate = make_pipeline(make_preprocessor(), algorithm)
    fold_scores = cross_val_score(
        candidate,
        X_train,
        y_train,
        cv=cross_validation,
        scoring="accuracy",
        n_jobs=1,
    )
    average_score = fold_scores.mean()
    cross_validation_results.append((average_score, model_name, algorithm))
    scores_text = ", ".join(f"{score:.1%}" for score in fold_scores)
    print(f"{model_name}: {scores_text} | Average: {average_score:.1%}")

# Tune settings for the three strongest model families without using test data.
tuning_candidates = [
    (
        "Decision tree",
        DecisionTreeClassifier(random_state=42),
        {
            "decisiontreeclassifier__max_depth": [3, 4, 5, 6, None],
            "decisiontreeclassifier__min_samples_leaf": [1, 2, 5, 10],
        },
    ),
    (
        "Random forest",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        {
            "randomforestclassifier__max_depth": [4, 5, 7, None],
            "randomforestclassifier__min_samples_leaf": [1, 2, 5],
        },
    ),
    (
        "XGBoost",
        XGBClassifier(
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
        {
            "xgbclassifier__n_estimators": [100, 200],
            "xgbclassifier__max_depth": [2, 3, 4],
            "xgbclassifier__learning_rate": [0.03, 0.05, 0.1],
        },
    ),
]

print("\nTuning the top three model families with cross-validation:")
tuned_results = []

for model_name, algorithm, parameter_grid in tuning_candidates:
    search = GridSearchCV(
        estimator=make_pipeline(make_preprocessor(), algorithm),
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    tuned_results.append((search.best_score_, model_name, search.best_estimator_))
    readable_settings = {
        name.split("__")[-1]: value
        for name, value in search.best_params_.items()
    }
    print(
        f"{model_name}: {search.best_score_:.1%} average, "
        f"best settings = {readable_settings}"
    )

# Choose by cross-validation average. The final test answers are not used here.
best_score, best_model_name, final_model = max(
    tuned_results,
    key=lambda result: result[0],
)
print(f"\nChosen tuned model: {best_model_name}")
print(f"Average cross-validation accuracy: {best_score:.1%}")

# GridSearchCV has already refitted this model on the full 80% training set.
test_predictions = final_model.predict(X_test)

print("\nFinal test results:")
print(f"Correct predictions: {(test_predictions == y_test).sum()} of {len(y_test)}")
print(f"Accuracy: {accuracy_score(y_test, test_predictions):.1%}")
print("\nConfusion matrix:")
print("Rows are actual [did not survive, survived].")
print("Columns are predicted [did not survive, survived].")
print(confusion_matrix(y_test, test_predictions))
print("\nDetailed results:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["Did not survive", "Survived"],
))


# Train the selected model on all 891 known passengers, then predict the
# 418 Kaggle passengers whose survival answers are hidden.
test_for_model = add_model_features(test)
X_kaggle_test = test_for_model[model_features]

final_model.fit(X, y)
kaggle_predictions = final_model.predict(X_kaggle_test).astype(int)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": kaggle_predictions,
})
submission_output = folder / "submission.csv"
submission.to_csv(submission_output, index=False)

print("\nKaggle prediction file:")
print(f"Predicted survivors: {int(kaggle_predictions.sum())} of {len(submission)}")
print(f"Saved: {submission_output.name}")
print("First five predictions:")
print(submission.head().to_string(index=False))
