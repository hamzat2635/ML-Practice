ML PRACTICE PROJECTS
====================

This repository records my machine learning practice, questions, findings, and lessons.
Each practice folder has its own README.txt with beginner-friendly explanations and
commands.

PROJECTS
--------
1. Iris Practice
   Predict one of three Iris flower species from four flower measurements.
   Main topics: supervised classification, KNN, model comparison, validation, testing.

2. Titanic Practice
   Predict whether a Titanic passenger survived.
   Main topics: missing data, feature engineering, pipelines, cross-validation,
   model tuning, random forest, XGBoost, precision, and recall.

3. Churn Prediction
   Predict whether a telecommunications customer will leave.
   Main topics: exploratory analysis, preprocessing, logistic regression,
   probability thresholds, precision, recall, and F2 score.

4. Uni Assignment
   University assignment files and shared dependency list.

SETUP
-----
The current local virtual environment is one directory above this repository.
From the repository folder, install the packages with:

  ../.venv/bin/python -m pip install -r "Uni Assignment/requirements.txt"

Read the README.txt inside a project before running it.

GENERAL MACHINE LEARNING WORKFLOW
---------------------------------
1. Understand the question and target.
2. Inspect and clean the data.
3. Explore patterns without claiming that correlation proves causation.
4. Select useful features and preprocess them.
5. Keep final test data separate.
6. Train and compare models using validation or cross-validation.
7. Tune choices using training data only.
8. Evaluate the finished model on the untouched test group.
9. Explain results with metrics that match the real goal.

