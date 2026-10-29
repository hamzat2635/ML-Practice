TITANIC SURVIVAL PREDICTION PRACTICE
====================================

1. PROJECT GOAL
---------------
Study passenger information and train a supervised classification model to predict
whether a Titanic passenger survived.

2. DATASET
----------
- train.csv: 891 passengers with the known Survived answer
- test.csv: 418 Kaggle passengers whose Survived answer is hidden
- Target: Survived (0 = did not survive, 1 = survived)

Useful clues include passenger class, sex, age, fare, embarkation port, family
information, cabin availability, and the title extracted from the passenger's name.

Embarked means the port where a passenger boarded:
- C = Cherbourg
- Q = Queenstown
- S = Southampton

3. CLEANING THE DATA
--------------------
The training data originally had:
- 177 missing Age values
- 687 missing Cabin values
- 2 missing Embarked values

What we did:
- Removed Cabin from the simple cleaned CSV because most values were missing.
- Filled missing Age with the training median: 28.0.
- Filled the missing test Fare with the training median: 14.4542.
- Filled missing Embarked with the most common port: S.
- Saved train_cleaned.csv and test_cleaned.csv without changing the original CSV files.

We did not pretend to know each missing answer. Median and mode are reasonable
estimates learned only from the training data. Inside model evaluation, imputation is
performed in the pipeline to avoid learning from validation rows.

4. WHAT WE DID
--------------
- Inspected and cleaned the data.
- Studied survival by sex, passenger class, and age group.
- Created extra features: Title, FamilySize, IsAlone, AgeWasMissing, CabinKnown,
  and FarePerPerson.
- Held out 20% (179 passengers) as a final test group.
- Used five-fold cross-validation on the other 80% (712 passengers).
- Compared KNN, logistic regression, decision tree, random forest, and XGBoost.
- Tuned decision tree, random forest, and XGBoost settings with GridSearchCV.
- Selected the model from cross-validation without looking at final test answers.
- Tested the selected model once on the held-out group.
- Trained on all 891 labeled passengers and created submission.csv for Kaggle.

5. FINDINGS
-----------
Data findings:
- Overall survival rate: 38.4%.
- Female survival rate: 74.2%; male survival rate: 18.9%.
- First-class survival: 63.0%; second class: 47.3%; third class: 24.2%.
- Children aged 0-12 with known ages had a 58.0% survival rate.

Model findings from the latest run:
- Initial five-fold averages ranged from 79.6% for KNN to 82.9% for XGBoost.
- After tuning: decision tree 82.7%, random forest 83.2%, XGBoost 83.4%.
- Tuned XGBoost was selected.
- Final test: 146 correct out of 179, or 81.6% accuracy.
- Of 69 actual survivors, 50 were correctly found and 19 were missed.
- The Kaggle file predicts 156 survivors among 418 passengers.

6. QUESTIONS WE ASKED AND ANSWERED
---------------------------------
Q: Should every row with an empty cell be deleted?
A: No. Deleting many rows loses useful passengers. We can drop a mostly empty feature
   or estimate a small number of missing values.

Q: How can we fill cells when we do not know the real value?
A: We use a clear estimate such as the training median or most common category and
   accept that it is an approximation.

Q: What is validation?
A: It is a practice exam used to compare choices while the final test stays untouched.

Q: What is cross-validation?
A: It repeats training and validation using different folds, then averages the scores.
   This gives a more dependable comparison than one lucky split.

Q: Why can XGBoost win validation but do worse on the final test?
A: The groups contain different people. Validation is an estimate, so its winner is
   not guaranteed to have the highest score on every unseen group.

Q: What is recall?
A: For survivors, recall is correctly predicted survivors divided by all actual
   survivors. Higher recall means the model missed fewer survivors.

7. WHAT I LEARNED
-----------------
- Cleaning choices must be based on the meaning and amount of missing data.
- Preprocessing belongs in a pipeline so every validation fold remains fair.
- Feature engineering turns existing information into clearer clues.
- Cross-validation compares models more reliably than one validation split.
- Grid search tests model settings using training folds, without using final test data.
- Accuracy describes all predictions, while precision and recall focus on one class.
- Good validation performance does not guarantee an identical final test score.
- Kaggle test.csv is different from our local held-out test: Kaggle hides its answers.

8. HOW TO RUN
-------------
From the ML-Practice repository:

  ../.venv/bin/python -u "Titanic Practice/titanic_practice.py"

The script prints cleaning, analysis, cross-validation, tuning, and final test results.
It also rewrites train_cleaned.csv, test_cleaned.csv, and submission.csv.

9. MAIN FILES
-------------
- train.csv and test.csv: original Kaggle datasets
- train_cleaned.csv and test_cleaned.csv: readable cleaned versions
- titanic_practice.py: complete cleaning, analysis, model comparison, and prediction
- submission.csv: predictions ready for Kaggle

