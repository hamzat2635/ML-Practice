TELCO CUSTOMER CHURN PREDICTION PRACTICE
========================================

1. PROJECT GOAL
---------------
Understand which customers leave a telecommunications company and train a supervised
classification model to predict churn.

Churn means a customer stopped using the company. The target column is Churn:
- No = customer stayed
- Yes = customer left

2. DATASET
----------
File: WA_Fn-UseC_-Telco-Customer-Churn.csv
Rows: 7,043 customers
Columns: 21, including the target

There are 5,174 customers who stayed (73.5%) and 1,869 who churned (26.5%).
The data contains customer services, contract, tenure, billing, payment, and charges.
customerID identifies a customer and is not used as a prediction clue.

3. CLEANING AND PREPARATION
---------------------------
- Found 11 blank TotalCharges values.
- All 11 belonged to new customers with tenure 0, so TotalCharges was set to 0.
- Converted TotalCharges from text to a number.
- Found no duplicate rows and no missing values after the fix.
- Removed customerID, Churn, and the analysis-only tenure_group from model inputs.
- Converted Churn from No/Yes to 0/1.
- One-hot encoded text categories and scaled numeric features.

4. WHAT WE DID
--------------
- Inspected the first rows, dimensions, columns, and data types.
- Counted and calculated the percentage of customers who churned.
- Compared churn by contract type, tenure group, and monthly charges.
- Plotted contract churn, tenure churn, monthly charges, and threshold tradeoffs.
- Split 80% for training and 20% for final testing.
- Split the training portion again into model-fitting and validation groups.
- Trained logistic regression to estimate each customer's churn probability.
- Compared probability thresholds from 0.10 through 0.90.
- Selected the threshold with the best F2 score because F2 gives recall more weight.
- Refit on all training rows and compared the normal 0.50 threshold with the selected
  0.15 threshold on the untouched test group.

5. FINDINGS
-----------
Customer findings:
- Month-to-month churn rate: 42.7%.
- One-year contract churn rate: 11.3%.
- Two-year contract churn rate: 2.8%.
- Customers who stayed had 37.6 months average tenure; churned customers had 18.0.
- Churn fell from 47.4% for 0-12 months to 6.6% for 61-72 months.
- Churned customers had higher average monthly charges: 74.44 versus 61.27.

Model findings on 1,409 final test customers:
- Always predicting Stay would be 73.5% accurate.
- Default 0.50 threshold: 80.6% accuracy, 65.7% precision, 55.9% recall,
  and 57.6% F2. It found 209 of 374 churners and missed 165.
- Selected 0.15 threshold: 66.8% accuracy, 44.0% precision, 91.4% recall,
  and 75.2% F2. It found 342 of 374 churners and missed only 32.
- The lower threshold catches many more possible churners but creates more false alarms.
  Which threshold is better depends on the business cost of misses and false alarms.

6. QUESTIONS WE ASKED AND ANSWERED
---------------------------------
Q: Does the dataset only list how many customers churned?
A: No. Each row describes one customer. Churn is the known answer, while the other
   columns are possible clues.

Q: How many customers are there?
A: 7,043. list_clients.py can print every record. A generated clients.txt may be kept
   locally, but it is not uploaded because it repeats all customer records.

Q: Does longer tenure mean less churn?
A: In this dataset, yes. Churn consistently decreases across the tenure groups. This
   is an association in the data; by itself it does not prove tenure causes the change.

Q: Does the model hide test data by itself?
A: No. We explicitly split the dataset. The model only learns from data passed to fit().

Q: Why have training, validation, and testing?
A: Training teaches the model, validation selects choices such as the threshold, and
   the untouched test data checks the finished choice fairly.

Q: Does the model know validation answers?
A: It does not train on them. We compare its predictions with those answers afterward
   so we can measure and select settings.

Q: Is lower recall better?
A: No. Recall = correctly found churners / all actual churners. Higher recall means
   fewer actual churners were missed.

Q: Why can accuracy fall when recall improves?
A: A lower threshold flags more customers. It catches more churners but also produces
   more false alarms, so overall accuracy can fall.

7. WHAT I LEARNED
-----------------
- A dataset contains rows (examples), features (clues), and a target (answer).
- Churn prediction is supervised binary classification.
- Data types and missing values must be fixed before model training.
- Categorical text must be converted into numerical model inputs.
- Scaling puts numeric features onto comparable ranges.
- A probability threshold converts churn probability into a Yes/No prediction.
- Precision asks how many churn warnings were correct.
- Recall asks how many actual churners were found.
- F2 values recall more heavily and can suit a goal of missing fewer churners.
- Model selection should reflect the real business goal, not accuracy alone.

8. HOW TO RUN
-------------
From the ML-Practice repository:

  MPLBACKEND=Agg ../.venv/bin/python -u "Churn Prediction/churn_practice.py"

Remove MPLBACKEND=Agg if you want chart windows to open:

  ../.venv/bin/python -u "Churn Prediction/churn_practice.py"

Print every customer:

  ../.venv/bin/python -u "Churn Prediction/list_clients.py"

Save the full list again:

  ../.venv/bin/python -u "Churn Prediction/list_clients.py" > "Churn Prediction/clients.txt"

9. MAIN FILES
-------------
- WA_Fn-UseC_-Telco-Customer-Churn.csv: original dataset
- churn_practice.py: cleaning, analysis, model training, and evaluation
- list_clients.py: prints all customer records
- clients.txt: optional local-only full listing generated by list_clients.py
- threshold_tradeoff.png: precision and recall threshold chart
- requirements.txt: Python packages used by the projects

