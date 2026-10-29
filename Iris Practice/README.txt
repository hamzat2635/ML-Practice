IRIS FLOWER CLASSIFICATION PRACTICE
===================================

1. PROJECT GOAL
---------------
Use four flower measurements to predict which Iris species a flower belongs to.
This is supervised classification because the dataset already contains the correct
species label for every flower.

2. DATASET
----------
File: Iris.csv
Rows: 150 flowers
Classes: 50 Iris-setosa, 50 Iris-versicolor, and 50 Iris-virginica

Input measurements (features):
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

Answer to predict (target): Species

Id identifies a row. It is not a flower measurement and is not used for prediction.
Sepal and petal are flower parts; they are not flower species.

3. WHAT WE DID
--------------
- Loaded and inspected the data with pandas.
- Counted the flowers in each species.
- Calculated average measurements for each species.
- Drew a petal length versus petal width scatter plot.
- Split the data into 60% learning, 20% validation, and 20% final testing.
- Compared KNN with k values 1, 3, 5, 7, and 9.
- Also compared logistic regression and a decision tree.
- Used validation accuracy to select the model.
- Retrained the selected model on 120 flowers and tested it on 30 unseen flowers.
- Added an option to classify a new flower from four measurements.

4. FINDINGS
-----------
- Iris-virginica had the largest average petals: 5.55 cm long and 2.03 cm wide.
- Iris-setosa had the smallest average petals: 1.46 cm long and 0.24 cm wide.
- KNN with k=5 was selected with 96.7% validation accuracy.
- The final test result was 28 correct predictions out of 30, or 93.3%.
- The two mistakes were Iris-virginica flowers classified as Iris-versicolor.
- Similar measurements can overlap, so measurements do not separate every flower perfectly.

5. QUESTIONS WE ASKED AND ANSWERED
---------------------------------
Q: Does Species mean sepal or petal?
A: No. Species is the flower type. Sepal and petal columns are measurements.

Q: Why does KNN look for nearby flowers?
A: It compares all four measurements and lets the nearest known flowers vote.

Q: Does it compare column 1 directly with column 3?
A: No. It compares the new flower with other rows using the same measurement columns.

Q: Who chooses the model?
A: We choose candidate models, then validation results help us select one fairly.

Q: Can k be 1, 4, 8, or 12?
A: Yes, but odd values often reduce tied votes in classification. The best k depends
   on validation results, not on whether the dataset has an even or odd row count.

Q: Is Iris supervised or unsupervised?
A: It is supervised here because Species supplies the known answers.

Q: Why keep test data separate?
A: Validation helps choose the model. The untouched test set gives a final, fair check.

6. WHAT I LEARNED
-----------------
- Features are the clues and the target is the answer.
- Classification predicts a category such as a flower species.
- KNN predicts from nearby examples and k is the number of neighbors voting.
- Scaling matters for distance-based models such as KNN.
- Training teaches the model, validation helps make choices, and testing checks the
  final choice on unseen examples.
- Accuracy means correct predictions divided by all predictions.
- A model can make mistakes where categories have similar measurements.

7. HOW TO RUN
-------------
From the ML-Practice repository:

  ../.venv/bin/python -u "Iris Practice/iris_practice.py" --no-plot

Run with the chart:

  ../.venv/bin/python -u "Iris Practice/iris_practice.py"

Predict a new flower (sepal length, sepal width, petal length, petal width):

  ../.venv/bin/python -u "Iris Practice/iris_practice.py" --no-plot --predict 5.1 3.5 1.4 0.2

8. MAIN FILES
-------------
- Iris.csv: dataset
- iris_practice.py: exploration, training, testing, and prediction
- pandas.ipynb: pandas practice notebook
- metplotlib.ipynb: plotting practice notebook

