# Machine_Learning_algo
implemented various ML algorithms 
I will show you how to perform Logistic Regression in Python. This would be very easy. An you will have all the codes.

These are the steps:

 

Step 1: Import the required modules

We  would import the following modules:

make_classification: available in sklearn.datasets and used to generate dataset

matplotlib.pyplot: for plotting

LogisticRegression: this is imported from sklearn.linear_model. Used for performing logistic regression

train_test_split:  imported from sklearn.model_selection and used to split dataset into training and test datasets

confusion matrix: imported from sklearn.metrics and used to generate the confusion matrix of the classifiers

Pandas for managing datasets.

The complete import statement is given below:

from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
 

Step 2: Generate the dataset

Now you need to generate the dataset using the make_classification() function. You need to specify the number of samples, the number of feature, number of classes and other parameters.

The code for the make_classification is given below:

# Generate and dataset for Logistic Regression
x, y = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)
 

Step 3: Visualize the Data

Now we would create a simple scatter plot just to see how the data looks like. The code and the output is given below:
# Create a scatter plot
plt.scatter(x, y, c=y, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()
Logistic Regression Plot

 

Step 4: Split the Dataset

Now we would split the dataset into training dataset and test dataset. The training dataset is used to train the model while the test dataset is used to test the model’s performance on new data.

# Split the dataset into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
 

Step 5: Perform Logistic Regression

Here we would create a LogistiRegression object and fit it with out dataset. (kind of similar to Linear Regression)

# Create a Logistic Regression Object, perform Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
The logistic regression output is given below:

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
You can view the logistic regression coefficient and intercept using the code below:

# Show to Coeficient and Intercept
print(lr.coef_)
print(lr.intercept_)
 

Step 6: Make prediction using the model

We now use the model to predict the outputs given the test dataset.

# Perform prediction using the test dataset
y_pred = lr.predict(x_test)
(you can view the predicted values using print(y_pred)

 

Step 7: Display the Confusion Matrix

The confusion matrix helps you to see how the model performed. It tells you the number of True positives, true negatives, false positives and false negatives. To see the confusion matrix, use:

# Show the Confusion Matrix
confusion_matrix(y_test, y_pred)
The output is:

array([[13,  1],
       [ 0, 11]], dtype=int64)
 

We can deduce from the confusion matrix that:

# True positive: 13 (upper-left) – Number of positives we predicted correctly
# True negative: 11(lower-right) – Number of negatives we predicted correctly
# False positive: 1 (top-right) – Number of positives we predicted wrongly
# False negative:  0(lower-left) – Number of negatives we predicted wrongly

Thanks for reading.
