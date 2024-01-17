# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# Loads the heart attack possibility dataset named as heart.csv which must stored in the same directory
heart_df = pd.read_csv("heart.csv")

# Cleaning and preprocessing the data from the dataset
# Remove duplicate rows
heart_df = heart_df.drop_duplicates() 
# Remove rows with missing values
heart_df = heart_df.dropna() 


# Visualizes the distribution of target variable
sns.countplot(x='target', data=heart_df)
plt.title("Distribution of target variable")
plt.show()


#show relationship between heart rate and count
sns.histplot(heart_df.thalach,kde=True)

#shows relation between gender and heart attack in numbers
pd.crosstab(heart_df.sex, heart_df.target, margins= True)

#shows relation between gender and heart attack in percentage
all=pd.crosstab(heart_df.sex,heart_df.target,margins=True)['All']
pd.crosstab(heart_df.sex,heart_df.target,).divide(all,axis=0).dropna()

# Convert categorical variables to numerical values
heart_df = pd.get_dummies(heart_df, columns=['cp', 'thal', 'slope'])

# Scale the input features
scaler = StandardScaler()
X = heart_df.drop('target', axis=1)
X = scaler.fit_transform(X)
y = heart_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logreg.predict(X_test)

# Evaluate the performance of the model
confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# Print the evaluation metrics
print('Confusion Matrix:\n', confusion_mat)
print('Accuracy Score:', accuracy)
print('Precision Score:', precision)
print('Recall Score:', recall)
print('F1 Score:', f1)

