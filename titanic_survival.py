# Import necessary packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
print(passengers)

# Update sex column to numerical
print(passengers['Sex'])
passengers['Sex'] = passengers['Sex'].apply(lambda x: 1 if x == 'female' else 0)
print(passengers['Sex'])
# Fill the nan values in the age column
mean_age = passengers['Age'].mean()
passengers['Age'].fillna(value=mean_age,inplace=True)
print(passengers['Age'].values)
# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
X_train,X_test,y_train,y_test = train_test_split(features,survival)
# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Create and train the model
lregr = LogisticRegression()
lregr.fit(X_train,y_train)

# Score the model on the train data
print(lregr.score(X_train,y_train))


# Score the model on the test data
print(lregr.score(X_test,y_test))


# Analyze the coefficients
print(lregr.coef_)

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([0.0,23.0,1.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack,Rose,You])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
print(lregr.predict(sample_passengers))
print(lregr.predict_proba(sample_passengers))
