import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')

# Streamlit App Title
st.title("Logistic Regression Titanic Survival")
st.subheader("Predict Your Survival Chance If You Were on the Titanic")
st.sidebar.header('User Input Parameters')

# Function to collect user input
def user_input_feature():
    PassengerAge = st.sidebar.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
    PassengerFare = st.sidebar.number_input("Fare", min_value=100.0, step=100.0)
    PassengerGender = st.sidebar.selectbox('Gender (Male=1, Female=0)', ("1", "0"))
    PassengerClass = st.sidebar.selectbox("Pclass (1=1st, 2=2nd, 3=3rd)", ("1", "2", "3"))
    PassengerSiblings = st.sidebar.selectbox("Siblings (SibSp)", ("0", "1", "2", "3", "4", "5", "6"))
    PassengerParents = st.sidebar.selectbox("Parents (Parch)", ("0", "1", "2", "3", "4", "5", "6", "7"))
    
    data = {
        "PassengerAge": PassengerAge,
        "PassengerFare": PassengerFare,
        "PassengerGender": int(PassengerGender),
        "PassengerClass": int(PassengerClass),
        "PassengerSiblings": int(PassengerSiblings),
        "PassengerParents": int(PassengerParents)
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
data = user_input_feature()
st.write("User Input Parameters")
st.write(data)

# Load training data
dftrain = pd.read_csv("Titanic_train.csv")

# Fill missing values
dftrain['Cabin'].fillna('Unknown', inplace=True)
dftrain['Age'].fillna(dftrain['Age'].median(), inplace=True)
dftrain['Fare'].fillna(dftrain['Fare'].median(), inplace=True)
dftrain = dftrain.dropna(subset=['Embarked'])

# Encode categorical columns
label = LabelEncoder()
dftrain['Sex'] = label.fit_transform(dftrain['Sex'])       # Male=1, Female=0
dftrain['Embarked'] = label.fit_transform(dftrain['Embarked'])  # C/Q/S to 0/1/2

# Scale numerical features
scaler = MinMaxScaler()
dftrain[['Age', 'Fare']] = scaler.fit_transform(dftrain[['Age', 'Fare']])

# Prepare features and target
x = dftrain[['Age', 'Fare', 'Sex', 'Pclass', 'SibSp', 'Parch']]
y = dftrain['Survived']

# Train model
model = LogisticRegression()
model.fit(x, y)

# Prepare user input for prediction
data.rename(columns={
    'PassengerAge': 'Age',
    'PassengerFare': 'Fare',
    'PassengerGender': 'Sex',
    'PassengerClass': 'Pclass',
    'PassengerSiblings': 'SibSp',
    'PassengerParents': 'Parch'
}, inplace=True)

# Scale Age and Fare together to match training scaler
data[['Age', 'Fare']] = scaler.transform(data[['Age', 'Fare']])

# Make prediction
y_pred_prob = model.predict_proba(data)

# Show prediction results
st.subheader('Prediction: Survived? Yes (1) / No (0)')
st.write(' Yes, you will survive the accident!' if y_pred_prob[0][1] > 0.5 else " No, you won't survive.")

st.subheader('Prediction Probability')
st.write(y_pred_prob)