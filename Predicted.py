import streamlit as st
import numpy as np
import pandas as pd
import joblib

model=pd.read_csv("Toyota.csv")


st.title("Toyota Car Price Predictor 🚗")

# Feature inputs
age = st.slider("Age of Car (Months)", 0, 100, 30)
km = st.number_input("Kilometers Driven", min_value=0, value=50000)
hp = st.number_input("Horse Power (HP)", min_value=40, value=70)
cc = st.number_input("Engine Capacity (cc)", min_value=800, value=1300)
weight = st.number_input("Car Weight (kg)", min_value=800, value=1000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
fuel_map = {"Diesel": 0, "CNG": 1, "Petrol": 2}
fuel_encoded = fuel_map[fuel_type]

automatic = st.selectbox("Is Automatic?", ["No", "Yes"])
automatic_encoded = 1 if automatic == "Yes" else 0

doors = st.selectbox("Number of Doors", [2, 3, 4, 5])
gears = st.selectbox("Number of Gears", [3, 4, 5, 6])

# Predict button
if st.button("Predict Price"):
    features = np.array([[age, km, hp, cc, weight, fuel_encoded, automatic_encoded, doors, gears]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Price of Toyota Car: €{prediction:,.2f}")
