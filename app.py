import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("☕ Coffee Price Prediction")

size = st.number_input("Size (ml)", 240, 600)
shots = st.number_input("Coffee shots", 1, 3)
milk = st.selectbox("Milk Type", ["Normal", "Oat/Soy"])
shop = st.selectbox("Shop Level", ["Local", "Chain", "Premium"])
iced = st.selectbox("Drink Type", ["Hot", "Iced"])

milk_val = 0 if milk == "Normal" else 1
shop_val = ["Local","Chain","Premium"].index(shop)
iced_val = 0 if iced == "Hot" else 1

if st.button("Predict Price"):
    data = np.array([[size, shots, milk_val, shop_val, iced_val]])
    prediction = model.predict(data)

    st.success(f"Estimated price: {prediction[0]:.2f} THB")