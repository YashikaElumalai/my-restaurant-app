import streamlit as st
import pandas as pd
import joblib

# 1. Load the brain
model = joblib.load('model.pkl')

st.title("🍽️ Restaurant Rating Predictor")

# 2. User Inputs
votes = st.number_input("Total Votes", min_value=0, value=100)
avg_cost = st.number_input("Average Cost for Two", min_value=0, value=500)
price_range = st.slider("Price Range (1-4)", 1, 4, 2)
table = st.selectbox("Has Table Booking?", ["Yes", "No"])
delivery = st.selectbox("Has Online Delivery?", ["Yes", "No"])

# Convert Yes/No to 1/0
table_val = 1 if table == "Yes" else 0
delivery_val = 1 if delivery == "Yes" else 0

# 3. Predict Button
if st.button("Predict Rating"):
    # IMPORTANT: The columns MUST match the training features exactly!
    input_data = pd.DataFrame([[avg_cost, votes, price_range, table_val, delivery_val]], 
                              columns=['Average Cost for two', 'Votes', 'Price range', 'Has Table booking', 'Has Online delivery'])
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Rating: {prediction[0]:.1f} ⭐")
