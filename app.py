import streamlit as st
import pandas as pd
import joblib

# 1. Load the brain
model = joblib.load('model.pkl')

st.title("🍽️ Restaurant Rating Predictor")
st.write("Enter the details below to see the predicted rating!")

# 2. Create the "Input Boxes" (The Icons)
votes = st.number_input("How many votes does it have?", min_value=0, value=100)
avg_cost = st.number_input("Average Cost for two?", min_value=0, value=500)
price_range = st.slider("Price Range (1-4)", 1, 4, 2)

# 3. The "Predict" Button
if st.button("Predict Rating"):
    # Arrange data exactly like our training features
    input_data = pd.DataFrame([[avg_cost, votes, price_range, 0, 0]], 
                              columns=['Average Cost for two', 'Votes', 'Price range', 'Has Table booking', 'City_Encoded'])
    
    prediction = model.predict(input_data)
    st.success(f"The predicted rating is: {prediction[0]:.1f} ⭐")
