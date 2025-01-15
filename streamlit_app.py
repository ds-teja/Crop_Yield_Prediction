import streamlit as st
import requests

# URL to your FastAPI app
API_URL = "http://localhost:8000"  # Change this to the hosted FastAPI app URL after deployment

def predict_data(rainfall, year, irrigation_area, crop_type, soil_type):
    data = {
        "rainfall": rainfall,
        "year": year,
        "irrigation_area": irrigation_area,
        "crop_type": crop_type,
        "soil_type": soil_type
    }
    response = requests.post(f"{API_URL}/predictdata", json=data)
    return response.json()

# Streamlit interface
st.title("Crop Yield Prediction")
st.sidebar.header("Prediction Inputs")

rainfall = st.sidebar.number_input("Rainfall", min_value=0.0)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2050)
irrigation_area = st.sidebar.number_input("Irrigation Area", min_value=0.0)
crop_type = st.sidebar.selectbox("Crop Type", ["Wheat", "Rice", "Corn", "Barley"])
soil_type = st.sidebar.selectbox("Soil Type", ["Clay", "Loam", "Silt", "Sand"])

if st.sidebar.button("Predict"):
    prediction = predict_data(rainfall, year, irrigation_area, crop_type, soil_type)
    st.write("Prediction: ", prediction)
