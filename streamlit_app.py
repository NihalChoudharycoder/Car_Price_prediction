import pickle
import numpy as np
import sklearn
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from PIL import Image

st.write("""
# Car-Price Prediction
  Predict the price of a used car provided details like Year, Showroom Price, Kms driven, fuel type, transmission type etc.
  To train the regressor model here we have used random forest regression and the data here is by Cardekho public dataset over kaggle.
 """)

model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

image1 = Image.open("car-price-prediction1.jpeg")
st.image(image1, caption='Cardekho', use_column_width=True)

df = pd.read_csv('car data.csv')
st.subheader('Data Information:')

# Display of data as table
st.dataframe(df)

# Display the statistics of the data
st.write(df.describe())
# Display the bar_chart of the data
chart = st.bar_chart(df)

options = pd.DataFrame({
  'Fuel_Type': ['Petrol','Diesel'],
  'Seller_Type': ['Individual','Dealer'],
  'Transmission_Type': ['Mannual','Automatic']
})

#Get the feature input from the user
def get_user_input():
    Year = st.sidebar.slider('Year', 2003, 2018, 2011)
    Year = 2020-Year
    Present_Price = st.sidebar.slider('Present_Price', 1, 35, 5)
    Kms_Driven = st.sidebar.slider('Kms_Driven', 500, 500000, 100000)
    Owner = st.sidebar.slider('Owner', 0, 3, 1)
    Fuel_Type = st.sidebar.selectbox('Fuel_Type', options['Fuel_Type'])
    if(Fuel_Type == 'Petrol'):
        Fuel_Type_Petrol = 1
        Fuel_Type_Diesel = 0
    else:
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 1
    Seller_Type = st.sidebar.selectbox('Seller_Type', options['Seller_Type'])
    if(Seller_Type == 'Individual'):
        Seller_Type = 1
    else:
        Seller_Type = 0

    Transmission_Type = st.sidebar.selectbox('Transmission_Type', options['Transmission_Type'])
    if(Transmission_Type == 'Mannual'):
        Transmission_Type = 1
    else:
        Transmission_Type = 0
    # Store a dictionanry into a variables
    user_data = {'Year' : Year,
                 'Present_Price': Present_Price,
                 'Kms_Driven': Kms_Driven,
                 'Owner': Owner,
                 'Fuel_Type_Petrol' : Fuel_Type_Petrol,
                 'Fuel_Type_Diesel' : Fuel_Type_Diesel,
                 'Seller_Type': Seller_Type,
                 'Transmission_Type':Transmission_Type
                 }
    #Transform the data into data dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variables
user_input = get_user_input()

# Set a subheader and display the users user_input
st.subheader('User Input:')
st.write(user_input)

prediction = model.predict(user_input)
output = round(prediction[0], 2)


if output<0:
    st.text('Sorry you cannot sell this car')

# Set a subheader and display
else:
    st.subheader('Re-Sale Price of the Car:')
    st.write(output)
