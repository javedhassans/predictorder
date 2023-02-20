import pandas as pd
import streamlit as st
import datetime
import os

from pycaret.regression import load_model, predict_model


st.title("""ASIN: XYZZZ
**This project application helps play with the ordered units and ASP.**
""")

def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['prediction_label'][0]

    
model = load_model('../notebooks/models/blender_top3')

# Your df that contains "Time" column
df = pd.DataFrame()

start_date = st.date_input('Enter start date', value=datetime.datetime(2019,7,6))

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.sidebar.header('User Input Parameters')
    price = st.sidebar.slider(label = 'ASP', min_value = 125.0,
                          max_value = 162.0 ,
                          value = 125.0,
                          step = 0.1)

start_datetime = (pd.to_datetime(start_date, format='%Y-%m-%d'))

df["Date"] = [start_datetime]
df["ASP"] = [price]



st.dataframe(df)

if st.button('Predict'):
    
    prediction = predict_quality(model, df)
    
    st.write(' Based on the date and ASP values, the predicted ordered units are '+ str(round(prediction)))

