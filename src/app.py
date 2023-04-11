#!/usr/bin/env python
# coding: utf-8

#from tkinter import CENTER

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import dill
import base64
from PIL import Image

# st.camera_input("Take a picture")

def user_input_features():
        climate_def = st.sidebar.number_input('climate_def')
        st.sidebar.info("""climate_def describes the Climate water deficit. It is derived using
        a one dimensional soil water balance model""")
        climate_vpd = st.sidebar.number_input('climate_vpd')
        st.sidebar.info("climate_vpd describes the Vapor pressure deficit")
        climate_vs = st.sidebar.number_input('climate_vs')
        st.sidebar.info("climate_vs describes the Wind speed at 10m")
        landcover_4 = st.sidebar.number_input('landcover_4')
        st.sidebar.info("""landcover_4 describes Deciduous Broadleaf Vegetation: dominated by deciduous 
        broadleaf trees and shrubs (>1m). Woody vegetation cover >10%.""")
        climate_pet = st.sidebar.number_input('climate_pet')
        st.sidebar.info("climate_pet describes the Reference evapotranspiration")
        climate_srad = st.sidebar.number_input('climate_srad')
        st.sidebar.info("climate_strad describes Downward surface shortwave radiation")
        elevation = st.sidebar.number_input('elevation')
        st.sidebar.info("elevation describes land elevation")
                                          
        data = {'climate_def': climate_def,
                'climate_vpd' : climate_vpd,
                'climate_vs': climate_vs,
                'landcover_4': landcover_4,
                'climate_pet': climate_pet,
                'climate_srad': climate_srad,
                'elevation': elevation
                }
                                          
        features = pd.DataFrame(data, index=[0])
        return features

def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download Prediction as CSV file</a>'
    return href
  

image = Image.open('utils/Hotspots.jpg')

st.write('Hotspots Prediction')
st.image(image, caption='Hotspot Fire Burning')

st.write("""
         Welcome to the Hotspots Prediction App. This app takes a CSV file as input
         and returns the predictions based on the columns. The model is built with Hotspots fire data
         and the target is burn area.
         """)
st.sidebar.header('User Input Features')
st.sidebar.markdown("Example CSV input file](to_train.csv)")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    _df = pd.read_csv(uploaded_file)
    scaled_df = StandardScaler().fit_transform(_df.copy())
    
else:
    scaled_df = user_input_features()
    
    
df = pd.DataFrame(scaled_df)
trained_df = df.copy()
    
# Displays the user input features
st.subheader('User Input features')
    
if uploaded_file is not None:
    st.dataframe(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.dataframe(df) 
    
 # Reads in saved Regression model
load_reg = dill.load(open('model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_reg.predict(trained_df)
burn = pd.DataFrame(prediction, columns = ['burn_area'])

#Shows new prediction
predicted_burn = pd.concat([trained_df, burn], axis=1)

st.subheader('Prediction')
st.dataframe(predicted_burn)

# Plot the correlation between the features and the target
if uploaded_file is not None:
    correlation = predicted_burn.corr()['burn_area'].sort_values()
    fig = px.bar(correlation)
    st.plotly_chart(fig)
else:
    st.write('Data not enough to create a plot')


st.markdown(get_table_download_link_csv(predicted_burn), unsafe_allow_html=True)
