# from st_aggrid import AgGrid
# import cv2
#from  PIL import ImageChops
from  PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
import io 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as html
import tensorflow as tf
import time
import math
import data_processing
import prediction
st.markdown('''
<style>
.stApp [data-testid="stToolbar"]{
    display:none;
}
</style>
''', unsafe_allow_html=True)
ohe_loaded = joblib.load('3_Models/transform_ohe.pkl')
df = pd.read_csv('bbox_and_commons.csv')
districts = df['district'].tolist()
# Folder containing district CSV files
data_folder = "2_Data/WeatherData/"
time_steps = 365  # Sequence length
with st.sidebar:
    choose = option_menu("MENU", ["Home", "Make Prediction", "Retrain Model", "Learn", "Contact"],
                         icons=['house', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
# logo = Image.open(r'C:\Users\13525\Desktop\Insights_Bees_logo.png')
# profile = Image.open(r'C:\Users\13525\Desktop\medium_profile.png')
if choose == "Home":
    st.subheader("ML Based Crop Yield Prediction", divider=True)
    st.markdown("""
    <p>This interface allows users to input key agricultural data, such as:</p>
    <ul>
        <li><strong>District:</strong> Select the district where the crop is being cultivated.</li>
        <li><strong>Season:</strong> Choose the season of cultivation (e.g., rainy season, dry season).</li>
        <li><strong>Area (in square meters):</strong> Enter the area under cultivation.</li>
    </ul>
    <p>The model will automatically estimate the following parameters:</p>
    <ul>
        <li><strong>GWETPROF, GWETTOP, GWETROOT:</strong> Soil moisture-related parameters.</li>
        <li><strong>CLOUD_AMT:</strong> Cloud cover amount.</li>
        <li><strong>TS:</strong> Soil temperature.</li>
        <li><strong>PS:</strong> Air pressure.</li>
        <li><strong>RH2M:</strong> Relative humidity at 2 meters.</li>
        <li><strong>QV2M:</strong> Specific humidity at 2 meters.</li>
        <li><strong>PRECTOTCORR:</strong> Corrected total precipitation.</li>
        <li><strong>T2M_MAX, T2M_MIN:</strong> Maximum and minimum 2-meter air temperature.</li>
        <li><strong>T2M_RANGE:</strong> Temperature range at 2 meters.</li>
        <li><strong>WS2M:</strong> Wind speed at 2 meters.</li>
    </ul>
    <p>After entering the district, season, and area, click the <span class="interactive-text">"Predict Crop Yield"</span> button to get the estimated crop yield.</p>""", unsafe_allow_html=True)
elif choose == "Make Prediction":
    st.subheader("Yield Prediction Interface", divider=True)
    district_selected = st.selectbox(
        "SELECT DISTRICT",
        districts,
        index=None,
        placeholder="--Select--",
    )
    # Load dataset if districts are selected
    if district_selected:
        # for district, dataset_path in zip(district_selected, dataset_paths):
        
        district=district_selected
        dataset_paths = f"2_Data/WeatherData/{district}.csv"
        dataset_path=dataset_paths
        model_paths = f"3_Models/weather_models/{district}_lstm_model.h5"
        scaler_paths = f"3_Models/weather_models/{district}_scaler.pkl"
        if os.path.exists(dataset_path):
            
            data = pd.read_csv(dataset_path)
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            numeric_columns = ['GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS2M']
            data = data[numeric_columns].dropna()
            
            data = data_processing.fill_outliers_with_median(data)
        else:
            st.write(f"No dataset found for {district}. Please check the file path.")
        selected_season = st.selectbox(
        "SELECT SEASON",
        ("Meher", "Belg"),
        index=None,
        placeholder="--Select--",
        )
        if selected_season==None:
            st.session_state.processing = True
        else:
            st.session_state.processing = False
        area = st.number_input("Enter Area(sq.m)", min_value=10, max_value=4000)
        # Function to predict the next 30 days
        
        if "processing" not in st.session_state:
            st.session_state.processing = False
        # Create a placeholder for the button
        button_placeholder = st.empty()

        # Show the button only if not processing
        if not st.session_state.processing:
            # Show the button with a unique key using a combination of the processing status
            
            if button_placeholder.button("Predict", key="run_task_button_visible"):
                st.session_state.processing = True
                button_placeholder.empty()  # Hide the button during task
                if os.path.exists(model_paths) and os.path.exists(scaler_paths):
                    # Display processing message and progress bar
                    
                    progress_bar = st.progress(0)
                    # Scale the data
                    scaler = joblib.load(scaler_paths)
                    data_scaled = scaler.transform(data)
                    
                    predictions = prediction.predict_next_30_days(model_paths, scaler_paths, data_scaled, time_steps, days=90, progress_bar=progress_bar)

                    # Convert predictions to DataFrame
                    predicted_df = pd.DataFrame(predictions, columns=['GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS2M'])
                    # st.write(f"Predicted Values for the Next 30 Days for {district_selected}:")
                    mean_predicted_df = predicted_df.mean(numeric_only=True)
                    # Append the mean row while retaining non-numeric columns
                    mean_row = pd.DataFrame([mean_predicted_df.tolist()], columns=predicted_df.columns)
                    # df1 = pd.read_csv('bbox_and_commons.csv')
                    # filtered_df1 = df1[df1['district'] == district_selected[0]]
                    filtered_df = df[df['district'] == district_selected]
                    filtered_df = filtered_df.reset_index(drop=True)
                    # important_columns = ['elevation', 'slope', 'soc', 'soilph']
                    # filtered_df = filtered_df[important_columns]
                    
                    mean_row['elevation']=filtered_df['elevation']
                    mean_row['slope']=filtered_df['slope']
                    mean_row['soc']=filtered_df['soc']
                    mean_row['soilph']=filtered_df['soilph']
                    mean_row['area(sq.m)']=area
                    mean_row['season']=selected_season
                    
                    # mean_row = mean_row[important_columns]
                    # final_df = pd.concat([mean_row, filtered_df], ignore_index=True)
                    # st.write(final_df.round(2))
                    # st.write(mean_row.round(2))
                    important_columns=['season','crop', 'area(sq.m)', 'GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS2M', 'elevation', 'slope', 'soc', 'soilph']
                    ch=pd.DataFrame()
                    crop_categories = ohe_loaded.categories_[0] 
                    for crop in crop_categories:
                        d=mean_row
                        d['crop'] = crop
                        # d.insert(1, 'crop', crop)
                        ch=pd.concat([ch,d])
                    final = ch[important_columns]
                    final=final.reset_index(drop=True)
                    encoded_final = ohe_loaded.transform(final[['crop', 'season']])
                    encoded_final = pd.DataFrame(encoded_final.toarray(), 
                              columns=[f"{val}" for cat, vals in zip(ohe_loaded.feature_names_in_, ohe_loaded.categories_) for val in vals])
                    final_df = pd.concat([final[['area(sq.m)', 
                   'GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 
                   'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 
                   'T2M_MIN', 'T2M_RANGE', 'WS2M', 'elevation', 'slope', 'soc', 'soilph',
                               ]], encoded_final], axis=1)
                    # st.write(final_df)
                    final=final_df
                    # predicted_df = predict_crop_yield(final, progress_bar=progress_bar)
                    with st.spinner("Generating Result..."):
                        predicted_df = prediction.predict_crop_yield(final, encoded_final, ohe_loaded)
                    season=predicted_df['season'].tolist()
                    tstr = f'Predicted Production in District: {district_selected}   Season: {season[0]}'
                    st.success("Prediction Done!")
                    # Plotting the predictions
                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.set_title(tstr, fontsize=15)
                    ax.set_ylabel('Production', fontsize=14)
                    ax.set_xlabel('Crop', fontsize=13)
                    ax.bar(predicted_df['crop'][:8], predicted_df['Predicted'][:8])
                    st.pyplot(fig)
                    # st.dataframe(predicted_df)
                    st.session_state.processing = False  # Reset the processing flag
                    button_placeholder.button("Predict", key="run_task_button_complete")  # Re-show the 
                
                else:
                    st.write(f"Model or Scaler for {district_selected} not found. Please check the file paths.")
elif choose == "Learn":
    st.subheader("How to", divider=True)
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Learn Python for Data Science</p>', unsafe_allow_html=True)

elif choose == "Contact":
    st.subheader("Contact System Admin", divider=True)
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Your Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')


# Retrain model section
elif choose == "Retrain Model":
    st.subheader("Retrain Model Interface", divider=True)
    district_selected = st.multiselect("Select Districts", districts)
    dataset_paths = [f"2_Data/WeatherData/{district}.csv" for district in district_selected]
    model_paths = [f"3_Models/weather_models/{district}_lstm_model.h5" for district in district_selected]
    scaler_paths = [f"3_Models/weather_models/{district}_scaler.pkl" for district in district_selected]
    # Load dataset if districts are selected
    if district_selected:
        for district, dataset_path in zip(district_selected, dataset_paths):
            if os.path.exists(dataset_path):
                
                data = pd.read_csv(dataset_path)
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
                numeric_columns = ['GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS2M']
                data = data[numeric_columns].dropna()
                data = data_processing.fill_outliers_with_median(data)
            else:
                st.write(f"No dataset found for {district}. Please check the file path.")
        if "processing" not in st.session_state:
            st.session_state.processing = False
        # Create a placeholder for the button
        button_placeholder = st.empty()
        import model_training
        # Show the button only if not processing
        if not st.session_state.processing:
            # Show the button with a unique key using a combination of the processing status
            if button_placeholder.button("Retrain", key="run_task_button_visible"):
                st.session_state.processing = True
                button_placeholder.empty()  # Hide the button during task
                with st.spinner("Please Wait, Training Model..."):
                    model_training.retrain_model_function(district_selected, dataset_paths)
                    st.session_state.processing = False  # Reset the processing flag
                    button_placeholder.button("Retrain", key="run_task_button_complete")
