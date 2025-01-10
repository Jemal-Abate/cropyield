from  PIL import Image
from base64 import b64encode
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
import data_processing
import io 
import joblib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as html
import tensorflow as tf
import time

time_steps = 365
def build_model(input_shape):
    """
    Builds and compiles an LSTM model.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(input_shape[1])  # Predicting all features
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

# def retrain_model(district, dataset_path, model_save_path, scaler_save_path, time_steps=365):
    # """
    # Retrains the LSTM model using district-specific data.
    # """
    # from data_processing import preprocess_data, prepare_data

    # # Preprocess data
    # data_scaled, scaler, _ = preprocess_data(dataset_path)
    # X, y = prepare_data(data_scaled, time_steps)

    # # Train-test split
    # train_size = int(0.8 * len(X))
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # # Build and train the model
    # model = build_model((time_steps, X.shape[2]))
    # early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # model.fit(
        # X_train, y_train,
        # validation_split=0.2,
        # epochs=50,
        # batch_size=32,
        # callbacks=[early_stopping],
        # verbose=1
    # )

    # # Save the model and scaler
    # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # model.save(model_save_path)
    # joblib.dump(scaler, scaler_save_path)

    # return model, scaler





def retrain_model_function(district_selected, dataset_paths):
    # Define GitHub variables
    token = st.secrets["GITHUB_TOKEN"]
    repo = "Jemal-Abate/cropyield"  # Replace with your repository
    commit_message_template = "Uploading {file_name} for district {district}"

    
    # Function to upload a file to GitHub
    def upload_to_github1(local_path, repo_path, commit_message):
        # Step 1: Check if the file exists in the repository
        url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        response = requests.get(url, headers=headers)        
        # Step 2: Get the sha of the file if it exists
        sha = None
        if response.status_code == 200:
            sha = response.json()['sha']        
        # Step 3: Read the file content and encode it
        with open(local_path, "rb") as file:
            file_content = file.read()
        encoded_content = b64encode(file_content).decode()        
        # Step 4: Prepare the data for uploading
        data = {
            "message": commit_message,
            "content": encoded_content
        }        
        # Include sha in the data if file exists
        if sha:
            data["sha"] = sha        
        # Step 5: Upload the file to GitHub
        response = requests.put(url, json=data, headers=headers)

    # Main training logic
    total_districts = len(district_selected)
    i = 0
    district_progress = st.progress(0)
    for district, dataset_path in zip(district_selected, dataset_paths):
        i += 1
        progress_desc = f"Processing district ({i}/{total_districts})"
        district_progress.progress(i / total_districts, text=progress_desc)
        data_scaled, scaler, data_original = data_processing.preprocess_data(dataset_path)
        X, y = data_processing.prepare_data(data_scaled, time_steps)

        # Train-test split
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train the model
        model = build_model((time_steps, X.shape[2]))
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        total_epoch = 20
        perc=(1/total_epoch) * 100
        progress_bar = st.progress(perc, text=f"{district} ({int(perc)}%)")
        def on_epoch_end(epoch, logs):
            percentage = ((epoch + 1) / total_epoch) * 100
            progress_bar.progress((epoch + 1) / total_epoch, text=f"{district} ({int(percentage)}%)")

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=total_epoch,
            batch_size=32,
            callbacks=[early_stopping, tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)],
            verbose=0
        )

        # Save the model and scaler locally
        os.makedirs("3_Models/weather_models", exist_ok=True)
        model_save_path = f"3_Models/weather_models/{district}_lstm_model.h5"
        scaler_save_path = f"3_Models/weather_models/{district}_scaler.pkl"
        model.save(model_save_path)
        joblib.dump(scaler, scaler_save_path)
        # Function to upload a file to GitHub
        def upload_to_github(local_path, repo_path, commit_message):
            # Step 1: Check if the file exists in the repository
            url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            response = requests.get(url, headers=headers)        
            # Step 2: Get the sha of the file if it exists
            sha = None
            if response.status_code == 200:
                sha = response.json()['sha']        
            # Step 3: Read the file content and encode it
            with open(local_path, "rb") as file:
                file_content = file.read()
            encoded_content = b64encode(file_content).decode()        
            # Step 4: Prepare the data for uploading
            data = {
                "message": commit_message,
                "content": encoded_content
            }        
            # Include sha in the data if file exists
            if sha:
                data["sha"] = sha        
            # Step 5: Upload the file to GitHub
            response = requests.put(url, json=data, headers=headers)
        # Upload the files to GitHub
        # model_repo_path = f"models/{district}_lstm_model.h5"
        model_repo_path = model_save_path
        # scaler_repo_path = f"models/{district}_scaler.pkl"
        scaler_repo_path = scaler_save_path
        upload_to_github(model_save_path, model_repo_path, commit_message_template.format(file_name="model", district=district))
        upload_to_github(scaler_save_path, scaler_repo_path, commit_message_template.format(file_name="scaler", district=district))

        progress_bar.empty()
        
        # st.write(f"Model and scaler have been retrained and saved for {district}")
    district_progress.empty()
    st.success("Training Completed!")