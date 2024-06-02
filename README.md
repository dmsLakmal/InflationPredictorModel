# Inflation Predictor ML Model

## Overview

The Inflation Predictor ML Model is designed to forecast inflation rates based on key economic indicators such as GDP and unemployment rates. This model leverages machine learning techniques, specifically neural networks, to predict inflation trends. The model is trained on historical economic data to provide accurate predictions for future inflation rates.

## Model Creation

The model creation process involves several steps:

1. **Data Collection**: Historical data on GDP, unemployment rates, and inflation rates is collected and loaded into the model.

2. **Data Preprocessing**: The data is preprocessed to handle missing values, normalize features, and prepare it for model training.

3. **Model Architecture**: A neural network model is built using TensorFlow/Keras. The model architecture consists of input layers, hidden layers, and an output layer.

4. **Model Training**: The neural network model is trained using the preprocessed data. Training involves optimizing model parameters to minimize the loss function.

5. **Model Evaluation**: The trained model is evaluated using metrics such as mean squared error (MSE) and R² score to assess its performance.

6. **Model Saving**: Once trained, the model and scaler used for data normalization are saved to disk for future use.

## Libraries Used

The following libraries are used in this project:

- Pandas: For data manipulation and analysis.
- Scikit-learn: For data preprocessing and model evaluation.
- TensorFlow/Keras: For building and training the neural network model.
- Pickle: For saving and loading the scaler object.

## How the Model Works

1. **Data Input**: The model takes input data consisting of GDP and unemployment rates for a given time period.

2. **Data Preprocessing**: The input data is preprocessed to normalize features and prepare it for model prediction.

3. **Prediction**: The preprocessed data is fed into the trained neural network model, which then predicts the inflation rate for the corresponding time period.

4. **Output**: The model provides the predicted inflation rate as output, which can be used for economic analysis and forecasting.

## Repository Structure

- `model_creation.py`: Python script containing the code for creating and training the machine learning model.
- `clean_data.csv`: Cleaned and preprocessed dataset used for model training.
- `model.h5`: Trained neural network model saved in HDF5 format.
- `scaler.pkl`: Scaler object used for data normalization saved using the pickle module.

## Usage

To use the Inflation Predictor ML Model:

1. Clone the repository to your local machine.
2. Ensure that Python and the required libraries are installed.
3. Run the `model_creation.py` script to create and train the model.
4. After training, the model and scaler will be saved to disk for future use.

## Frontend

For accessing the frontend application built using the Inflation Predictor ML Model, you can visit the following GitHub repository: [Inflation Predictor ML Flask App](https://github.com/dmsLakmal/Inflation-Predictor-ML-Flask-App.git). The frontend provides an interactive web-based interface for inputting economic indicators and receiving predicted inflation rates.

## Conclusion

The Inflation Predictor ML Model provides a valuable tool for forecasting inflation rates based on economic indicators. However, it's essential to note that this model is currently in its prototype stage and may not be suitable for real-world scenarios. Creating a robust and reliable model for accurate inflation prediction requires additional data and more complex processes.

To develop a model suitable for real-world deployment, it's necessary to gather comprehensive economic data, including additional indicators and historical trends. Moreover, implementing more sophisticated machine learning techniques and refining the model architecture can enhance prediction accuracy and reliability.

While the Inflation Predictor ML Model serves as a starting point for exploring inflation trends and patterns, further research and development are needed to create a production-ready solution for economic forecasting and analysis.

