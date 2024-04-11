import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Step 1: Dataset Preparation
# Fetch the weather forecast dataset from the internet
url = 'weather_forecast_dataset.csv'  # Replace 'example.com' with the actual URL of the dataset
response = requests.get(url)
if response.status_code == 200:
    data = response.content.decode('utf-8')
    weather_data = pd.read_csv(StringIO(data))
else:
    print("Failed to fetch data. Error:", response.status_code)

# Preprocess the dataset
# Assuming 'Date' column is in datetime format, if not, convert it using:
# weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Drop any unnecessary columns if needed
# weather_data = weather_data.drop(['unnecessary_column1', 'unnecessary_column2'], axis=1)

# Perform normalization on the numerical columns
scaler = MinMaxScaler()
weather_data[['Temperature', 'Humidity', 'Wind_Speed']] = scaler.fit_transform(weather_data[['Temperature', 'Humidity', 'Wind_Speed']])

# Split the dataset into training and test sets
train_data, test_data = train_test_split(weather_data, test_size=0.2, shuffle=False)

# Step 2: Model Architecture
# Design an LSTM-based architecture capable of capturing long-term dependencies
# Experiment with stacking multiple LSTM layers and adjusting the number of units in each layer
# Consider using dropout layers to prevent overfitting and improve generalization

def create_lstm_model(input_shape, num_units=64, num_layers=1, dropout_rate=0.2):
    model = Sequential()
    for _ in range(num_layers - 1):
        model.add(LSTM(units=num_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(units=num_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    return model

# Step 3: Model Training
# Compile the LSTM model with an appropriate loss function and optimizer
# Train the model on the training set for a fixed number of epochs
# Monitor training/validation loss to ensure proper convergence

# Define input shape based on the number of features
input_shape = (train_data.shape[1], 1)  # Assuming each feature is considered separately

# Create and compile the LSTM model
lstm_model = create_lstm_model(input_shape)
lstm_model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model
history = lstm_model.fit(train_data.values.reshape(-1, train_data.shape[1], 1), train_labels, 
                         epochs=10, batch_size=32, validation_split=0.2)

# Step 4: Model Evaluation
# Evaluate the trained model on the test set using relevant evaluation metrics such as MAE or RMSE
# Visualize the model's predictions against the ground truth to assess its accuracy and performance

# Evaluate the model on the test set
test_loss = lstm_model.evaluate(test_data.values.reshape(-1, test_data.shape[1], 1), test_labels)

# Make predictions on the test set
test_predictions = lstm_model.predict(test_data.values.reshape(-1, test_data.shape[1], 1))

# Calculate evaluation metrics
mae = mean_absolute_error(test_labels, test_predictions)
rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Visualize the model's predictions against the ground truth
plt.plot(test_labels, label='True Values')
plt.plot(test_predictions, label='Predictions')
plt.title('True vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Step 5: Hyperparameter Tuning
# Experiment with different hyperparameters such as learning rate, batch size, number of LSTM units, and dropout rate
# Use techniques like grid search or random search to find the optimal set of hyperparameters

# Step 6: Discussion and Analysis
# Discuss the challenges encountered during model training and optimization
# Describe how you decided on the number of LSTM layers and units
# Explain the preprocessing steps performed on the time series data before training the model
# Analyze the purpose of dropout layers in LSTM networks and how they prevent overfitting
# Reflect on potential improvements or alternative approaches for enhancing forecasting performance
