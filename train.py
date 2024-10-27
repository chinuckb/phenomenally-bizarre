import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Set random seed for reproducibility
np.random.seed(42)

# Load the data with datetime parsing
file_path = r'E:\Python Projects\A33 Hedge Fund\GoLang\phenomenally-bizarre\Data\EURUSD_Candlestick_1_M_BID_30.09.2024-05.10.2024.csv'
data = pd.read_csv(file_path)
data['Date & time'] = pd.to_datetime(data['Date & time'], format='%d.%m.%Y %H:%M:%S.%f')

# Preprocess the data
data['Price Change'] = data['Close'].diff()
data['State'] = np.where(data['Price Change'] > 0, 1, 0)  # 1 = Up (Green), 0 = Down (Red)
data.dropna(inplace=True)

# Split the data into training and testing sets
train_size = int(len(data) * 0.7)
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

# Prepare data for HMM training
Y_train = train_data['State'].values
X_train = Y_train.reshape(-1, 1)  # HMM expects a 2D array for observations

# Train the HMM
model = GaussianHMM(n_components=2, covariance_type="full", n_iter=2000, tol=0.01, random_state=42)
model.fit(X_train)

# Prepare test data for predictions
Y_test = test_data['State'].values
X_test = Y_test.reshape(-1, 1)

# Predict the states for the test set
predicted_states = model.predict(X_test)

# Match predictions with original data
test_data.loc[:, 'Predicted State'] = predicted_states
test_data.loc[:, 'Predicted Price'] = np.where(test_data['Predicted State'] == 1, test_data['Close'], np.nan)

# Fill forward to create a continuous line based on predictions
test_data['Predicted Price'] = test_data['Predicted Price'].ffill()

# Function to plot dynamic trend lines
def plot_dynamic_trend(train, test):
    plt.figure(figsize=(14, 8))
    # Plot actual close prices
    plt.plot(train['Date & time'], train['Close'], label='Actual price (Train)', color='blue', alpha=0.6)
    plt.plot(test['Date & time'], test['Close'], label='Actual price (Test)', color='blue', alpha=0.6)
    
    # Plot predicted trend based on Predicted State
    current_color = 'red' if test['Predicted State'].iloc[0] == 1 else 'green'
    for i in range(1, len(test)):
        next_color = 'red' if test['Predicted State'].iloc[i] == 1 else 'green'
        if next_color != current_color:
            plt.plot(test['Date & time'].iloc[i-1:i+1], test['Predicted Price'].iloc[i-1:i+1], color=current_color, linestyle='--')
            current_color = next_color
        else:
            plt.plot(test['Date & time'].iloc[i-1:i+1], test['Predicted Price'].iloc[i-1:i+1], color=current_color, linestyle='--')
    
    # Plot configuration
    plt.title('EUR/USD Close Prices with Predicted Trend')
    plt.xlabel('Date & Time')
    plt.ylabel('Price')

    # Custom legend
    actual_legend = mlines.Line2D([], [], color='blue', label='Actual Price')
    prediction_legend_up = mlines.Line2D([], [], color='green', linestyle='--', label='Prediction (Up Trend)')
    prediction_legend_down = mlines.Line2D([], [], color='red', linestyle='--', label='Prediction (Down Trend)')
    plt.legend(handles=[actual_legend, prediction_legend_up, prediction_legend_down], loc='best')
    
    plt.xticks(rotation=45)
    plt.show()

# Plot dynamic trend for test data only
plot_dynamic_trend(train_data, test_data)

# Compare the predicted price with the test dataset and measure accuracy
test_data.loc[:, 'Actual Price'] = test_data['Close']
accuracy = np.mean(test_data['Actual Price'] == test_data['Predicted Price'])
print(f'Model Prediction Accuracy: {accuracy * 100:.2f}%')

# Count of up state and down state
up_count = np.sum(predicted_states == 1)
down_count = np.sum(predicted_states == 0)
print(f'Count of Up State: {up_count}')
print(f'Count of Down State: {down_count}')