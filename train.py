import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Step 1: Load the data with datetime parsing
file_path = r'E:\Python Projects\A33 Hedge Fund\GoLang\phenomenally-bizarre\Data\EURUSD_Candlestick_1_M_BID_30.09.2024-05.10.2024.csv'
data = pd.read_csv(file_path)

# Convert "Date & time" to datetime format
data['Date & time'] = pd.to_datetime(data['Date & time'], format='%d.%m.%Y %H:%M:%S.%f')

# Step 2: Preprocess the data
data['Price Change'] = data['Close'].diff()
data['State'] = np.where(data['Price Change'] > 0, 1, 0)  # 1 = Up (Green), 0 = Down (Red)
data.dropna(inplace=True)

# Step 3: Split the data into training and testing sets
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

# Step 4: Prepare data for HMM training
Y_train = train_data['State'].values
X_train = Y_train.reshape(-1, 1)  # HMM expects a 2D array for observations

# Step 5: Train the HMM
model = hmm.MultinomialHMM(n_components=2, n_iter=1000, tol=0.01)
model.fit(X_train)

# Step 6: Prepare test data for predictions
Y_test = test_data['State'].values
X_test = Y_test.reshape(-1, 1)

# Step 7: Predict the states for the test set
predicted_states = model.predict(X_test)

# Step 8: Match predictions with original data
test_data['Predicted State'] = predicted_states
test_data['Predicted Price'] = np.where(test_data['Predicted State'] == 1, test_data['Close'], np.nan)

# Step 9: Plot the actual prices and predicted trend line
plt.figure(figsize=(14, 8))

# Plot actual close prices
plt.plot(data['Date & time'], data['Close'], label='Actual Close Price', color='blue', alpha=0.6)

# Plot predicted trend based on Predicted State
# Use ffill to create a continuous line based on predictions
test_data['Predicted Price'] = test_data['Predicted Price'].ffill()
plt.plot(test_data['Date & time'], test_data['Predicted Price'], label='Predicted Trend', color='red', linestyle='--')

# Plot configuration
plt.title('EUR/USD Close Prices with Predicted Trend')
plt.xlabel('Date & Time')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()
