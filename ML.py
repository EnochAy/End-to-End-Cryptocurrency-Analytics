"""
Feature Engineering (for ML)
Create new features that might improve machine learning models.

Steps:

Derive percentage price changes, 7-day moving averages, or volume-to-market-cap ratios.
Normalize or standardize the features
"""


"""
Step 3: Machine Learning Integration (Price Prediction)
Now, let's build a simple Linear Regression model using the historical data to predict future prices of cryptocurrencies.
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data for ML training
df = load_data()
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Prepare the data for the model
df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()  # Convert time to numerical

# Feature and target variable
X = df[['time_since']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Dashboard integration: Predict future prices
@app.callback(
    Output('prediction-output', 'children'),
    [Input('crypto-dropdown', 'value')]
)
def predict_price(selected_crypto):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    
    # Use the trained model for prediction (adjust this for better models)
    last_time = (pd.to_datetime(df['timestamp'].max()) - df['timestamp'].min()).total_seconds()
    future_time = last_time + 3600  # Predict one hour into the future
    future_time_scaled = scaler.transform(np.array([[future_time]]))
    
    predicted_price = model.predict(future_time_scaled)[0]
    
    return f"Predicted price for {selected_crypto} in 1 hour: ${predicted_price:.2f}"
