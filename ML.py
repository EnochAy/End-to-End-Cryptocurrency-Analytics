import mysql.connector
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dash import Dash, html, dcc, Input, Output
import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Connect to MySQL Database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="EnochAy@88",
        database="crypto_db"
    )

# Function to load data from MySQL database
def load_data():
    conn = get_db_connection()
    query = "SELECT * FROM crypto_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Feature Engineering: Create new features for ML
def feature_engineering(df):
    # Ensure timestamps are parsed correctly
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate percentage price change
    df['price_change_pct'] = df['price'].pct_change() * 100
    
    # Calculate 7-day moving average
    df['price_ma_7d'] = df['price'].rolling(window=7, min_periods=1).mean()
    
    # Volume-to-market-cap ratio
    df['vol_to_market_cap'] = df['volume_24h'] / df['market_cap']
    
    # Time since the earliest record (for numerical modeling)
    df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    
    # Drop rows with NaN values introduced by calculations
    df.dropna(inplace=True)
    
    return df

# Save model and scaler for later use
def save_model(model, scaler):
    with open('price_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

# Load saved model and scaler
def load_model():
    with open('price_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Train the Linear Regression model
def train_price_model():
    # Load and prepare the data
    df = load_data()
    
    # Perform feature engineering
    df = feature_engineering(df)
    
    if df.empty:
        print("No data available for training. Skipping model training.")
        return None, None
    
    # Feature and target variable
    X = df[['time_since', 'price_change_pct', 'price_ma_7d', 'vol_to_market_cap']]
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
    
    print("Model trained successfully.")
    
    # Save model and scaler
    save_model(model, scaler)
    
    return model, scaler

# Initialize Dash app
app = Dash(__name__)

# Layout for the dashboard
app.layout = html.Div([
    html.H1("Cryptocurrency Price Prediction Dashboard"),
    dcc.Dropdown(
        id='crypto-dropdown',
        options=[],  # Dynamically populated
        placeholder="Select a cryptocurrency"
    ),
    html.Div(id='prediction-output', style={'margin-top': '20px'}),
    dcc.Graph(id='price-graph', style={'margin-top': '20px'}),
])

# Check if model and scaler exist, otherwise train new model
try:
    model, scaler = load_model()
    print("Model and scaler loaded.")
except FileNotFoundError:
    model, scaler = train_price_model()
    print("Model trained and saved.")

# Populate dropdown options dynamically
@app.callback(
    Output('crypto-dropdown', 'options'),
    Input('crypto-dropdown', 'value')
)
def update_dropdown(_):
    df = load_data()
    options = [{'label': name, 'value': name} for name in df['name'].unique()]
    return options

# Predict future prices and update graph
@app.callback(
    [Output('prediction-output', 'children'),
     Output('price-graph', 'figure')],
    [Input('crypto-dropdown', 'value')]
)
def predict_and_display(selected_crypto):
    if not selected_crypto:
        return "Please select a cryptocurrency.", {}

    df = load_data()
    df = df[df['name'] == selected_crypto]
    
    if df.empty:
        return f"No data available for {selected_crypto}.", {}

    # Perform feature engineering
    df = feature_engineering(df)
    
    # Predict future price (1 hour ahead)
    last_time = df['time_since'].max()
    future_time = last_time + 3600  # Predict 1 hour into the future
    last_row_features = df[['price_change_pct', 'price_ma_7d', 'vol_to_market_cap']].iloc[-1].values
    future_data = np.array([[future_time] + list(last_row_features)])
    
    try:
        future_data_scaled = scaler.transform(future_data)
        predicted_price = model.predict(future_data_scaled)[0]
        
        # Create a graph of historical prices
        fig = {
            'data': [
                {'x': df['timestamp'], 'y': df['price'], 'type': 'line', 'name': selected_crypto},
            ],
            'layout': {
                'title': f"Historical Prices for {selected_crypto}",
                'xaxis': {'title': 'Timestamp'},
                'yaxis': {'title': 'Price (USD)'}
            }
        }
        
        return f"Predicted price for {selected_crypto} in 1 hour: ${predicted_price:.2f}", fig
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction error. Please check the logs.", {}

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
