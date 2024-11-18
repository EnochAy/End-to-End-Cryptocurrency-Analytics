import mysql.connector
import requests
import time
from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc  # For Bootstrap theme
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# CoinMarketCap API Configuration
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD'
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'your_api_key_here'  # Replace with your API key
}

# Connect to MySQL Database (historical data)
conn = mysql.connector.connect(
    host="localhost",  # Your MySQL host (e.g., localhost or IP)
    user="root",  # Your MySQL username
    password="EnochAy@88",  # Your MySQL password
    database="crypto_db"  # Your database name
)
c = conn.cursor()

# Drop table if it exists (to recreate with correct schema)
c.execute('DROP TABLE IF EXISTS crypto_data')

# Create table with the 'timestamp' column
c.execute('''CREATE TABLE IF NOT EXISTS crypto_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    symbol VARCHAR(255),
    price DECIMAL(18, 8),
    volume_24h DECIMAL(18, 2),
    market_cap DECIMAL(18, 2),
    timestamp TIMESTAMP
)''')
conn.commit()

# Function to fetch and store data in the database
def fetch_and_store_data():
    try:
        response = requests.get(url, headers=headers, params=parameters)
        data = response.json()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add current timestamp
        
        for entry in data['data']:
            c.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp) 
                         VALUES (%s, %s, %s, %s, %s, %s)''',
                      (entry['name'], entry['symbol'], entry['quote']['USD']['price'], 
                       entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))
        
        conn.commit()
        print(f"Data successfully stored at {timestamp}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CoinMarketCap API: {e}")

# Function to load data from MySQL for Dash visualization
def load_data():
    query = "SELECT * FROM crypto_data"
    df = pd.read_sql(query, conn)
    return df

df = load_data()
print(df.head())  # Check the first few rows of data

# Machine Learning - Train a Linear Regression Model for Price Prediction
def train_price_model():
    df = load_data()
    
    if df.empty:
        print("No data available for training. Skipping model training.")
        return None, None  # Return None if there's no data
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Prepare the data for the model
    df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()  # Convert time to numerical

    # Feature and target variable
    X = df[['time_since']]
    y = df['price']

    if len(X) == 0 or len(y) == 0:
        print("Insufficient data for training the model.")
        return None, None  # Return None if there isn't enough data for training

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Split resulted in empty train or test set. Check data availability.")
        return None, None

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler

# Load data and check if it's empty before training
model, scaler = train_price_model()

if model is not None:
    print("Model trained successfully.")
else:
    print("Model training skipped due to insufficient data.")


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    html.H1("Cryptocurrency Dashboard"),
    dcc.Dropdown(id='crypto-dropdown',
                 options=[
                     {'label': 'Bitcoin', 'value': 'Bitcoin'},
                     {'label': 'Ethereum', 'value': 'Ethereum'}
                 ],
                 value='Bitcoin'),
    dcc.Graph(id='price-graph'),
    html.Div(id='prediction-output'),
    dcc.Interval(id='interval-component', interval=600 * 1000, n_intervals=0)  # Refresh every 10 minutes
])

# Update graph based on selected cryptocurrency and refresh interval
@app.callback(
    Output('price-graph', 'figure'),
    [Input('crypto-dropdown', 'value'), Input('interval-component', 'n_intervals')]
)
def update_graph(selected_crypto, n):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'],
                             mode='lines+markers', name='Price'))
    
    fig.update_layout(title=f'Price of {selected_crypto} Over Time', xaxis_title='Time', yaxis_title='Price (USD)')
    
    return fig

# Predict future prices
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

# Run the app in a separate thread to keep the server running continuously
if __name__ == '__main__':
    # Train the model once
    model, scaler = train_price_model()
    
    # Fetch and store data every 10 minutes
    while True:
        fetch_and_store_data()
        time.sleep(600)  # Fetch data every 10 minutes

    app.run_server(debug=True)
