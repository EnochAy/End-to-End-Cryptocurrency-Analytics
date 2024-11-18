import mysql.connector
import requests
import time
from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from sqlalchemy import create_engine
import pandas as pd
import dash_bootstrap_components as dbc  # For Bootstrap theme
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import threading

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

# Set up the connection using SQLAlchemy
DATABASE_URI = "mysql+mysqlconnector://root:EnochAy@88@localhost/crypto_db"  # Modify this with your credentials
engine = create_engine(DATABASE_URI)

# Load data from MySQL using SQLAlchemy engine
def load_data():
    query = "SELECT * FROM crypto_data"
    df = pd.read_sql(query, engine)  # This will use SQLAlchemy for the connection
    return df

# Drop table if it exists (to recreate with correct schema)
conn = mysql.connector.connect(
    host="localhost",  # Your MySQL host (e.g., localhost or IP)
    user="root",  # Your MySQL username
    password="EnochAy@88",  # Your MySQL password
    database="crypto_db"  # Your database name
)
c = conn.cursor()
c.execute('DROP TABLE IF EXISTS crypto_data')
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

# Function to fetch and store data with retry logic
def fetch_and_store_data():
    retries = 3  # Number of retries in case of failure
    delay = 5  # Delay between retries in seconds
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=parameters)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add current timestamp

            for entry in data['data']:
                c.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp) 
                             VALUES (%s, %s, %s, %s, %s, %s)''',
                          (entry['name'], entry['symbol'], entry['quote']['USD']['price'], 
                           entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))

            conn.commit()
            print(f"Data successfully stored at {timestamp}")
            break  # If the request is successful, break out of the retry loop

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from CoinMarketCap API: {e}")
            if attempt < retries - 1:  # If it's not the last retry attempt, wait before retrying
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this fetch.")

# Machine Learning - Train a Linear Regression Model for Price Prediction
def train_price_model():
    df = load_data()
    
    if df.empty:
        print("No data available for training. Skipping model training.")
        return None, None  # Return None if there's no data
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()  # Convert time to numerical
    X = df[['time_since']]
    y = df['price']

    if len(X) == 0 or len(y) == 0:
        print("Insufficient data for training the model.")
        return None, None  # Return None if there isn't enough data for training

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Split resulted in empty train or test set. Check data availability.")
        return None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler

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

    if model is None:
        return "Model is not available for prediction."

    last_time = (pd.to_datetime(df['timestamp'].max()) - df['timestamp'].min()).total_seconds()
    future_time = last_time + 3600  # Predict one hour into the future
    future_time_scaled = scaler.transform(np.array([[future_time]]))
    
    predicted_price = model.predict(future_time_scaled)[0]
    
    return f"Predicted price for {selected_crypto} in 1 hour: ${predicted_price:.2f}"

# Run the app in a separate thread to keep the server running continuously
def fetch_data_periodically():
    while True:
        fetch_and_store_data()
        time.sleep(600)  # Fetch data every 10 minutes

if __name__ == '__main__':
    # Train the model once
    model, scaler = train_price_model()
    
    # Start the data fetching in a separate thread
    threading.Thread(target=fetch_data_periodically, daemon=True).start()

    # Run the Dash app
    app.run_server(debug=True)
