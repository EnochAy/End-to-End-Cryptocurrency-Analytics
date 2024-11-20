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
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import threading
import logging
import os
import joblib
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API and database configurations
API_KEY = os.getenv('CMC_API_KEY')
if not API_KEY:
    logging.error("API key is missing. Please set it in your .env file.")
else:
    logging.info(f"API key loaded successfully: {API_KEY[:5]}******")

DB_HOST = os.getenv("MYSQL_HOST")
if not DB_HOST:
    logging.error("Database host is missing.")
else:
    logging.info(f"Database host: {DB_HOST}")

DATABASE_URI = os.getenv('DATABASE_URI', 'mysql+mysqlconnector://root:password@localhost/crypto_db')
engine = create_engine(DATABASE_URI)

# CoinMarketCap API Configuration
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {'start': '1', 'limit': '100', 'convert': 'USD'}
headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': 'CMC_API_KEY'}

# Database setup (ensure the table exists without dropping it)
def setup_database():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="EnochAy@88",
        database="crypto_db"
    )
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS crypto_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255),
        symbol VARCHAR(255),
        price DECIMAL(18, 8),
        volume_24h DECIMAL(18, 2),
        market_cap DECIMAL(18, 2),
        timestamp TIMESTAMP
    )''')
    conn.commit()
    cursor.close()
    conn.close()

setup_database()

# Fetch and store data
def fetch_and_store_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="EnochAy@88",
        database="crypto_db"
    )
    cursor = conn.cursor()

    retries = 3
    delay = 5
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=parameters)
            response.raise_for_status()
            data = response.json()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            for entry in data['data']:
                cursor.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp)
                                  VALUES (%s, %s, %s, %s, %s, %s)''',
                               (entry['name'], entry['symbol'], entry['quote']['USD']['price'],
                                entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))
            conn.commit()
            logging.info(f"Data successfully stored at {timestamp}")
            break
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Skipping this fetch.")
    cursor.close()
    conn.close()

# Load data for Dash
def load_data():
    query = "SELECT * FROM crypto_data"
    df = pd.read_sql(query, engine)
    return df

# Train a predictive model
def train_price_model():
    df = load_data()
    if df.empty:
        logging.warning("No data available for training. Skipping model training.")
        return None, None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    X = df[['time_since']]
    y = df['price']

    if len(X) == 0 or len(y) == 0:
        logging.error("Insufficient data for model training.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    logging.info("Model trained successfully.")

    joblib.dump((model, scaler), 'price_model.pkl')

    return model, scaler

# Load model and scaler if available
try:
    model, scaler = joblib.load('price_model.pkl')
    logging.info("Loaded pre-trained model.")
except FileNotFoundError:
    logging.warning("No pre-trained model found. Training a new one.")
    model, scaler = train_price_model()

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    dcc.Interval(id='interval-component', interval=600 * 1000, n_intervals=0)
])

@app.callback(
    Output('price-graph', 'figure'),
    [Input('crypto-dropdown', 'value'), Input('interval-component', 'n_intervals')]
)
def update_graph(selected_crypto, n):
    df = load_data()
    df = df[df['name'] == selected_crypto]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines+markers', name='Price'))
    fig.update_layout(title=f'Price of {selected_crypto} Over Time', xaxis_title='Time', yaxis_title='Price (USD)')

    return fig

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
    future_time = last_time + 3600
    future_time_scaled = scaler.transform(np.array([[future_time]]))

    predicted_price = model.predict(future_time_scaled)[0]
    return f"Predicted price for {selected_crypto} in 1 hour: ${predicted_price:.2f}"

def fetch_data_periodically(stop_event):
    while not stop_event.is_set():
        fetch_and_store_data()
        stop_event.wait(600)

if __name__ == '__main__':
    stop_event = threading.Event()
    fetch_thread = threading.Thread(target=fetch_data_periodically, args=(stop_event,), daemon=True)
    fetch_thread.start()

    try:
        app.run_server(debug=False)
    except KeyboardInterrupt:
        logging.info("Stopping the application...")
        stop_event.set()
        fetch_thread.join()
