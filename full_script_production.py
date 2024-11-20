import os
import dash
import pickle
import requests
import numpy as np
import pandas as pd
import mysql.connector
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly import graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from datetime import datetime
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment Variables (Set API Key Securely)
api_key = os.getenv('CMC_API_KEY', 'b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c')
db_password = os.getenv('DB_PASSWORD', 'your_db_password_here')

# API Configuration
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {'start': '1', 'limit': '5000', 'convert': 'USD'}
headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': 'b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c'}

# Database Initialization
def init_db():
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password=db_password, database='crypto_db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS crypto_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255),
                        symbol VARCHAR(255),
                        price DECIMAL(18, 8),
                        volume_24h DECIMAL(18, 2),
                        market_cap DECIMAL(18, 2),
                        timestamp TIMESTAMP,
                        UNIQUE(name, timestamp)
                    )''')
        conn.commit()
        print("Database initialized.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

# Fetch and Store Data
def fetch_and_store_data():
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password=db_password, database='crypto_db')
        c = conn.cursor()
        response = requests.get(url, headers=headers, params=parameters)
        data = response.json()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for entry in data['data']:
            c.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp)
                         VALUES (%s, %s, %s, %s, %s, %s)
                         ON DUPLICATE KEY UPDATE price=VALUES(price), volume_24h=VALUES(volume_24h), market_cap=VALUES(market_cap)''',
                      (entry['name'], entry['symbol'], entry['quote']['USD']['price'],
                       entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))
        conn.commit()
        print("Data fetched and stored successfully.")
    except Exception as e:
        print(f"Error fetching data: {e}")
    finally:
        conn.close()

# Load Data from Database
def load_data():
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password=db_password, database='crypto_db')
        df = pd.read_sql("SELECT * FROM crypto_data", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Feature Engineering
def feature_engineering(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price_change_24h'] = df['price'].pct_change().fillna(0)
    df['volume_market_cap_ratio'] = (df['volume_24h'] / df['market_cap']).fillna(0)
    df['price_ma_7d'] = df['price'].rolling(window=7).mean().fillna(df['price'])
    return df

# ARIMA Model Training
def train_arima(df):
    model = ARIMA(df['price'], order=(5, 1, 0))
    return model.fit()

# XGBoost Model Training
def train_xgboost(df):
    df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    X = df[['time_since', 'market_cap', 'volume_market_cap_ratio', 'price_change_24h']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model

# Save and Load Models
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Initialize and Train Models
def initialize_models():
    df = load_data()
    if not df.empty:
        print("Training models...")

        # Feature Engineering
        df = feature_engineering(df)

        # Train ARIMA Model
        arima_model = train_arima(df)
        save_model(arima_model, 'arima_model.pkl')
        print("ARIMA model trained successfully.")

        # Train XGBoost Model
        xgboost_model = train_xgboost(df)
        save_model(xgboost_model, 'xgboost_model.pkl')
        print("XGBoost model trained successfully.")
    else:
        print("No data available for model training.")

# Dash App Setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Cryptocurrency Dashboard"),
    dcc.Dropdown(id='crypto-dropdown', options=[{'label': 'Bitcoin', 'value': 'Bitcoin'}, {'label': 'Ethereum', 'value': 'Ethereum'}], value='Bitcoin'),
    dcc.Graph(id='price-graph'),
    html.Div(id='prediction-output', style={'fontSize': 24, 'marginTop': 20}),
    dcc.Interval(id='interval-component', interval=600 * 1000, n_intervals=0)
])

# Update Graph Callback
@app.callback(Output('price-graph', 'figure'), [Input('crypto-dropdown', 'value'), Input('interval-component', 'n_intervals')])
def update_graph(selected_crypto, n):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    if df.empty:
        return go.Figure().update_layout(title="No Data Available")

    df = feature_engineering(df)
    fig = go.Figure(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines+markers', name='Price'))
    fig.update_layout(title=f'Price of {selected_crypto} Over Time', xaxis_title='Time', yaxis_title='Price (USD)')
    return fig

# Prediction Callback
@app.callback(Output('prediction-output', 'children'), [Input('crypto-dropdown', 'value')])
def predict_price(selected_crypto):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    if df.empty:
        return "No data available for prediction."

    arima_model = load_model('arima_model.pkl')
    arima_forecast = arima_model.forecast(steps=1)[0]

    xgboost_model = load_model('xgboost_model.pkl')
    last_row = df.iloc[-1]
    future_time = (pd.to_datetime(last_row['timestamp']) - df['timestamp'].min()).total_seconds() + 3600
    x_input = np.array([[future_time, last_row['market_cap'], last_row['volume_market_cap_ratio'], last_row['price_change_24h']]])
    xgboost_forecast = xgboost_model.predict(x_input)[0]

    return f"ARIMA Predicted price: ${arima_forecast}, XGBoost Predicted price: ${xgboost_forecast}"

# Main Execution
if __name__ == '__main__':
    def init_and_train():
        init_db()
        fetch_and_store_data()
        initialize_models()

    # Run data initialization and model training in a separate thread
    threading.Thread(target=init_and_train).start()
    print("Starting Dash server...")
    app.run_server(debug=True, port=8050)
