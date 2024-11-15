import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import dash_bootstrap_components as dbc
from datetime import datetime
import requests

# API Configuration
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {'start': '1', 'limit': '5000', 'convert': 'USD'}
headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': 'your_api_key_here'}

# Connect to MySQL and create table if not exists
def init_db():
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password='EnochAy@88', database='crypto_db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS crypto_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255),
                        symbol VARCHAR(255),
                        price DECIMAL(18, 8),
                        volume_24h DECIMAL(18, 2),
                        market_cap DECIMAL(18, 2),
                        timestamp TIMESTAMP)''')
        conn.commit()
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

# Fetch and store data
def fetch_and_store_data():
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password='EnochAy@88', database='crypto_db')
        c = conn.cursor()
        response = requests.get(url, headers=headers, params=parameters)
        data = response.json()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for entry in data['data']:
            c.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp)
                         VALUES (%s, %s, %s, %s, %s, %s)''',
                      (entry['name'], entry['symbol'], entry['quote']['USD']['price'],
                       entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))
        conn.commit()
        print("Data fetched and stored successfully.")
    except Exception as e:
        print(f"Error fetching data: {e}")
    finally:
        conn.close()

# Load data
def load_data():
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password='EnochAy@88', database='crypto_db')
        query = "SELECT * FROM crypto_data"
        df = pd.read_sql(query, conn)
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

# Train ARIMA model
def train_arima(df):
    try:
        model = ARIMA(df['price'], order=(5, 1, 0))
        arima_model = model.fit()
        return arima_model
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None

# Train XGBoost model
def train_xgboost(df):
    try:
        df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        X = df[['time_since', 'market_cap', 'volume_market_cap_ratio', 'price_change_24h']]
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        return None

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    html.H1("Cryptocurrency Dashboard"),
    dcc.Dropdown(id='crypto-dropdown',
                 options=[{'label': 'Bitcoin', 'value': 'Bitcoin'}, {'label': 'Ethereum', 'value': 'Ethereum'}],
                 value='Bitcoin'),
    dcc.Graph(id='price-graph'),
    dcc.Interval(id='interval-component', interval=600 * 1000, n_intervals=0),
    html.Div(id='prediction-output', style={'fontSize': 24, 'marginTop': 20})
])

# Update graph
@app.callback(
    Output('price-graph', 'figure'),
    [Input('crypto-dropdown', 'value'), Input('interval-component', 'n_intervals')]
)
def update_graph(selected_crypto, n):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    if df.empty:
        return go.Figure().update_layout(title="No Data Available")

    df = feature_engineering(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines+markers', name='Price'))
    fig.update_layout(title=f'Price of {selected_crypto} Over Time', xaxis_title='Time', yaxis_title='Price (USD)')
    return fig

# Make predictions
@app.callback(
    Output('prediction-output', 'children'),
    [Input('crypto-dropdown', 'value')]
)
def predict_price(selected_crypto):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    if df.empty:
        return "No data available for prediction."

    df = feature_engineering(df)

    # ARIMA prediction
    arima_model = train_arima(df)
    if arima_model:
        arima_forecast = arima_model.forecast(steps=1)[0]
    else:
        arima_forecast = "N/A"

    # XGBoost prediction
    xgboost_model = train_xgboost(df)
    if xgboost_model:
        last_row = df.iloc[-1]
        future_time = (pd.to_datetime(last_row['timestamp']) - df['timestamp'].min()).total_seconds() + 3600
        x_input = np.array([[future_time, last_row['market_cap'], last_row['volume_market_cap_ratio'], last_row['price_change_24h']]])
        xgboost_forecast = xgboost_model.predict(x_input)[0]
    else:
        xgboost_forecast = "N/A"

    return (f"ARIMA Predicted price for {selected_crypto} in 1 hour: ${arima_forecast}, "
            f"XGBoost Predicted price: ${xgboost_forecast}")

# Run the app
if __name__ == '__main__':
    init_db()
    fetch_and_store_data()
    app.run_server(debug=True)
