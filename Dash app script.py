import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import mysql.connector
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
#from prophet import Prophet
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
import dash_bootstrap_components as dbc
from datetime import datetime
from cachetools import cached, TTLCache

# Load environment variables for security
DB_USER = os.getenv('DB_USER', 'host')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'EnochAy@88')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'crypto_db')

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Cache for database queries (TTL: 10 minutes)
cache = TTLCache(maxsize=10, ttl=600)

# Connect to MySQL and load data
@cached(cache)
def load_data():
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    query = "SELECT * FROM crypto_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Feature Engineering
def feature_engineering(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price_change_24h'] = df['price'].pct_change()
    df['volume_market_cap_ratio'] = df['volume_24h'] / df['market_cap']
    df['price_ma_7d'] = df['price'].rolling(window=7).mean()
    df.fillna(0, inplace=True)
    return df

# ARIMA Model
def train_arima(df):
    model = ARIMA(df['price'], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

# Prophet Model
def train_prophet(df):
    prophet_df = df[['timestamp', 'price']].rename(columns={'timestamp': 'ds', 'price': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    return model

# LSTM Model
def train_lstm(df):
    data = df[['price']].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Prepare data for LSTM
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)
    return model, scaler

# App layout
app.layout = html.Div([
    html.H1("Advanced Cryptocurrency Dashboard"),
    dcc.Dropdown(id='crypto-dropdown',
                 options=[
                     {'label': 'Bitcoin', 'value': 'Bitcoin'},
                     {'label': 'Ethereum', 'value': 'Ethereum'}
                 ],
                 value='Bitcoin'),
    dcc.Graph(id='price-graph'),
    dcc.RadioItems(id='model-selector',
                   options=[
                       {'label': 'ARIMA', 'value': 'ARIMA'},
                       {'label': 'Prophet', 'value': 'Prophet'},
                       {'label': 'LSTM', 'value': 'LSTM'}
                   ],
                   value='ARIMA'),
    html.Div(id='prediction-output', style={'fontSize': 24, 'marginTop': 20}),
    dcc.Interval(id='interval-component', interval=3600 * 1000, n_intervals=0)
])

# Update Graph
@app.callback(
    Output('price-graph', 'figure'),
    [Input('crypto-dropdown', 'value'), Input('interval-component', 'n_intervals')]
)
def update_graph(selected_crypto, n):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    df = feature_engineering(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name='Price'))
    fig.update_layout(title=f'{selected_crypto} Price Over Time', xaxis_title='Time', yaxis_title='Price (USD)')

    return fig

# Prediction Callback
@app.callback(
    Output('prediction-output', 'children'),
    [Input('crypto-dropdown', 'value'), Input('model-selector', 'value')]
)
def predict_price(selected_crypto, model_choice):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    df = feature_engineering(df)

    if model_choice == 'ARIMA':
        model = train_arima(df)
        forecast = model.forecast(steps=1)[0]
    elif model_choice == 'Prophet':
        model = train_prophet(df)
        future = pd.DataFrame({'ds': [df['timestamp'].max() + pd.Timedelta(hours=1)]})
        forecast = model.predict(future)['yhat'].values[0]
    elif model_choice == 'LSTM':
        model, scaler = train_lstm(df)
        last_60 = df[['price']].values[-60:]
        scaled_last_60 = scaler.transform(last_60)
        X_test = np.expand_dims(scaled_last_60, axis=0)
        forecast = scaler.inverse_transform(model.predict(X_test))[0][0]
    else:
        return "Invalid model selected."

    return f"Predicted price for {selected_crypto} in 1 hour: ${forecast:.2f}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
