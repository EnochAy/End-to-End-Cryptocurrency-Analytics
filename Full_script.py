"""
a complete end-to-end cryptocurrency analytics project incorporating data pipeline, real-time storage, 
data visualization dashboard, and machine learning for price prediction. We'll break the project down into 
the following sections:

Data Pipeline for Real-Time and Historical Data

Fetching real-time cryptocurrency data from CoinMarketCap API.
Storing the data in a database (SQLite).
Storing both real-time and historical data for deeper analysis.
Visualization Dashboard (using Plotly Dash)

Creating an interactive dashboard to explore the data dynamically.
Showing real-time prices, trends, and historical data.
Machine Learning Integration

A basic price prediction model using historical data (using a simple Linear Regression model as an example).
Incorporating the model into the dashboard for prediction insights.
Step 1: Data Pipeline for Real-Time and Historical Data
You will build a pipeline to fetch, store, and update cryptocurrency data periodically. 
This part sets up the data engineering foundation for your project."""

import requests
import sqlite3
import time
import json
from datetime import datetime

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

# Connect to SQLite Database (historical data)
conn = sqlite3.connect('crypto_data.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS crypto_data (
    id INTEGER PRIMARY KEY,
    name TEXT,
    symbol TEXT,
    price REAL,
    volume_24h REAL,
    market_cap REAL,
    timestamp TEXT
)''')
conn.commit()

# Function to fetch and store data in the database
def fetch_and_store_data():
    try:
        response = requests.get(url, headers=headers, params=parameters)
        data = response.json()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for entry in data['data']:
            c.execute('INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
                      (entry['name'], entry['symbol'], entry['quote']['USD']['price'], 
                       entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))
        
        conn.commit()
        print(f"Data successfully stored at {timestamp}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CoinMarketCap API: {e}")

# Fetch data every 10 minutes for real-time updates
while True:
    fetch_and_store_data()
    time.sleep(600)  # Sleep for 10 minutes


"""
Step 2: Visualization Dashboard (using Plotly Dash)
Next, we'll use Plotly Dash to create an interactive dashboard for visualizing cryptocurrency data. This dashboard will show real-time price trends and historical data.

Install dash, dash-bootstrap-components, and plotly via pip:
"""
#1. pip install dash dash-bootstrap-components plotly
#pip install dash dash-bootstrap

#2.Create the Dash app:
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import sqlite3

# Connect to SQLite and load data
def load_data():
    conn = sqlite3.connect('crypto_data.db')
    df = pd.read_sql_query("SELECT * FROM crypto_data", conn)
    conn.close()
    return df

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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
