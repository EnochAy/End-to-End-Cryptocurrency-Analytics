import mysql.connector
import requests
import time
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

# Connect to MySQL Database (historical data)
conn = mysql.connector.connect(
    host="localhost",          # Your MySQL host (e.g., localhost or IP)
    user="root",       # Your MySQL username
    password="EnochAy@88",   # Your MySQL password
    database="crypto_db"        # Your database name
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

# Fetch data every 10 minutes for real-time updates
while True:
    fetch_and_store_data()
    time.sleep(600)  # Sleep for 10 minutes



"""
A complete end-to-end cryptocurrency analytics project incorporating data pipeline, real-time storage, 
data visualization dashboard, and machine learning for price prediction. I'll break the project down into 
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
This part sets up the data engineering foundation for your project.
"""


"""
Project Overview:
Data Pipeline:

Fetches and stores real-time cryptocurrency data from the CoinMarketCap API every 10 minutes.
Stores data in an SQLite database for historical analysis.
Dashboard:

Displays real-time and historical price trends using Plotly Dash.
Updates dynamically every 10 minutes.
Allows the user to select different cryptocurrencies for analysis.
Machine Learning:

A Linear Regression model predicts future cryptocurrency prices.
Integrates predictions into the dashboard for real-time forecasting.
"""



import sqlite3
import requests
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

# Drop table if it exists (to recreate with correct schema)
c.execute('DROP TABLE IF EXISTS crypto_data')

# Create table with the 'timestamp' column
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
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add current timestamp
        
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







# Add timestamp column if it doesn't exist
try:
    c.execute('ALTER TABLE crypto_data ADD COLUMN timestamp TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    print(f"Error adding timestamp column: {e}")



"""
Step 2: Visualization Dashboard (using Plotly Dash)
Next, I'll use Plotly Dash to create an interactive dashboard for visualizing cryptocurrency data. 
This dashboard will show real-time price trends and historical data.

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

"""
Cryptocurrency Market Monitoring and Insights
Use Case: Provide real-time or near real-time insights on market trends, price movements, and trading volumes 
of various cryptocurrencies.
Application: This project can be used by traders, investors, or portfolio managers to track specific 
cryptocurrencies, analyze market volatility, and identify profitable trading opportunities.
This will be acheived by building dashboards that show live price updates, top gainers/losers, 
and market cap changes, helping users make informed investment decisions.
"""

