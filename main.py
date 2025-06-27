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
import logging

# Configure logging
logging.basicConfig(
    filename='crypto_analytics.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Load environment variables
load_dotenv()
logger.info("Loading environment variables from .env file")
api_key = os.getenv('CMC_API_KEY')
db_password = os.getenv('DB_PASSWORD')

if not api_key or not db_password:
    logger.error("Missing CMC_API_KEY or DB_PASSWORD in .env file")
    raise ValueError("CMC_API_KEY and DB_PASSWORD must be set in .env file")

# API Configuration
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {'start': '1', 'limit': '5000', 'convert': 'USD'}
headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': api_key}

# Database Initialization
def init_db():
    """Initialize MySQL database and create crypto_data table if it doesn't exist."""
    logger.info("Initializing database")
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
        logger.info("Database initialized successfully")
    except mysql.connector.Error as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

# Fetch and Store Data
def fetch_and_store_data():
    """Fetch cryptocurrency data from CoinMarketCap API and store in MySQL."""
    logger.info("Starting data fetch from CoinMarketCap API")
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password=db_password, database='crypto_db')
        c = conn.cursor()
        response = requests.get(url, headers=headers, params=parameters)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Fetched {len(data['data'])} records from API")

        for entry in data['data']:
            logger.debug(f"Inserting data for {entry['name']}")
            c.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp)
                         VALUES (%s, %s, %s, %s, %s, %s)
                         ON DUPLICATE KEY UPDATE price=VALUES(price), volume_24h=VALUES(volume_24h), market_cap=VALUES(market_cap)''',
                      (entry['name'], entry['symbol'], entry['quote']['USD']['price'],
                       entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))
        conn.commit()
        logger.info("Data stored successfully in database")
    except requests.RequestException as e:
        logger.error(f"Error fetching data from API: {e}")
        raise
    except mysql.connector.Error as e:
        logger.error(f"Error storing data in database: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

# Load Data from Database
def load_data():
    """Load cryptocurrency data from MySQL database into a Pandas DataFrame."""
    logger.info("Loading data from database")
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password=db_password, database='crypto_db')
        df = pd.read_sql("SELECT * FROM crypto_data", conn)
        logger.info(f"Loaded {len(df)} records from database")
        conn.close()
        return df
    except mysql.connector.Error as e:
        logger.error(f"Error loading data from database: {e}")
        return pd.DataFrame()
    finally:
        logger.info("Database connection closed")

# Feature Engineering
def feature_engineering(df):
    """Add derived features to the DataFrame for analysis and modeling."""
    logger.info("Starting feature engineering")
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price_change_24h'] = df['price'].pct_change().fillna(0)
        df['volume_market_cap_ratio'] = (df['volume_24h'] / df['market_cap']).fillna(0)
        df['price_ma_7d'] = df['price'].rolling(window=7).mean().fillna(df['price'])
        logger.info("Feature engineering completed")
        return df
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

# ARIMA Model Training
def train_arima(df):
    """Train an ARIMA model on price data."""
    logger.info("Training ARIMA model")
    try:
        model = ARIMA(df['price'], order=(5, 1, 0))
        fitted_model = model.fit()
        logger.info("ARIMA model training completed")
        return fitted_model
    except Exception as e:
        logger.error(f"Error training ARIMA model: {e}")
        raise

# XGBoost Model Training
def train_xgboost(df):
    """Train an XGBoost model for price prediction."""
    logger.info("Training XGBoost model")
    try:
        df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        X = df[['time_since', 'market_cap', 'volume_market_cap_ratio', 'price_change_24h']]
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)
        logger.info("XGBoost model training completed")
        return model
    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}")
        raise

# Save and Load Models
def save_model(model, filename):
    """Save a trained model to a file."""
    logger.info(f"Saving model to {filename}")
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved successfully to {filename}")
    except Exception as e:
        logger.error(f"Error saving model to {filename}: {e}")
        raise

def load_model(filename):
    """Load a trained model from a file."""
    logger.info(f"Loading model from {filename}")
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded successfully from {filename}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filename}: {e}")
        raise

# Initialize and Train Models
def initialize_models():
    """Load data, perform feature engineering, and train models."""
    logger.info("Initializing model training process")
    df = load_data()
    if not df.empty:
        logger.info("Data loaded, proceeding with feature engineering and model training")
        df = feature_engineering(df)
        arima_model = train_arima(df)
        save_model(arima_model, 'models/arima_model.pkl')
        xgboost_model = train_xgboost(df)
        save_model(xgboost_model, 'models/xgboost_model.pkl')
    else:
        logger.warning("No data available for model training")

# Dash App Setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
logger.info("Initializing Dash app")

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
    """Update the price graph based on selected cryptocurrency."""
    logger.info(f"Updating graph for {selected_crypto}")
    try:
        df = load_data()
        df = df[df['name'] == selected_crypto]
        if df.empty:
            logger.warning(f"No data available for {selected_crypto}")
            return go.Figure().update_layout(title="No Data Available")
        df = feature_engineering(df)
        fig = go.Figure(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines+markers', name='Price'))
        fig.update_layout(title=f'Price of {selected_crypto} Over Time', xaxis_title='Time', yaxis_title='Price (USD)')
        logger.info(f"Graph updated successfully for {selected_crypto}")
        return fig
    except Exception as e:
        logger.error(f"Error updating graph for {selected_crypto}: {e}")
        return go.Figure().update_layout(title="Error Generating Graph")

# Prediction Callback
@app.callback(Output('prediction-output', 'children'), [Input('crypto-dropdown', 'value')])
def predict_price(selected_crypto):
    """Predict the next price for the selected cryptocurrency using ARIMA and XGBoost."""
    logger.info(f"Generating price prediction for {selected_crypto}")
    try:
        df = load_data()
        df = df[df['name'] == selected_crypto]
        if df.empty:
            logger.warning(f"No data available for prediction for {selected_crypto}")
            return "No data available for prediction."
        arima_model = load_model('models/arima_model.pkl')
        arima_forecast = arima_model.forecast(steps=1)[0]
        xgboost_model = load_model('models/xgboost_model.pkl')
        last_row = df.iloc[-1]
        future_time = (pd.to_datetime(last_row['timestamp']) - df['timestamp'].min()).total_seconds() + 3600
        x_input = np.array([[future_time, last_row['market_cap'], last_row['volume_market_cap_ratio'], last_row['price_change_24h']]])
        xgboost_forecast = xgboost_model.predict(x_input)[0]
        logger.info(f"Prediction generated for {selected_crypto}: ARIMA=${arima_forecast}, XGBoost=${xgboost_forecast}")
        return f"ARIMA Predicted price: ${arima_forecast:.2f}, XGBoost Predicted price: ${xgboost_forecast:.2f}"
    except Exception as e:
        logger.error(f"Error generating prediction for {selected_crypto}: {e}")
        return "Error generating prediction"

# Main Execution
if __name__ == '__main__':
    logger.info("Starting main execution")
    def init_and_train():
        """Initialize database, fetch data, and train models in a separate thread."""
        logger.info("Starting initialization and training thread")
        init_db()
        fetch_and_store_data()
        initialize_models()
        logger.info("Initialization and training completed")

    threading.Thread(target=init_and_train).start()
    logger.info("Starting Dash server on port 8050")
    app.run_server(debug=True, port=8050)