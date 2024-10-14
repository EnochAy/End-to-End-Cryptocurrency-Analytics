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
