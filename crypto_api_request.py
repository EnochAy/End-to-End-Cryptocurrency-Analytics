import requests
import time
from datetime import datetime
import mysql.connector

# CoinMarketCap API Configuration
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {'start': '1', 'limit': '5000', 'convert': 'USD'}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'your_api_key_here'  # Replace with your API key
}

# Connect to MySQL Database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="EnochAy@88",
    database="crypto_db"
)
c = conn.cursor()

# Drop table if it exists
c.execute('DROP TABLE IF EXISTS crypto_data')

# Create table
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
    retries = 3
    delay = 5

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=parameters)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for entry in data['data']:
                c.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp) 
                             VALUES (%s, %s, %s, %s, %s, %s)''',
                          (entry['name'], entry['symbol'], entry['quote']['USD']['price'],
                           entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))

            conn.commit()
            print(f"Data successfully stored at {timestamp}")
            break  # Exit the loop if successful

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this fetch.")

# Fetch data every hour
while True:
    fetch_and_store_data()
    time.sleep(3600)
