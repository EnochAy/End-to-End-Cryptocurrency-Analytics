import mysql.connector
import requests
import time
import threading
import pandas as pd
from datetime import datetime

# CoinMarketCap API Configuration
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {'start': '1', 'limit': '5000', 'convert': 'USD'}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'CMC_API_KEY'  # Replace with your API key
}

# Connect to MySQL Database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="EnochAy@88",
    database="crypto_db"
)
c = conn.cursor()

# Drop table if it exists and create a new table
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
            print(f"Error fetching data from CoinMarketCap API: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this fetch.")




# Function to periodically fetch data in a separate thread
def fetch_data_periodically():
    while True:
        fetch_and_store_data()
        time.sleep(3600)  # Fetch data every hour

# Start the data fetching thread
thread = threading.Thread(target=fetch_data_periodically, daemon=True)
thread.start()

# Data Analysis Option
def load_data():
    query = "SELECT * FROM crypto_data"
    df = pd.read_sql(query, conn)
    return df

# Basic Data Analysis
def analyze_data():
    df = load_data()
    if df.empty:
        print("No data available for analysis.")
        return

    print("Summary Statistics:")
    print(df.describe())

    # Example: Show top 5 cryptocurrencies by market cap
    top_5 = df.sort_values(by='market_cap', ascending=False).head(5)
    print("\nTop 5 Cryptocurrencies by Market Cap:")
    print(top_5[['name', 'symbol', 'market_cap', 'price']])

# Main function to run data analysis
if __name__ == '__main__':
    print("Starting cryptocurrency data fetching and analysis...")
    time.sleep(10)  # Wait for some data to be fetched
    analyze_data()


#logging.info("Fetching and storing data...")
fetch_and_store_data()



