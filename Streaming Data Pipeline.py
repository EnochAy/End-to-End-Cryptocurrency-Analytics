import mysql.connector
import requests
import threading
import time
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# CoinMarketCap API Configuration
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD'
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': os.getenv('CMC_API_KEY')  # Fetch API key from .env file
}

# Connect to MySQL Database using credentials from .env
conn = mysql.connector.connect(
    host=os.getenv('MYSQL_HOST'),
    user=os.getenv('MYSQL_USER'),
    password=os.getenv('MYSQL_PASSWORD'),
    database=os.getenv('MYSQL_DB')
)
c = conn.cursor()

# Create Table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS crypto_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    symbol VARCHAR(255),
    price DECIMAL(18, 8),
    volume_24h DECIMAL(18, 2),
    market_cap DECIMAL(18, 2),
    timestamp TIMESTAMP
)
''')
conn.commit()

# Function to fetch and store data with retry logic
def fetch_and_store_data():
    retries = 3
    delay = 5  # Delay between retries in seconds
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=parameters)
            response.raise_for_status()  # Check for HTTP errors
            data = response.json()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Insert data into MySQL
            for entry in data['data']:
                c.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp)
                             VALUES (%s, %s, %s, %s, %s, %s)''',
                          (entry['name'], entry['symbol'], entry['quote']['USD']['price'],
                           entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))

            conn.commit()
            print(f"Data successfully stored at {timestamp}")
            break  # Exit loop if successful

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this fetch.")

# Function to run the data fetch in a separate thread
def start_data_streaming(interval=3600):
    def streaming():
        while True:
            fetch_and_store_data()
            time.sleep(interval)  # Fetch data at the specified interval

    # Start the thread
    thread = threading.Thread(target=streaming)
    thread.daemon = True  # Daemon thread will exit when the main program exits
    thread.start()
    print("Data streaming started.")

# Function for simple data analysis (e.g., average price)
def analyze_data():
    query = "SELECT name, AVG(price) as avg_price FROM crypto_data GROUP BY name"
    df = pd.read_sql(query, conn)
    print("Average Price Analysis:")
    print(df)

# Main Execution
if __name__ == "__main__":
    start_data_streaming(interval=3600)  # Fetch data every hour

    # Example analysis call (can be triggered based on certain conditions)
    time.sleep(5)  # Wait a bit before running analysis
    analyze_data()

    # Keep the main program running
    while True:
        time.sleep(10)
