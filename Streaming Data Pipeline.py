import logging
import sys
import os
import requests
import threading
import time
from datetime import datetime
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CoinMarketCap API Configuration
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD'
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': os.getenv('CMC_API_KEY')
}

# Connect to MySQL Database
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DB')
    )

# Create Table if it doesn't exist
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS crypto_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255),
                        symbol VARCHAR(255),
                        price DECIMAL(18, 8),
                        volume_24h DECIMAL(18, 2),
                        market_cap DECIMAL(18, 2),
                        timestamp TIMESTAMP)''')
    conn.commit()
    conn.close()

# Function to fetch and store data with retry logic
def fetch_and_store_data():
    retries = 3
    delay = 5
    conn = get_db_connection()
    cursor = conn.cursor()

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=parameters)
            response.raise_for_status()  # Raise error for HTTP issues
            data = response.json()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Insert data into MySQL
            for entry in data['data']:
                cursor.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp)
                                  VALUES (%s, %s, %s, %s, %s, %s)''',
                               (entry['name'], entry['symbol'], entry['quote']['USD']['price'],
                                entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap'], timestamp))

            conn.commit()
            logger.info(f"Data successfully stored at {timestamp}")
            break  # Exit loop if successful

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached. Skipping this fetch.")

    conn.close()

# Function to run the data fetch in a separate thread
def start_data_streaming(interval=3600):
    def streaming():
        while True:
            fetch_and_store_data()
            time.sleep(interval)  # Fetch data every hour

    thread = threading.Thread(target=streaming)
    thread.daemon = True
    thread.start()
    logger.info("Data streaming started.")

# Function for simple data analysis (e.g., average price)
def analyze_data():
    query = "SELECT name, AVG(price) as avg_price FROM crypto_data GROUP BY name"
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    logger.info("Average Price Analysis:")
    logger.info(df)

# Graceful shutdown signal handler
def signal_handler(signal, frame):
    logger.info("Gracefully shutting down...")
    sys.exit(0)

import signal
signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

# Main Execution
if __name__ == "__main__":
    create_table()
    start_data_streaming(interval=3600)

    # Example analysis call (can be triggered based on certain conditions)
    time.sleep(5)
    analyze_data()

    while True:
        time.sleep(10)
