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
    'X-CMC_PRO_API_KEY': 'b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c'  # Replace with your API key
}

# MySQL Database Configuration
DB_CONFIG = {
    "host": "localhost",          # Your MySQL host (e.g., localhost or IP)
    "user": "root",               # Your MySQL username
    "password": "EnochAy@88",     # Your MySQL password
    "database": "crypto_db"       # Your database name
}

# Connect to MySQL Database
def connect_to_database(config):
    try:
        conn = mysql.connector.connect(**config)
        print("Database connection established.")
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        exit(1)

conn = connect_to_database(DB_CONFIG)
c = conn.cursor()

# Initialize database table
def initialize_database():
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
    print("Database table initialized.")

initialize_database()

# Function to fetch data from CoinMarketCap API
def fetch_data_from_api():
    try:
        response = requests.get(url, headers=headers, params=parameters)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CoinMarketCap API: {e}")
        return None

# Function to store data in the MySQL database
def store_data_in_database(data):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
    for entry in data['data']:
        try:
            c.execute('''INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap, timestamp) 
                         VALUES (%s, %s, %s, %s, %s, %s)''',
                      (entry['name'], entry['symbol'], 
                       entry['quote']['USD']['price'], 
                       entry['quote']['USD']['volume_24h'], 
                       entry['quote']['USD']['market_cap'], 
                       timestamp))
        except Exception as e:
            print(f"Error inserting data for {entry['name']}: {e}")
    conn.commit()
    print(f"Data successfully stored at {timestamp}")

# Function to run the data fetch-and-store process
def fetch_and_store_data():
    print("Fetching data from CoinMarketCap API...")
    data = fetch_data_from_api()
    if data:
        print("Storing data into the database...")
        store_data_in_database(data)

# Real-time data fetching loop (interval: 1 hour)
try:
    print("Starting real-time data fetching...")
    while True:
        fetch_and_store_data()
        print("Sleeping for 1 hour...")
        time.sleep(3600)  # 1-hour interval
except KeyboardInterrupt:
    print("Process interrupted. Closing database connection...")
    conn.close()
    print("Database connection closed.")
