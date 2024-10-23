"""
1. Data Storage (Data Engineering)
First, store the scraped data in a structured format. This will be useful for analysis, modeling, 
and future reference.

Steps:

Option 1: Store as a CSV or JSON File Save the fetched data locally as a CSV or JSON file.



import csv

# Save as CSV
with open('crypto_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(data['data'][0].keys())  # Headers
    for entry in data['data']:
        writer.writerow(entry.values())



Option 2: Store in a Database (MySQL, PostgreSQL, etc.)

Create a database table to store the cryptocurrency data into MySQL to insert the data into the database.

For storing in MySQL:
"""

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
    time.sleep(3600)  # Sleep for 1 hour
