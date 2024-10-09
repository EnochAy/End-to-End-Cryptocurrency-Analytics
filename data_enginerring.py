"""
1. Data Storage (Data Engineering)
First, store the scraped data in a structured format. This will be useful for analysis, modeling, 
and future reference.

Steps:

Option 1: Store as a CSV or JSON File Save the fetched data locally as a CSV or JSON file. For example:
"""


"""
import csv

# Save as CSV
with open('crypto_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(data['data'][0].keys())  # Headers
    for entry in data['data']:
        writer.writerow(entry.values())



Option 2: Store in a Database (MySQL, PostgreSQL, etc.)

Create a database table to store the cryptocurrency data.
Use libraries like SQLAlchemy or PyMySQL to insert the data into the database.
Example for storing in SQLite:

import sqlite3

conn = sqlite3.connect('crypto_data.db')
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS crypto_data (
    id INTEGER PRIMARY KEY,
    name TEXT,
    symbol TEXT,
    price REAL,
    volume_24h REAL,
    market_cap REAL
)''')


# Insert data into table
for entry in data['data']:
    c.execute('INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap) VALUES (?, ?, ?, ?, ?)',
              (entry['name'], entry['symbol'], entry['quote']['USD']['price'], 
               entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap']))

conn.commit()
conn.close()


"""





import requests
import json
import csv
import sqlite3

# Step 1: Fetch data from CoinMarketCap API
url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD'
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'your_api_key_here',  # Replace with your API key
}

try:
    # Send GET request to the CoinMarketCap API
    response = requests.get(url, headers=headers, params=parameters)
    
    # Parse response data into JSON format
    data = response.json()
    
    # Step 2: Store the data in a CSV file
    filename = 'crypto_data.csv'
    try:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data['data'][0].keys())  # Write the header (column names)

            for entry in data['data']:
                writer.writerow(entry.values())

        print(f"Data successfully saved to {filename}")

    except Exception as e:
        print(f"Error occurred while saving data to CSV: {e}")

    # Step 3: Store the data in SQLite Database
    try:
        conn = sqlite3.connect('crypto_data.db')  # Establish connection to SQLite database
        c = conn.cursor()

        # Create table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS crypto_data (
            id INTEGER PRIMARY KEY,
            name TEXT,
            symbol TEXT,
            price REAL,
            volume_24h REAL,
            market_cap REAL
        )''')

        # Insert data into the table
        for entry in data['data']:
            c.execute('INSERT INTO crypto_data (name, symbol, price, volume_24h, market_cap) VALUES (?, ?, ?, ?, ?)',
                      (entry['name'], entry['symbol'], entry['quote']['USD']['price'], 
                       entry['quote']['USD']['volume_24h'], entry['quote']['USD']['market_cap']))

        # Commit transaction
        conn.commit()
        print("Data successfully stored in the SQLite database.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    finally:
        if conn:
            conn.close()  # Ensure connection is closed after use

except requests.exceptions.RequestException as e:
    print(f"Error fetching data from CoinMarketCap API: {e}")
