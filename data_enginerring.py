"""
1. Data Storage (Data Engineering)
First, store the scraped data in a structured format. This will be useful for analysis, modeling, 
and future reference.

Steps:

Option 1: Store as a CSV or JSON File Save the fetched data locally as a CSV or JSON file. For example:
"""

import csv

# Save as CSV
with open('crypto_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(data['data'][0].keys())  # Headers
    for entry in data['data']:
        writer.writerow(entry.values())




"""
Option 2: Store in a Database (MySQL, PostgreSQL, etc.)

Create a database table to store the cryptocurrency data.
Use libraries like SQLAlchemy or PyMySQL to insert the data into the database.
Example for storing in SQLite:
"""
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
