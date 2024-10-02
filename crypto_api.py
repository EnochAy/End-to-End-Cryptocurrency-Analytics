 #Python 2.7 and the python-request library.
"""
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'5000',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': 'b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c',
}

session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters)
  data = json.loads(response.text)
  print(data)
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)
  
"""



import requests
import json

url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD'
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c',  # Your API key here
}

try:
    # Send GET request to the CoinMarketCap API
    response = requests.get(url, headers=headers, params=parameters)
    
    # Parse response data into JSON format
    data = response.json()
    
    # Pretty-print the JSON response
    print(json.dumps(data, indent=4))
    
except requests.exceptions.RequestException as e:
    # Handle any request exceptions
    print(f"Error: {e}")



"""
import requests
import json
import os

# Fetch API key from environment variable
api_key = os.getenv('CMC_PRO_API_KEY')

if not api_key:
    raise ValueError("API key not found! Set the 'CMC_PRO_API_KEY' environment variable.")

url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD'
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': api_key,  # Using API key from environment
}

try:
    # Send GET request to the CoinMarketCap API
    response = requests.get(url, headers=headers, params=parameters)

    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        try:
            # Parse response data into JSON format
            data = response.json()
            
            # Pretty-print the JSON response
            print(json.dumps(data, indent=4))
        except json.JSONDecodeError:
            print("Error: Unable to parse the response as JSON.")
    else:
        print(f"Error: Received response with status code {response.status_code}")

except requests.exceptions.RequestException as e:
    # Handle any request exceptions
    print(f"Request Error: {e}")

    
command prompt
set CMC_PRO_API_KEY=your_api_key_here

powershell
$env:CMC_PRO_API_KEY="your_api_key_here"
  
For a permanent solution on Windows:
Press Win + R, type sysdm.cpl, and press Enter.
Go to the Advanced tab and click Environment Variables.
Under User variables, click New, and set:
Variable name: CMC_PRO_API_KEY
Variable value: your_api_key_here
Click OK and restart your terminal.
"""