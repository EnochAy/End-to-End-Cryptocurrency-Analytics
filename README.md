# End-to-End Cryptocurrency Analytics
This project is an end-to-end data analytics pipeline for cryptocurrency data, leveraging the CoinMarketCap API to retrieve real-time cryptocurrency data, process and store it in a database, and provide analytical insights using machine learning models and a data visualization dashboard.

# Table of Contents
1. Project Overview
2. Features
3. Technologies Used
4. Setup Instructions
    - Prerequisites
    - Installation
    - API Setup
    - Database Setup
5. Usage
6. Future Enhancements
7. License

# Project Overview
This project provides a comprehensive analytics pipeline for analyzing and visualizing cryptocurrency market trends. The key components of this pipeline include:

1. Data Collection: Fetches real-time cryptocurrency data from the CoinMarketCap API.
2. Data Storage: Stores both real-time and historical data in a local SQLite database, allowing for longitudinal analysis.
3. Machine Learning: Integrates machine learning models to predict future cryptocurrency prices and market movements.
4. Dashboard: Builds a dynamic dashboard using Plotly Dash. Either Power BI or Tableau will be used later to visualize real-time trends, predictions, and market insights.

# Features
1. Fetches up-to-date cryptocurrency data from CoinMarketCap API.
2. Stores real-time and historical data in a local SQLite database.
3. Implements a data pipeline to ensure continuous data collection and storage.
4. Applies machine learning models for price prediction and market analysis.
5. Provides real-time data visualization and interactive dashboard.
6. Easy-to-configure scripts for users to run in their local environment.

# Technologies Used
1. Languages: Python
2. APIs: CoinMarketCap API
3. Libraries:
    - requests for API communication
    - sqlite3 for database storage
    - scikit-learn for machine learning models
    - plotly or Dash for interactive data visualization
    - pandas for data manipulation
4. Database: SQLite (can be expanded to use PostgreSQL or MySQL later)
5. Visualization: Plotly, Dash, or Power BI/Tableau for dashboard creation

# Setup Instructions
1. Prerequisites
- Make sure you have Python 3.x installed. You’ll also need to install the following Python libraries:
pip install requests sqlite3 pandas scikit-learn plotly dash

# Installation
1. Clone this repository:
git clone https://github.com/EnochAy/End-to-End-Cryptocurrency-Analytics.git
cd End-to-End-Cryptocurrency-Analytics

2. Install the required libraries:
pip install -r requirements.txt

# API Setup
1. Sign up for a CoinMarketCap API Key:
    - Go to the CoinMarketCap Developer Portal.
    - Sign up and generate an API key.
2. Add your API Key to the Script:
Replace the placeholder your_api_key_here in the headers of the Python script with your actual API key.
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': 'your_api_key_here',  # Add your API key here
}

# Database Setup
By default, this project uses SQLite to store the cryptocurrency data. You can modify the database settings to use a more robust database like MySQL or PostgreSQL if needed.

To initialize the SQLite database, no manual setup is required. The Python script automatically creates a table and stores data in the crypto_data.db file.

# Usage
1. Run the Data Pipeline: The following script fetches real-time data from CoinMarketCap and stores it in the SQLite database.
python full_script.py
The script collects data every 10 minutes, storing cryptocurrency data and timestamps for historical tracking.

2. View the Data: The data is stored in an SQLite database (crypto_data.db). You can query it manually using SQLite, or you can visualize it using Python libraries like pandas or dash.

3. Machine Learning: You can extend the pipeline by training machine learning models (e.g., price prediction, trend classification). Modify the ML section of the code and integrate your models.

4. Visualize the Data: If you have Plotly Dash or a visualization tool like Power BI/Tableau, use the stored data to create dynamic dashboards.


# Future Enhancements
1. Historical Data Storage: Implement a more robust data pipeline for storing and retrieving historical cryptocurrency data.
2. Advanced Machine Learning Models: Integrate more complex models (e.g., LSTM, ARIMA) to enhance prediction accuracy.
3. Live Streaming Dashboard: Incorporate streaming data visualization for real-time insights using Plotly Dash or Power BI.
4. Deploy on the Cloud: Host the application and database on cloud platforms like AWS, GCP, or Azure for scalability.