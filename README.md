# End-to-End Cryptocurrency Analytics

## Problem Statement
Cryptocurrency markets are highly volatile and complex, making it challenging for investors and traders to make informed decisions. This project aims to solve the problem of accessing actionable insights by providing a pipeline that collects real-time market data, stores it for historical analysis, applies machine learning for price predictions, and visualizes trends through an interactive dashboard. The goal is to empower users with data-driven insights to navigate the crypto market effectively.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Architecture](#architecture)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [API Setup](#api-setup)
    - [Database Setup](#database-setup)
7. [Usage](#usage)
8. [Future Enhancements](#future-enhancements)
9. [License](#license)

## Project Overview
This project builds an end-to-end analytics pipeline for cryptocurrency data, fetching real-time data from the CoinMarketCap API, storing it in a MySQL database, performing feature engineering, training machine learning models (ARIMA and XGBoost) for price prediction, and visualizing insights through a Plotly Dash dashboard.

## Features
- Fetches real-time cryptocurrency data from CoinMarketCap API.
- Stores data in a MySQL database for historical and real-time analysis.
- Implements feature engineering for advanced analytics (e.g., price change, moving averages).
- Trains ARIMA and XGBoost models for price prediction.
- Provides an interactive dashboard with real-time price trends and predictions.
- Logs operations for debugging and monitoring.

## Technologies Used
- **Python**: General-purpose language for data processing and scripting.
- **CoinMarketCap API**: Provides reliable, real-time cryptocurrency market data.
- **MySQL**: Structured database for efficient storage and querying of time-series data.
- **Pandas**: Handles data manipulation and feature engineering.
- **Scikit-learn, Statsmodels, XGBoost**: Enable machine learning for price prediction.
- **Plotly Dash**: Creates an interactive, web-based dashboard for visualization in pure Python—ideal for quickly deploying ML results visually.
- **Logging**: Tracks operations for debugging and monitoring.
- **Dotenv**: Secures sensitive credentials (API key, database password).
- **ARIMA/XGBoost**: Combines traditional time series with modern ML for robust short-term crypto price forecasting.

## Architecture
```
[CoinMarketCap API] --> [Data Ingestion: fetch_data.py]
                                  |
                                  v
[MySQL Database: crypto_db] <-- [Database: database.py]
                                  |
                                  v
[Feature Engineering: transform_data.py]
                                  |
                                  v
[Model Training: train_models.py]
                                  |
                                  v
[Dash Dashboard: dashboard.py]
```

## Project Structure
```
End-to-End-Cryptocurrency-Analytics/
├── data_ingestion/
│   └── fetch_data.py           # Fetches and stores API data
├── data_transformation/
│   └── transform_data.py       # Feature engineering
├── modeling/
│   └── train_models.py         # ARIMA and XGBoost training
├── visualization/
│   └── dashboard.py            # Dash dashboard
├── utils/
│   └── database.py             # Database initialization
├── models/
│   ├── arima_model.pkl         # Saved ARIMA model
│   └── xgboost_model.pkl       # Saved XGBoost model
├── .env                        # Environment variables (not tracked)
├── .gitignore                  # Git ignore file
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
└── main.py                     # Orchestrates pipeline
```

## Setup Instructions

### Prerequisites
- Python 3.10+ (verified with your `python --version` output: 3.10.4).
- MySQL Server installed and running.
- CoinMarketCap API key (sign up at [CoinMarketCap Developer Portal](https://coinmarketcap.com/api/)).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/EnochAy/End-to-End-Cryptocurrency-Analytics.git
   cd End-to-End-Cryptocurrency-Analytics
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### API Setup
1. Sign up for a CoinMarketCap API key at [CoinMarketCap Developer Portal](https://coinmarketcap.com/api/).
2. Create a `.env` file in the project root:
   ```plaintext
   CMC_API_KEY=your_api_key_here
   DB_PASSWORD=your_mysql_password_here
   ```
3. Ensure the `.env` file is not committed (covered by `.gitignore`).

### Database Setup
1. Install MySQL if not already installed (download from [mysql.com](https://dev.mysql.com/downloads/installer/)).
2. Create a database named `crypto_db`:
   ```sql
   CREATE DATABASE crypto_db;
   ```
3. The `main.py` script automatically initializes the `crypto_data` table.

## Usage
1. Run the main script to start the pipeline and dashboard:
   ```bash
   python main.py
   ```
2. Access the dashboard at `http://localhost:8050`.
3. View logs in `crypto_analytics.log` for debugging.

## Future Enhancements
- Add historical data storage with a robust ETL pipeline.
- Integrate advanced models like LSTM for better predictions.
- Deploy the dashboard on a cloud platform (e.g., AWS, GCP).
- Expand visualization options with Power BI or Tableau.

## License
MIT License