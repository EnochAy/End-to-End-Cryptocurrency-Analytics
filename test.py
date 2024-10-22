import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc

# Connect to MySQL and load data
def load_data():
    conn = mysql.connector.connect(
        host='localhost',
        user='host',  # Replace with your MySQL username
        password='EnochAy@88',  # Replace with your MySQL password
        database='crypto_db'  # Replace with your MySQL database name
    )
    query = "SELECT * FROM crypto_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Feature Engineering
def feature_engineering(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create new features for machine learning
    df['price_change_24h'] = df['price'].pct_change()  # Percentage price change over 24 hours
    df['volume_market_cap_ratio'] = df['volume_24h'] / df['market_cap']  # Volume to Market Cap ratio
    
    # 7-day moving average for price
    df['price_ma_7d'] = df['price'].rolling(window=7).mean()
    
    # Fill NaN values
    df.fillna(0, inplace=True)
    
    return df

# Prepare data for ML
def prepare_data(df):
    # Feature and target variable selection
    df['time_since'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()  # Time as numerical value
    X = df[['time_since', 'market_cap', 'volume_market_cap_ratio', 'price_change_24h']]  # Feature columns
    y = df['price']  # Target column
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    html.H1("Cryptocurrency Dashboard"),
    dcc.Dropdown(id='crypto-dropdown',
                 options=[
                     {'label': 'Bitcoin', 'value': 'Bitcoin'},
                     {'label': 'Ethereum', 'value': 'Ethereum'}
                 ],
                 value='Bitcoin'),
    dcc.Graph(id='price-graph'),
    dcc.Interval(id='interval-component', interval=600 * 1000, n_intervals=0),  # Refresh every 10 minutes
    html.Div(id='prediction-output', style={'fontSize': 24, 'marginTop': 20})  # Predicted price output
])

# Train Linear Regression model on data
@app.callback(
    Output('price-graph', 'figure'),
    [Input('crypto-dropdown', 'value'), Input('interval-component', 'n_intervals')]
)
def update_graph(selected_crypto, n):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    df = feature_engineering(df)
    
    # Prepare data for plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines+markers', name='Price'))
    
    fig.update_layout(title=f'Price of {selected_crypto} Over Time', xaxis_title='Time', yaxis_title='Price (USD)')
    
    return fig

# Machine Learning Prediction: Predict future prices
@app.callback(
    Output('prediction-output', 'children'),
    [Input('crypto-dropdown', 'value')]
)
def predict_price(selected_crypto):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    df = feature_engineering(df)
    
    # Prepare data for model
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predict future price
    last_time = (pd.to_datetime(df['timestamp'].max()) - df['timestamp'].min()).total_seconds()
    future_time = last_time + 3600  # Predict one hour into the future
    future_time_scaled = scaler.transform(np.array([[future_time, df['market_cap'].iloc[-1], 
                                                     df['volume_market_cap_ratio'].iloc[-1], 
                                                     df['price_change_24h'].iloc[-1]]]))
    
    predicted_price = model.predict(future_time_scaled)[0]
    
    return f"Predicted price for {selected_crypto} in 1 hour: ${predicted_price:.2f}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
