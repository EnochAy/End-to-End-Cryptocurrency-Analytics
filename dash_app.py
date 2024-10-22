"""
Step 2: Visualization Dashboard (using Plotly Dash)
Next, I'll use Plotly Dash to create an interactive dashboard for visualizing cryptocurrency data. 
This dashboard will show real-time price trends and historical data.

Install dash, dash-bootstrap-components, and plotly via pip:
"""
#1. pip install dash dash-bootstrap-components plotly
#pip install dash dash-bootstrap

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import mysql.connector
import dash_bootstrap_components as dbc  # Ensure you import dbc for the Bootstrap theme

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
    dcc.Interval(id='interval-component', interval=600 * 1000, n_intervals=0)  # Refresh every 10 minutes
])

# Update graph based on selected cryptocurrency and refresh interval
@app.callback(
    Output('price-graph', 'figure'),
    [Input('crypto-dropdown', 'value'), Input('interval-component', 'n_intervals')]
)
def update_graph(selected_crypto, n):
    df = load_data()
    df = df[df['name'] == selected_crypto]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'],
                             mode='lines+markers', name='Price'))
    
    fig.update_layout(title=f'Price of {selected_crypto} Over Time', xaxis_title='Time', yaxis_title='Price (USD)')
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
