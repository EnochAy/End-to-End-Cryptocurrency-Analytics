"""
. Data Visualization & Dashboard (Streaming or Static)
Create a real-time or static dashboard to visualize the cryptocurrency trends, prices, or predictions.

Tools:

Streamlit or Dash can be used for building interactive dashboards.
Tableau or Power BI can be used for creating static or streaming dashboards.
Example using Streamlit:
 """

pip install streamlit #bash code


#In your script (app.py):


import streamlit as st
import pandas as pd

# Load your cleaned data
df = pd.read_csv('crypto_data.csv')

# Create Streamlit dashboard
st.title('Cryptocurrency Dashboard')

# Display data table
st.write(df)

# Plot price trend
st.line_chart(df['price'])


#Run the dashboard with:
streamlit run app.py
