import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Cryptocurrency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load your cleaned data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('cleaned_crypto_data.csv')
        data['last_updated'] = pd.to_datetime(data['last_updated'], errors='coerce')  # Ensure correct datetime parsing
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Sidebar Filters
st.sidebar.title("Filters")
cryptos = df['name'].unique() if not df.empty else []
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", options=cryptos)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['last_updated'].min(), df['last_updated'].max()) if not df.empty else (None, None),
)

# Filter data based on user input
if not df.empty:
    filtered_df = df[(df['name'] == selected_crypto) & 
                     (df['last_updated'] >= pd.to_datetime(date_range[0])) & 
                     (df['last_updated'] <= pd.to_datetime(date_range[1]))]
else:
    filtered_df = pd.DataFrame()

# Main Dashboard
st.title("Cryptocurrency Dashboard")
if df.empty:
    st.warning("No data available to display. Please check your data source.")
else:
    # 1. Display selected cryptocurrency data
    st.subheader(f"Overview: {selected_crypto}")
    st.write(filtered_df[['last_updated', 'price', 'volume_24h', 'market_cap']])

    # 2. Visualize price trends
    st.subheader(f"{selected_crypto} Price Trend")
    if filtered_df.empty:
        st.warning("No data available for the selected cryptocurrency and date range.")
    else:
        st.line_chart(filtered_df.set_index('last_updated')['price'])

    # 3. Visualize market cap and volume trends
    st.subheader("Market Cap and Volume (24h) Trends")
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    sns.lineplot(data=filtered_df, x='last_updated', y='market_cap', ax=ax[0], label='Market Cap', color='blue')
    ax[0].set_title("Market Cap Trend")
    ax[0].set_ylabel("Market Cap (USD)")
    ax[0].tick_params(axis='x', rotation=45)

    sns.lineplot(data=filtered_df, x='last_updated', y='volume_24h', ax=ax[1], label='24h Volume', color='green')
    ax[1].set_title("24h Volume Trend")
    ax[1].set_ylabel("Volume (USD)")
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    # 4. Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(filtered_df[['price', 'volume_24h', 'market_cap']].describe())

    # 5. Correlation heatmap
    st.subheader("Correlation Heatmap")
    if not filtered_df[['price', 'volume_24h', 'market_cap']].empty:
        corr = filtered_df[['price', 'volume_24h', 'market_cap']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Between Features")
        st.pyplot(fig)
    else:
        st.warning("Not enough data for correlation analysis.")

# Run Streamlit: Save this script as `app.py` and run `streamlit run app.py` in your terminal.
#streamlit run app.py
