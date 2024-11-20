import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data into DataFrame
df = pd.read_csv('crypto_data.csv')

# Data Cleaning & Preparation
# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Drop columns with all missing values
df.dropna(axis=1, how='all', inplace=True)

# Extract relevant information from the 'quote' column
df['quote'] = df['quote'].apply(eval)  # Convert string to dictionary
df['price'] = df['quote'].apply(lambda x: x['USD']['price'] if 'USD' in x else None)
df['volume_24h'] = df['quote'].apply(lambda x: x['USD']['volume_24h'] if 'USD' in x else None)
df['market_cap'] = df['quote'].apply(lambda x: x['USD']['market_cap'] if 'USD' in x else None)

# Convert 'last_updated' and 'date_added' to datetime format
df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Fill missing values in numeric columns with 0 (to avoid NaN issues during analysis)
df[['price', 'volume_24h', 'market_cap']] = df[['price', 'volume_24h', 'market_cap']].fillna(0)

# Remove rows where 'price' is 0 (invalid data or failed fetch)
df = df[df['price'] > 0]

# Summary of the cleaned data
print("Cleaned Data Summary:")
print(df[['price', 'volume_24h', 'market_cap']].describe())

# Exploratory Data Analysis (EDA)

# 1. Price Distribution: Visualizing the distribution of cryptocurrency prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Cryptocurrency Price Distribution')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()

# 2. Market Cap Trend Over Time: Visualizing market cap trends over time
plt.figure(figsize=(12, 6))
df_sorted = df.sort_values(by='last_updated')
sns.lineplot(data=df_sorted, x='last_updated', y='market_cap', marker='o')
plt.title('Market Cap Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Market Cap (USD)')
plt.xticks(rotation=45)
plt.show()

# 3. Volume Spikes Over Time: Visualizing 24h volume spikes
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_sorted, x='last_updated', y='volume_24h', color='red', marker='o')
plt.title('24h Volume Spikes Over Time')
plt.xlabel('Date')
plt.ylabel('Volume (USD)')
plt.xticks(rotation=45)
plt.show()

# 4. Correlation Heatmap: Analyzing the correlation between numeric features
plt.figure(figsize=(8, 6))
corr = df[['price', 'volume_24h', 'market_cap']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# Optional: Save cleaned data for further analysis or use
df.to_csv('cleaned_crypto_data.csv', index=False)
