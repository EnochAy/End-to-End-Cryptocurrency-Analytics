"""
Feature Engineering (for ML)
Create new features that might improve machine learning models.

Steps:

Derive percentage price changes, 7-day moving averages, or volume-to-market-cap ratios.
Normalize or standardize the features
"""


df['price_change_24h'] = df['price'].pct_change()  # Percentage change over 24h



"""
Machine Learning Modeling
Use the prepared data to build and train machine learning models for prediction, classification, or clustering.

Steps:

Predicting Prices: Use regression models (e.g., Linear Regression, Random Forest, etc.) to predict future prices based on historical data.
Clustering: Cluster cryptocurrencies based on their features (e.g., market cap, volume) using algorithms like K-Means.
Example (Simple Linear Regression to predict price):
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['market_cap', 'volume_24h']]  # Features
y = df['price']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
