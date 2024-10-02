"""
. Data Cleaning & Preparation
Once stored, clean and prepare the data for analysis and machine learning.

Steps:

Handle missing values, duplicates, or inconsistencies in the data.
Convert date formats if needed and scale the numeric features like prices, volume, etc.
"""

import pandas as pd

# Load data into DataFrame
df = pd.read_csv('crypto_data.csv')

# Clean data: drop duplicates, handle NaN values, etc.
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)



"""
Exploratory Data Analysis (EDA)
Perform EDA to understand the data and discover patterns or trends.

Steps:

Use Matplotlib or Seaborn to create visualizations like price distribution, market cap trends, or volume spikes.
Calculate summary statistics (mean, median, standard deviation).
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogram of crypto prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50)
plt.title('Cryptocurrency Price Distribution')
plt.show()



