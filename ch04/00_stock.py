import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file into a pandas DataFrame
file_path = './portfolio_data.csv'
df = pd.read_csv(file_path, parse_dates=['Date'])

# Set Date as the index
df.set_index('Date', inplace=True)

# Plotting the data from the CSV file in separate graphs
fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

# Plot for AMZN
axes[0].plot(df.index, df['AMZN'], label='AMZN', marker='o', color='blue')
axes[0].set_ylabel('AMZN Value')
axes[0].legend()

# Plot for DPZ
axes[1].plot(df.index, df['DPZ'], label='DPZ', marker='o', color='green')
axes[1].set_ylabel('DPZ Value')
axes[1].legend()

# Plot for BTC
axes[2].plot(df.index, df['BTC'], label='BTC', marker='o', color='orange')
axes[2].set_ylabel('BTC Value')
axes[2].legend()

# Plot for NFLX
axes[3].plot(df.index, df['NFLX'], label='NFLX', marker='o', color='red')
axes[3].set_ylabel('NFLX Value')
axes[3].set_xlabel('Date')
axes[3].legend()

# Adjust layout for better appearance
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plots
plt.show()
