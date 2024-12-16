
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the dataset
# df = pd.read_csv('/path/to/your/csv/file.csv')

# Filter the DataFrame by specific 'num_process' and 's' values
# df = df[df['num_process'].isin([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])]
# df = df[df['s'].isin([0, 2, 4, 8, 16, 32, 128, 256, 512])]

# Create the plot (This is just a placeholder; replace with your actual plotting code)
plt.figure(figsize=(10, 6))

# Assuming 'x_values' and 'y_values' are columns in your DataFrame
# plt.plot(df['x_values'], df['y_values'], marker='o')

# Set the y-axis and x-axis to log scale with base 2
plt.yscale('log', basey=2)
plt.xscale('log', basex=2)

# Show the plot
plt.show()
