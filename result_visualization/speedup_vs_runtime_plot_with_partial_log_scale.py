
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the dataset
df = pd.read_csv('/Users/shaozishan/Desktop/new_outputs/load_balance/prefinal/caksvm_data_mock2_poly.csv')

print(df.describe())

# Filter the DataFrame by specific 'num_process' and 's' values
df = df[df['num_process'].isin([2,4,8,16,32,64,128,256,512,1024])]
df = df[df['s'].isin([0, 2, 4, 8, 16, 32, 64, 128, 256, 512])]

print(df.describe())

# Determine the kernel type and filename based on the original file name
file = 'caksvm_data_mock2_gauss.csv'

# kernel_type = "gauss"
if "linear" in file:
    kernel_type = 'linear'
if "poly" in file:
    kernel_type = 'poly'
if "gauss" in file:
    kernel_type = 'gauss'
    
if "news" in file:
    filename = 'Caksvm_Newsbinary_{}_General.png'.format(kernel_type)
if "duke" in file:
    filename = 'Caksvm_Duke_{}_General.png'.format(kernel_type)
if "colon" in file:
    filename = 'Caksvm_Colon_{}_General.png'.format(kernel_type)
if "leu" in file:
    filename = 'Caksvm_Leu_{}_General.png'.format(kernel_type)
if "rcv" in file:
    filename = 'Caksvm_rcv_{}_General.png'.format(kernel_type)
if "mock" in file:
    filename = 'Caksvm_mock_{}_General.png'.format(kernel_type)


# Initialize data structures to store speedup and minimum runtime
speedup_s_0 = []
min_runtime_s = []
s_values_min_runtime = []

# Get unique number of processes
unique_num_processes = sorted(df['num_process'].unique())

# Calculate speedup and minimum runtime for each unique number of processes
for num_process in unique_num_processes:
    subset_df = df[df['num_process'] == num_process]
    
    # Calculate runtime for s=0 for the given num_process
    runtime_s_0_value = subset_df[subset_df['s'] == 0]['major_time'].iloc[0]
    
    # Find the minimum runtime for the given num_process
    min_runtime = subset_df['major_time'].min()
    
    # Calculate speedup
    speedup = runtime_s_0_value / min_runtime
    
    # Find the 's' value corresponding to the minimum runtime
    s_value_min_runtime = subset_df[subset_df['major_time'] == min_runtime]['s'].iloc[0]
    
    # Append to lists
    speedup_s_0.append(speedup)
    min_runtime_s.append(min_runtime)
    s_values_min_runtime.append(s_value_min_runtime)

# Create the plot with log scale base 2 for only the time part
fig, ax1 = plt.subplots(figsize=(12, 8))

# Adjusted unique_num_processes for bar plot to make bars close to each other
adjusted_num_processes = np.arange(len(unique_num_processes))

# Plot the histogram for speedup
ax1.bar(adjusted_num_processes, speedup_s_0, alpha=0.6, label='Speedup')
ax1.set_xlabel('Number of Processes')
ax1.set_ylabel('Speedup', color='b')
ax1.set_xticks(adjusted_num_processes)
ax1.set_xticklabels(unique_num_processes)

# Annotate speedup values on the bars
for i, txt in enumerate(speedup_s_0):
    ax1.annotate(f'x{{txt:.2f}}'.format(txt=txt), (adjusted_num_processes[i], speedup_s_0[i]), textcoords="offset points", xytext=(0,10), ha='center')
ax1.tick_params(axis='y', labelcolor='b')

# Create another y-axis for the line plots
ax2 = ax1.twinx()
ax2.set_yscale('log', base=2)  # Set y-axis to log scale with base 2 for the time part

# Plot line for minimum runtime
ax2.plot(adjusted_num_processes, min_runtime_s, color='r', marker='o', label='s-step DCD Runtime')

# Plot line for major_time at s=0
major_time_s_0 = [df[(df['num_process'] == num_process) & (df['s'] == 0)]['major_time'].iloc[0] for num_process in unique_num_processes]
ax2.plot(adjusted_num_processes, major_time_s_0, color='g', marker='x', linestyle='--', label='DCD runtime')

ax2.set_ylabel('Time (seconds)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Annotate turning points with the 's' value
for i, (x, y, s_value) in enumerate(zip(adjusted_num_processes, min_runtime_s, s_values_min_runtime)):
    # ax2.annotate(f's={{s_value}}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', weight='bold')
    ax2.annotate(f's={{}}'.format(s_value), (x, y), textcoords="offset points", xytext=(0,-15), ha='center', weight='bold')


# Show the plot
plt.title('Dual Coordinate Descent Strong Scaling | Mock, Poly')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid(True)

# Save or display the plot
# plt.savefig('/path/to/save/plot.png')

# Save the plot to a specific directory
directory = '/Users/shaozishan/Desktop/new_outputs/load_balance/prefinal/general_plot'
full_path = os.path.join(directory, filename)
plt.savefig(full_path)