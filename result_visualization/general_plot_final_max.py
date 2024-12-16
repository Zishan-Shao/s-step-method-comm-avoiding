import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

## All dataset in this directory is fine: '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/'

# Set file path
file_path = '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/caksvm_data_newsbinary_linear_barrier.csv'

# Extract the filename (without extension) from the file path
file_name = os.path.basename(file_path).split('.')[0]

# Load the dataset into a variable named 'df'
df = pd.read_csv(file_path)

# Filter the DataFrame by specific 'num_process' and 's' values
df = df[df['num_process'].isin([2,4,8,16,32,64,128,256,512, 1024, 2048, 4096])]
df = df[df['s'].isin([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])]

print(df.describe())

# Determine the kernel type and filename based on the original file name
file = file_name
DName = ''

if "linear" in file:
    kernel_type = 'linear'
elif "poly" in file:
    kernel_type = 'poly'
elif "gauss" in file:
    kernel_type = 'gauss'
    
if "news" in file:
    filename = f'Caksvm_Newsbinary_{kernel_type}_Max.pdf'
    DName = "news"
elif "duke" in file:
    filename = f'Caksvm_Duke_{kernel_type}_Max.pdf'
    DName = "duke"
elif "colon" in file:
    filename = f'Caksvm_Colon_{kernel_type}_Max.pdf'
    DName = "colon"
elif "leu" in file:
    filename = f'Caksvm_Leu_{kernel_type}_Max.pdf'
    DName = "leu"
elif "rcv" in file:
    filename = f'Caksvm_rcv_{kernel_type}_Max.pdf'
    DName = "rcv"
elif "mock" in file:
    filename = f'Caksvm_synthetic_{kernel_type}_Max.pdf'
    DName = "mock"
elif "improved" in file:
    filename = f'Caksvm_synthetic_{kernel_type}_Max_Improved.pdf'
    DName = "improved"

# Initialize data structures to store speedup and minimum runtime
speedup_s_0 = []
min_runtime_s = []
s_values_min_runtime = []

# Get unique number of processes
unique_num_processes = sorted(df['num_process'].unique())

# Calculate speedup and minimum runtime for each unique number of processes
for num_process in unique_num_processes:
    subset_df = df[df['num_process'] == num_process]
    
    runtime_s_0_value = subset_df[subset_df['s'] == 0]['max_major_time'].iloc[0]
    min_runtime = subset_df['max_major_time'].min()
    speedup = runtime_s_0_value / min_runtime
    s_value_min_runtime = subset_df[subset_df['max_major_time'] == min_runtime]['s'].iloc[0]
    
    speedup_s_0.append(speedup)
    min_runtime_s.append(min_runtime)
    s_values_min_runtime.append(s_value_min_runtime)

# Create the plot with log scale base 2 for only the time part
plt.rcParams.update({'font.size': 48}) # Set font size for all plot elements
fig, ax1 = plt.subplots(figsize=(24, 15))  # Adjust the width and height as needed

adjusted_num_processes = np.arange(len(unique_num_processes))

# Plot the histogram for speedup
ax1.bar(adjusted_num_processes, speedup_s_0, alpha=0.6, label='Speedup' if not (kernel_type == 'gauss' and 'duke' in file) else '')
ax1.set_xlabel('Number of Processes')
ax1.set_ylabel('Speedup', color='b')
ax1.set_xticks(adjusted_num_processes)
ax1.set_xticklabels(unique_num_processes)

for i, txt in enumerate(speedup_s_0):
    ax1.annotate(f'{txt:.2f}x', (adjusted_num_processes[i], speedup_s_0[i]), textcoords="offset points", xytext=(0,10), ha='center')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.set_yscale('log', base=2)

max_major_time_s_0 = [df[(df['num_process'] == num_process) & (df['s'] == 0)]['max_major_time'].iloc[0] for num_process in unique_num_processes]
ax2.plot(adjusted_num_processes, min_runtime_s, color='r', marker='o', label='s-step DCD Runtime', linewidth=3)
ax2.plot(adjusted_num_processes, max_major_time_s_0, color='g', marker='x', linestyle='--', label='DCD runtime', linewidth=3)

ax2.set_ylabel('Time (seconds)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

for i, (x, y, s_value) in enumerate(zip(adjusted_num_processes, min_runtime_s, s_values_min_runtime)):
    ax2.annotate(f's={s_value}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', weight='bold')

# Set title with conditional check for 'mock' in file name
#plot_title = 'DCD Strong Scaling (Max) | Duke, Poly' if 'mock' in file else 'DCD Strong Scaling (Max)'
# Create the plot title
plot_title = f'DCD Strong Scaling (Max) | {DName}, {kernel_type}'
plt.title(plot_title)
#if not (kernel_type == 'gauss' and 'duke' in file):
#    ax1.legend(loc='upper left')
# ax2.legend(loc='lower left')
plt.grid(True)

# Save the plot to a specific directory
directory = '/Users/shaozishan/Desktop/new_outputs/final_plots/HPCAsia/SVM/general_plot'
full_path = os.path.join(directory, filename)
plt.savefig(full_path, dpi=300, format='pdf')
