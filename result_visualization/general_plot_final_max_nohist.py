import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the dataset
df = pd.read_csv('/Users/shaozishan/Desktop/new_outputs/final_plots/Data/caksvm_data_duke_poly_barrier.csv')

print(df.describe())

# Filter the DataFrame by specific 'num_process' and 's' values
df = df[df['num_process'].isin([2,4,8,16,32,64,128,256,512])]
df = df[df['s'].isin([0, 2, 4, 8, 16, 32, 64, 128, 256, 512])]

print(df.describe())

# Determine the kernel type and filename based on the original file name
file = 'caksvm_data_duke_poly.csv'


if "linear" in file:
    kernel_type = 'linear'
elif "poly" in file:
    kernel_type = 'poly'
elif "gauss" in file:
    kernel_type = 'gauss'
    
if "news" in file:
    filename = f'Caksvm_Newsbinary_{kernel_type}_Max.pdf'
elif "duke" in file:
    filename = f'Caksvm_Duke_{kernel_type}_Max.pdf'
elif "colon" in file:
    filename = f'Caksvm_Colon_{kernel_type}_Max.pdf'
elif "leu" in file:
    filename = f'Caksvm_Leu_{kernel_type}_Max.pdf'
elif "rcv" in file:
    filename = f'Caksvm_rcv_{kernel_type}_Max.pdf'
elif "mock" in file:
    filename = f'Caksvm_synthetic_{kernel_type}_Max.pdf'
elif "improved" in file:
    filename = f'Caksvm_synthetic_{kernel_type}_Max_Improved.pdf'

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

# Plotting adjustments for significance in turning points
plt.rcParams.update({'font.size': 64})  # Adjust font size for all plot elements
fig, ax2 = plt.subplots(figsize=(20, 15))  # Adjust figure size as needed

adjusted_num_processes = np.arange(len(unique_num_processes))
ax2.set_yscale('log', base=2)

# Plot CA-DCD (s-step DCD) Runtime with circles at each turning point
max_major_time_s_0 = [df[(df['num_process'] == num_process) & (df['s'] == 0)]['max_major_time'].iloc[0] for num_process in unique_num_processes]
ax2.plot(adjusted_num_processes, min_runtime_s, color='r', marker='o', linestyle='-', label='s-step DCD Runtime', markersize=24, linewidth=3)  # circles for CA-DCD

# Plot DCD runtime with squares at each turning point
ax2.plot(adjusted_num_processes, max_major_time_s_0, color='g', marker='s', linestyle='--', label='DCD runtime', markersize=18, linewidth=3)  # squares for DCD

ax2.set_xlabel('Number of Processes')
ax2.set_ylabel('Time (seconds)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_xticks(adjusted_num_processes)
ax2.set_xticklabels(unique_num_processes)

for i, (x, y, s_value) in enumerate(zip(adjusted_num_processes, min_runtime_s, s_values_min_runtime)):
    ax2.annotate(f's={s_value}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', weight='bold')

plt.title('DCD Strong Scaling (Max) | Duke Poly')
plt.grid(True)

# Save the plot
directory = '/Users/shaozishan/Desktop/new_outputs/final_plots/HPCAsia'
full_path = os.path.join(directory, filename)
plt.savefig(full_path, dpi=300, format='pdf')