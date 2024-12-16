import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as mticker

# Folders with your data 
files = ['/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_colon_linear.csv',
         '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_colon_gauss.csv',
         '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_colon_poly.csv',
         '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_duke_linear.csv',
         '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_duke_gauss.csv',
         '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_duke_poly.csv',
         '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_newsbinary_linear.csv',
         '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_newsbinary_gauss.csv',
         '/Users/shaozishan/Desktop/new_outputs/final_plots/Data/para_ca_data_newsbinary_poly.csv']
# Additional files can be added as needed


# Set global parameters for font size
plt.rcParams.update({'font.size': 48, 'pdf.fonttype': 42})

for file in files:
    curr_data = pd.read_csv(file)
    unique_blksize = sorted(curr_data['blksize'].unique())
    
    if 'duke' in file:
        directory = '/Users/shaozishan/Desktop/new_outputs/final_plots/HPCAsia/Ridge/composition_plot/duke'
    elif 'mock' in file:
        directory = '/Users/shaozishan/Desktop/new_outputs/final_plots/HPCAsia/Ridge/composition_plot/mock'
    elif 'newsbinary' in file:
        directory = '/Users/shaozishan/Desktop/new_outputs/final_plots/HPCAsia/Ridge/composition_plot/newsbinary'
    elif 'colon' in file:
        directory = '/Users/shaozishan/Desktop/new_outputs/final_plots/HPCAsia/Ridge/composition_plot/colon'
    else:
        directory = '/Users/shaozishan/Desktop/new_outputs/final_plots/HPCAsia/Ridge/composition_plot'
    
    for blksize in unique_blksize:
        data = curr_data[curr_data['blksize'] == blksize]
        data = data[data['num_process'].isin([2,4,8,16,32,64,128,256,512,1024,2048,4096])]

        kernel_type = 'linear' if 'linear' in file else 'poly' if 'poly' in file else 'gauss' if 'gauss' in file else ''
        data_type = 'Duke' if 'duke' in file else 'Colon' if 'colon' in file else 'Synthetic' if 'mock' in file else 'Newsbinary' if 'newsbinary' in file else ''

        # sort the dataset
        data = data.sort_values(by='num_process', ascending=False)
        unique_num_process_values = sorted(data['num_process'].unique(), reverse=True)
        
        for nump in unique_num_process_values: 
            df = data[data['num_process'] == nump].drop_duplicates(subset='s', keep='first').sort_values(by='s', ascending=True)
            s_values = sorted(df['s'].unique())
            s_values[0] = 'DCD'  # Assuming the first value should be labeled 'DCD'
            x_ticks = np.arange(len(s_values))
            
            # Set the figure size here
            plt.figure(figsize=(20, 16))  # Width, Height in inches

            # Plot the stacked bar chart
            plt.bar(x_ticks, df['sample_time'], label='Sample Time')
            plt.bar(x_ticks, df['max_kernel_computation'], bottom=df['sample_time'], label='Kernel Computation')
            plt.bar(x_ticks, df['allreduce_time'], bottom=(df['sample_time'] + df['max_kernel_computation']), label='Allreduce Time')
            plt.bar(x_ticks, df['gradient_comp_time'], bottom=(df['sample_time'] + df['max_kernel_computation'] + df['allreduce_time']), label='Gradient Computation Time')
            plt.bar(x_ticks, df['alpha_update'], bottom=(df['sample_time'] + df['max_kernel_computation'] + df['allreduce_time'] + df['gradient_comp_time']), label='Alpha Update Time')
            plt.bar(x_ticks, df['memory_reset'], bottom=(df['sample_time'] + df['max_kernel_computation'] + df['allreduce_time'] + df['gradient_comp_time'] + df['alpha_update']), label='Memory Reset Time')

            # Add labels and legend
            plt.xlabel('s')
            plt.ylabel('Running Time (sec.)')
            plt.xticks(x_ticks, s_values)  # Set x-tick labels to the values of 's'
            plt.legend(fontsize = 40)
            
            # Format y-axis to two significant digits
            #plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))
            # Force y-axis to always use scientific notation
            plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Force scientific format
            plt.gca().yaxis.get_offset_text().set_fontsize(40)  # Adjust font size of scientific notation
            
            # Save the plot as a PDF with 300 DPI
            #filename = f'Synthetic_{kernel_type}_np{nump}_caksvm_Composition.pdf'
            filename = f'{data_type}_{kernel_type}_np{nump}_b{blksize}_Composition.pdf'
            full_path = os.path.join(directory, filename)
            os.makedirs(directory, exist_ok=True)  # Ensure directory exists
            plt.savefig(full_path, format='pdf', dpi=300)

            plt.close()
