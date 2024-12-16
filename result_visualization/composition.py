import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Folders with your data
files = [ '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_gauss.csv']


directory = '/Users/shaozishan/Desktop/Data_Plot_BDCD/Composition_plot'



for file in files:
    
    curr_data = pd.read_csv(file)
    
    unique_blksize = curr_data['blksize'].unique()
    unique_blksize = sorted(unique_blksize)
    
    print(unique_blksize)
    
    for blksize in unique_blksize:
        
        data = curr_data[curr_data['blksize'] == blksize]
        # Assuming data is your DataFrame
        
        if "newsbinary" in file: 
            # unique_num_process_values = unique_num_process_values[7:]
            data = data[~data['num_process'].isin(([32]))]  # , 64, 128, 256, 320, 384
            #for elts in [32,64,128,256,320,384]:
            #    data = data[data['num_process'] != elts]

        
        # kernel_type = data['kernel'][0]
        if "linear" in file:
            kernel_type = "linear"
        if "duke" in file:
            kernel_type = "poly"
        if "mnist" in file:
            kernel_type = "gauss"

        # sort the dataset
        data = data.sort_values(by='num_process', ascending=False)
        
        unique_num_process_values = data['num_process'].unique()
        
        print(unique_num_process_values)
        
        # Reverse the order of unique_num_process_values
        unique_num_process_values = sorted(unique_num_process_values, reverse=True)
        
        
        for nump in unique_num_process_values: 
            df = data[(data['num_process'] == nump)]
                    
            # Assuming df is your DataFrame, sorted by 's'
            s_values = df['s'].unique()
            s_values = sorted(s_values, reverse=True)
            x_ticks = np.arange(len(s_values))


            # Plot the stacked bar chart
            #plt.bar(x_ticks, df['csr_setup_time'], label='CSR Setup Time')
            #plt.bar(x_ticks, df['kernel_computation'], bottom=df['csr_setup_time'], label='Kernel Computation')
            #plt.bar(x_ticks, df['kernel_computation'], bottom=df['csr_setup_time'], label='Kernel Computation')
            #plt.bar(x_ticks, df['allreduce_time'], bottom=(df['csr_setup_time'] + df['kernel_computation']), label='Allreduce Time')
            #plt.bar(x_ticks, df['sample_time'], bottom=(df['csr_setup_time'] + df['kernel_computation'] + df['allreduce_time']), label='Sample Time')
            #plt.bar(x_ticks, df['csr_read_time'], bottom=(df['csr_setup_time'] + df['kernel_computation'] + df['allreduce_time'] + df['sample_time']), label='CSR Read Time')
            # sample_time	csr_read_time	gradient_comp_time	alpha_update	memory_reset
            plt.bar(x_ticks, df['sample_time'], label='Sample Time')
            plt.bar(x_ticks, df['kernel_computation'], bottom=df['sample_time'], label='Kernel Computation')
            plt.bar(x_ticks, df['allreduce_time'], bottom=(df['sample_time'] + df['kernel_computation']), label='Allreduce Time')
            plt.bar(x_ticks, df['gradient_comp_time'], bottom=(df['sample_time'] + df['kernel_computation'] + df['allreduce_time']), label='Gradient Computation Time')
            plt.bar(x_ticks, df['alpha_update'], bottom=(df['sample_time'] + df['kernel_computation'] + df['allreduce_time'] + df['gradient_comp_time']), label='Alpha Update Time')
            plt.bar(x_ticks, df['memory_reset'], bottom=(df['sample_time'] + df['kernel_computation'] + df['allreduce_time'] + df['gradient_comp_time'] + df['alpha_update']), label='Memory Reset Time')



            # Add labels and legend
            plt.xlabel('s')
            plt.ylabel('Value')
            plt.title('Time Composition Plot')
            plt.xticks(x_ticks, s_values)  # Set x-tick labels to the values of 's'
            plt.legend()

            # Show the plot
            # plt.show()
            
            if "newsbinary" in file:
                filename = 'Newsbinary_{}_b{}_np{}_Cabdcd_Composition.png'.format(kernel_type,blksize,nump)
            if "duke" in file:
                filename = 'Duke_{}_b{}_np{}_Cabdcd_Composition.png'.format(kernel_type,blksize,nump)
            if "mnist" in file:
                filename = 'MNIST_{}_b{}_np{}_Cabdcd_Composition.png'.format(kernel_type,blksize,nump)
                
            # Use os.path.join to create the full path    
            full_path = os.path.join(directory, filename)

            plt.savefig(full_path)

            # Display the plot
            # plt.show()

            plt.close()

