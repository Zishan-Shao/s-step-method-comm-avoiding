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


directory = '/Users/shaozishan/Desktop/Data_Plot_BDCD/general_plot'



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
        df = data.sort_values(by='num_process', ascending=False)
        
        
        unique_num_process_values = df['num_process'].unique()
        
        print(unique_num_process_values)
        
        # Reverse the order of unique_num_process_values
        unique_num_process_values = sorted(unique_num_process_values, reverse=True)
        
        
        #data_test = data['filename'][0]

        # data = pd.read_csv("/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_mnist_linear.csv")

        # df = data

        # Calculating the average speeding-up and smallest runtime
        avg_speedup = []
        runtime_s_0 = []
        min_runtime_s = []

        print(type(min_runtime_s))
        '''
        for num_process in df['num_process'].unique():
            subset_df = df[df['num_process'] == num_process]
            runtime_s_0_value = subset_df[subset_df['s'] == 0]['major_time'].mean()
            # runtime_s_0.append(runtime_s_0_value)
            runtime_s_0 = np.concatenate(([runtime_s_0_value], runtime_s_0))
            min_runtime = subset_df['major_time'].min()
            min_runtime_s.append(min_runtime)
            speedup = runtime_s_0_value / min_runtime
            avg_speedup.append(speedup)
        '''

        for num_process in df['num_process'].unique():
            subset_df = df[df['num_process'] == num_process]
            runtime_s_0_value = subset_df[subset_df['s'] == 0]['major_time'].min()
            
            # Inserting at the beginning of the list to reverse the order
            runtime_s_0.insert(0, runtime_s_0_value)
            
            min_runtime = subset_df['major_time'].min()
            # min_runtime_s.insert(0, min_runtime) # Inserting here as well
            min_runtime_s.append(min_runtime)
            
            speedup = runtime_s_0_value / min_runtime
            avg_speedup.insert(0, speedup) # Inserting here as well




        # ... (Same as above, including DataFrame and calculations)


        # Calculating the 's' value corresponding to the smallest runtime
        s_values_min_runtime = [df[(df['num_process'] == num_process) & (df['major_time'] == min_runtime)]['s'].iloc[0]
                                for num_process, min_runtime in zip(df['num_process'].unique(), min_runtime_s)]

        reversed = []

        for elts in s_values_min_runtime:
            reversed.insert(0,elts)



        # Getting unique num_processes values
        unique_num_processes = sorted(df['num_process'].unique())

        # Plotting the bar plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(range(len(unique_num_processes)), avg_speedup, color='cyan', edgecolor='blue', width=1)
        ax1.set_xlabel('Number of Process')
        ax1.set_ylabel('Average Speeding-up')
        ax1.yaxis.label.set_color('blue')
        ax1.tick_params(axis='y', colors='blue')
        # ax1.set_title('Bar & Line Plot of Average Speeding-up & Runtime')
        plt.xticks(range(len(unique_num_processes)), unique_num_processes) # Setting x-axis ticks to unique values

        # Creating a second y-axis for the line plot
        ax2 = ax1.twinx()
        ax2.set_ylabel('Runtime in Seconds')

        # Line plots
        line1, = ax2.plot(range(len(unique_num_processes)), runtime_s_0, 'g-', label='Runtime at s=0', linewidth=2)
        line2, = ax2.plot(range(len(unique_num_processes)), min_runtime_s, 'r-', label='Smallest Runtime at any s', linewidth=2)

        # Adding s-value annotations  # s_values_min_runtime
        for i, y, s_value in zip(range(len(unique_num_processes)), min_runtime_s, reversed):
            plt.annotate(f's={s_value}', (i, y), textcoords="offset points", xytext=(0,10), ha='center')

        # Adding a legend
        ax2.legend(handles=[line1, line2], loc='upper right', title='Lines', bbox_to_anchor=(1.05, 1))

        # Saving the plot
        # plt.savefig('/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/general_mnist_linear.png')
        
        # Add a global title
        # plt.suptitle('s vs Major time | blksize = 1, {}'.format(kernel_type))

        # Save the figure before displaying
        #filename = '{}_s_vs_major_time.png'.format(kernel_type)
        if "newsbinary" in file:
            filename = 'Cabdcd_Newsbinary_{}_b{}_General.png'.format(kernel_type,blksize)
        if "duke" in file:
            filename = 'Cabdcd_Duke_{}_b{}_General.png'.format(kernel_type,blksize)
        if "mnist" in file:
            filename = 'Cabdcd_MNIST_{}_b{}_General.png'.format(kernel_type,blksize)
            
        # Use os.path.join to create the full path
        full_path = os.path.join(directory, filename)

        plt.savefig(full_path)

        # Display the plot
        # plt.show()

        plt.close()

        
        # plt.show()
