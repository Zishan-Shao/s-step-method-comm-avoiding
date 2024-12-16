# for loop to save the figures
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math


# Folders with your data
files = [ '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_poly.csv']


directories = ['/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_duke_linear',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_mnist_linear',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_newsbinary_linear',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_duke_poly',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_mnist_poly',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_newsbinary_poly']



# data = pd.read_csv("/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_linear.csv")

#for file in files: 
for file, directory in zip(files, directories):
    
    data = pd.read_csv(file)
    
    # Assuming data is your DataFrame
    unique_num_process_values = data['num_process'].unique()
    
    
    # Reverse the order of unique_num_process_values
    unique_num_process_values = sorted(unique_num_process_values, reverse=True)
    
    
    if "newsbinary" in directory:
        unique_num_process_values = unique_num_process_values[:-4]
    
    
    kernel_type = data['kernel'][0]
    #data_test = data['filename'][0]
    
    
    # get the unique number of blksize (in case of CABDCD)
    unique_blksize_values = data['blksize'].unique()

    # Specify your directory
    # directory = "/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/para_ca_newsbinary_linear"
    
    
    for blksize in unique_blksize_values: 
        
        # Get the number of unique num_process values
        num_columns = len(unique_num_process_values)

        # Define the number of rows
        num_rows = 3
        num_cols = math.ceil(num_columns / num_rows)
        
        # Determine the width of each individual subplot
        subplot_width = 7  # You can adjust this value

        # Determine the total figure width based on the number of columns
        fig_width = subplot_width * num_cols

        # Determine the figure height
        fig_height = 14  # You can adjust this value

        # Create a figure with subplots organized in two rows, without shared x and y axes
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))


        # Create a figure with subplots organized in two rows
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10), sharex=True, sharey=True)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=False)

        # Loop through the unique num_process values
        for i, num_process in enumerate(unique_num_process_values):
            df = data[(data['num_process'] == num_process) & (data['blksize'] == blksize)]
            df = df[0:7]

            # Sort s and major_time in ascending order of s
            sorted_indices = df['s'].argsort()
            s = df['s'].values[sorted_indices]
            major_time = df['allreduce_time'].values[sorted_indices]

            # Create a new variable for plotting, with equally spaced values
            s_plot = np.arange(len(s))

            # Get the current axis based on the index i
            ax = axes[i // num_cols, i % num_cols]

            # Plot on the current axis
            ax.plot(s_plot, major_time, marker='o')

            # Set the labels and title
            ax.set_xlabel('s')
            ax.set_ylabel('Allreduce time (s)')
            title_str = 'np = {}'.format(num_process)
            ax.set_title(title_str)
            ax.set_xticks(ticks=s_plot)
            ax.set_xticklabels(labels=s)

        # Add a global title
        plt.suptitle('s vs Allreduce time | blksize = {}, {}'.format(blksize, kernel_type))

        # Save the figure before displaying
        if "newsbinary" in directory:
            filename = 'CA_Newsbinary_b{}_{}_s_vs_allreduce_time_ds.png'.format(blksize, kernel_type)
        if "duke" in directory:
            filename = 'CA_Duke_b{}_{}_s_vs_allreduce_time_ds.png'.format(blksize, kernel_type)
        if "mnist" in directory:
            filename = 'CA_MNIST_b{}_{}_s_vs_allreduce_time_ds.png'.format(blksize, kernel_type)


        # Use os.path.join to create the full path
        full_path = os.path.join(directory, filename)

        plt.savefig(full_path)

        # Display the plot
        #plt.show()

        plt.close()



        ## Major Time Plot
        
        # Specify your directory
        # directory = "/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/para_ca_newsbinary_linear"

        # Get the number of unique num_process values
        num_columns = len(unique_num_process_values)

        # Define the number of rows
        num_rows = 3
        num_cols = math.ceil(num_columns / num_rows)
        
        # Determine the width of each individual subplot
        subplot_width = 7  # You can adjust this value

        # Determine the total figure width based on the number of columns
        fig_width = subplot_width * num_cols

        # Determine the figure height
        fig_height = 14  # You can adjust this value

        # Create a figure with subplots organized in two rows
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10), sharex=True, sharey=True)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=False)

        # Loop through the unique num_process values
        for i, num_process in enumerate(unique_num_process_values):
            df = data[(data['num_process'] == num_process) & (data['blksize'] == blksize)]
            df = df[0:7]

            # Sort s and major_time in ascending order of s
            sorted_indices = df['s'].argsort()
            s = df['s'].values[sorted_indices]
            major_time = df['major_time'].values[sorted_indices]

            # Create a new variable for plotting, with equally spaced values
            s_plot = np.arange(len(s))

            # Get the current axis based on the index i
            ax = axes[i // num_cols, i % num_cols]

            # Plot on the current axis
            ax.plot(s_plot, major_time, marker='o')

            # Set the labels and title
            ax.set_xlabel('s')
            ax.set_ylabel('Major time (s)')
            title_str = 'np = {}'.format(num_process)
            ax.set_title(title_str)
            ax.set_xticks(ticks=s_plot)
            ax.set_xticklabels(labels=s)

        # Add a global title
        plt.suptitle('s vs Major time | blksize = {}, {}'.format(blksize, kernel_type))

        # Save the figure before displaying
        #filename = '{}_s_vs_major_time.png'.format(kernel_type)
        if "newsbinary" in directory:
            filename = 'CA_Newsbinary_b{}_{}_s_vs_major_time_ds.png'.format(blksize, kernel_type)
        if "duke" in directory:
            filename = 'CA_Duke_b{}_{}_s_vs_major_time_ds.png'.format(blksize, kernel_type)
        if "mnist" in directory:
            filename = 'CA_MNIST_b{}_{}_s_vs_major_time_ds.png'.format(blksize, kernel_type)
            
        # Use os.path.join to create the full path
        full_path = os.path.join(directory, filename)

        plt.savefig(full_path)

        # Display the plot
        # plt.show()

        plt.close()
        
 
        # num_process vs blksize
        
        # Plot the impact of increasing processes vs major time reduced (all with blksize = 1)
        df = data[(data['s'] == 0) & (data['blksize'] == blksize)]

        df = df.drop_duplicates(subset='num_process', keep='first')
        df = df.sort_values(by='num_process')

        num_process = df['num_process']
        major_time = df['major_time']
        
        # p_plot = np.arange(len(num_process))

        # Create the line plot
        plt.plot(num_process, major_time, marker='o')

        # Set the labels and title
        plt.xlabel('num processes')
        plt.ylabel('Major time (s)')
        title_str = 'CABDCD Num Processes vs Major time {} BLKSIZE {}'.format(kernel_type, blksize)
        plt.title(title_str)
        # plt.xticks(ticks=p_plot, labels=num_process)
        
        # Save the figure before displaying
        filename = title_str.replace(' ', '_') + '.png'
    
        # Use os.path.join to create the full path
        full_path = os.path.join(directory, filename)

        plt.savefig(full_path)

        plt.close()
        
        # Display the plot
        # plt.show()





