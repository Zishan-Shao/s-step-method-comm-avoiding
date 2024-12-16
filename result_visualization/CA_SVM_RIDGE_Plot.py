# for loop to save the figures
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Folders with your data
files = [ '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_duke_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_mnist_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_newsbinary_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_gauss.csv']


directories = ['/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/caksvm_duke_gauss',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/caksvm_mnist_gauss',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/caksvm_newsbinary_gauss',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_duke_gauss',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_mnist_gauss',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/cabdcd_newsbinary_gauss']



# data = pd.read_csv("/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_newsbinary_linear.csv")

#for file in files: 
for file, directory in zip(files, directories):
    
    data = pd.read_csv(file)
    
    # Assuming data is your DataFrame
    unique_num_process_values = data['num_process'].unique()
    
    kernel_type = data['kernel'][0]

    # Specify your directory
    # directory = "/Users/shaozishan/Desktop/Data_Plot_BDCD/figures/caksvm_newsbinary_linear"

    for num_process in unique_num_process_values:
        df = data[(data['num_process'] == num_process) & (data['blksize'] == 1)]
        df = df[0:7]

        # s = df['s']
        # major_time = df['major_time']

        # Create the line plot
        # plt.figure()  # start a new figure for each loop
        # plt.plot(s, major_time, marker='o')
        # Sort s and major_time in ascending order of s
        sorted_indices = df['s'].argsort()
        s = df['s'].values[sorted_indices]
        major_time = df['major_time'].values[sorted_indices]

        # Create a new variable for plotting, with equally spaced values
        s_plot = np.arange(len(s))

        # Create the line plot, using s_plot for the x-coordinates
        plt.plot(s_plot, major_time, marker='o')

        # Set the labels and title
        plt.xlabel('s')
        plt.ylabel('Major time (s)')
        title_str = 's vs Major time | blksize = 1, np = {}, {}'.format(num_process, kernel_type)
        plt.title(title_str)
        plt.xticks(ticks=s_plot, labels=s)

        # Save the figure before displaying
        filename = title_str.replace(' ', '_') + '.png'
        
        # Use os.path.join to create the full path
        full_path = os.path.join(directory, filename)

        plt.savefig(full_path)
        
        # Display the plot
        # plt.show()
        plt.close()
        
    for num_process in unique_num_process_values:
        df = data[(data['num_process'] == num_process) & (data['blksize'] == 1)]
        df = df[0:7]

        #s = df['s']
        #major_time = df['allreduce_time']

        # Create the line plot
        #plt.figure()  # start a new figure for each loop
        #plt.plot(s, major_time, marker='o')
        
        # Sort s and major_time in ascending order of s
        sorted_indices = df['s'].argsort()
        s = df['s'].values[sorted_indices]
        major_time = df['allreduce_time'].values[sorted_indices]

        # Create a new variable for plotting, with equally spaced values
        s_plot = np.arange(len(s))

        # Create the line plot, using s_plot for the x-coordinates
        plt.plot(s_plot, major_time, marker='o')


        # Set the labels and title
        plt.xlabel('s')
        plt.ylabel('Allreduce time (s)')
        title_str = 's vs Allreduce time | blksize = 1, np = {}, {}'.format(num_process, kernel_type)
        plt.title(title_str)
        plt.xticks(ticks=s_plot, labels=s)
    
        # Save the figure before displaying
        filename = title_str.replace(' ', '_') + '.png'
        
        # Use os.path.join to create the full path
        full_path = os.path.join(directory, filename)

        plt.savefig(full_path)
        
        # Display the plot
        # plt.show()
        plt.close()
    
    # Plot the impact of increasing processes vs major time reduced (all with blksize = 1)
    df = data[(data['s'] == 0) & (data['blksize'] == 1)]

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
    title_str = 'Num Processes vs Major time {}'.format(kernel_type)
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




