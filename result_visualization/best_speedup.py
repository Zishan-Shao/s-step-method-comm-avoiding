import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Folders with your data
files = [ '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_duke_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_mnist_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_newsbinary_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_duke_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_mnist_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_newsbinary_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_duke_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_mnist_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/caksvm_data_newsbinary_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_linear.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_poly.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_duke_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_mnist_gauss.csv',
           '/Users/shaozishan/Desktop/Data_Plot_BDCD/NERSC_outputs/para_ca_data_newsbinary_gauss.csv']


directory = '/Users/shaozishan/Desktop/Data_Plot_BDCD/best_speedup.csv'


# Define the column names
columns = ["experiment", "data", "kernel", "blksize", "speedup", "best_np", "best_s"]

# Create an empty DataFrame with the specified columns
outputs = pd.DataFrame(columns=columns)

#if not os.path.exists(directory):
#    os.makedirs(directory)


for file in files:
    
    if "caksvm" in file:
        experiment = "ksvm"
    if "para" in file:
        experiment = "bdcd"
    
    curr_data = pd.read_csv(file)
    
    unique_blksize = curr_data['blksize'].unique()
    unique_blksize = sorted(unique_blksize)
    
    print(unique_blksize)
    
    if "linear" in file:
        kernel_type = "linear"
    if "poly" in file:
        kernel_type = "poly"
    if "gauss" in file:
        kernel_type = "gauss"
        
    if "mnist" in file:
        exp_data = "mnist"
    if "duke" in file:
        exp_data = "duke"
    if "news" in file:
        exp_data = "newsbinary"
        
    
    for blksize in unique_blksize:
        
        data = curr_data[curr_data['blksize'] == blksize]
        # Assuming data is your DataFrame
        
        # find the smallest major_time value at s = 0
        best_bdcd = data[data['s'] == 0]["major_time"].min()
        
        # find the smallest major_time value at any s
        best_cabdcd = data["major_time"].min()
        
        # compute the speedup
        speedup = best_bdcd / best_cabdcd
        
        if (speedup <= 1):
            speedup = 0
        
        # find the settings for the best output
        min_major_time_row = data[data['major_time'] == data['major_time'].min()]

        # Extract the 'num_process' value from that row
        best_np = min_major_time_row['num_process'].values[0]
        best_s = min_major_time_row['s'].values[0]
        
        
        curr_row = [experiment, exp_data, kernel_type, blksize, speedup, best_np, best_s]
        
        # append current speedup info to the output
        outputs.loc[len(outputs)] = curr_row
        
        
print(outputs.describe())
        
# Write to csv
outputs.to_csv(directory, index=False)