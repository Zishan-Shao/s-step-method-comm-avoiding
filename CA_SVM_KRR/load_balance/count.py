import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    """
    Read data from a file into a sparse format.
    Each line in the file is assumed to be an observation.
    Each observation is a series of feature:value pairs.
    """
    feature_counts = {}
    max_feature = 0

    with open(filename, 'r') as file:
        for line in file:
            pairs = line.strip().split()
            for pair in pairs:
                if ':' in pair:  # Skip if the pair is not in 'feature:value' format
                    feature, value = pair.split(':')
                    feature = int(feature)
                    value = float(value)
                    max_feature = max(max_feature, feature)
                    if feature not in feature_counts:
                        feature_counts[feature] = 0
                    if value != 0:
                        feature_counts[feature] += 1

    # Create a numpy array to represent the counts
    nnz_counts = np.zeros(max_feature, dtype=int)
    for feature, count in feature_counts.items():
        nnz_counts[feature - 1] = count  # Adjusting index to be zero-based

    return nnz_counts

def plot_cumulative_histogram(nnz_counts, filename):
    plt.figure()
    plt.hist(nnz_counts, bins=2500, color='green', edgecolor='black', cumulative=True)
    plt.title('Cumulative Histogram of NNZ per Feature')
    plt.xlabel('Number of Non-Zero Elements (NNZ)')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True)
    plt.savefig(filename, format='png')
    plt.close()

def plot_boxplot(nnz_counts, filename):
    plt.figure()
    plt.boxplot(nnz_counts, vert=False)
    plt.title('Boxplot of NNZ per Feature')
    plt.xlabel('Number of Non-Zero Elements (NNZ)')
    plt.grid(True)
    plt.savefig(filename, format='png')
    plt.close()
    
def plot_histogram(nnz_counts, filename):
    plt.figure()

    # Truncate data at nnz = 1000 and count everything beyond 1000 as the last bin
    truncated_counts = np.where(nnz_counts > 50, 50, nnz_counts)
    bins = np.arange(52) - 0.5  # Create 1001 bins (0 to 1000, and one extra for 1000+)

    plt.hist(truncated_counts, bins=bins, color='blue', edgecolor='black')
    plt.title('Histogram of NNZ per Feature')
    plt.xlabel('Number of Non-Zero Elements (NNZ)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.xlim([0, 51])  # Set the x-axis limit to show up to the 1000+ bin
    plt.xticks(np.arange(0, 51, 5))  # Adjust ticks for better readability
    plt.savefig(filename, format='png')
    plt.close()

def main():
    filename = 'news20.binary'  # Replace with your dataset filename
    output_file = 'nnz_output.txt'

    # Read data and count nnz per feature
    nnz_counts = read_data(filename)

    # Find the feature with the maximum number of observations
    max_nnz_feature = np.argmax(nnz_counts) + 1  # Adding 1 to make it one-based index
    max_nnz_count = nnz_counts[max_nnz_feature - 1]

    print(f"Total nnz: {np.sum(nnz_counts)}")
    print(f'Feature with maximum observations: Feature {max_nnz_feature}, Count: {max_nnz_count}')

    # Write nnz counts to a file
    with open(output_file, 'w') as f:
        for count in nnz_counts:
            f.write(f'{count} ')
        #f.write(f'{np.sum(nnz_counts)}')  # Append total nnz at the end
        # Plotting the histogram
    
    # Plot and save cumulative histogram
    plot_cumulative_histogram(nnz_counts, filename='cum_hist_nnz.png')

    # Plot and save boxplot
    plot_boxplot(nnz_counts, filename='box_nnz.png')
    
    plot_histogram(nnz_counts, filename='hist_nnz.png')


if __name__ == '__main__':
    main()
