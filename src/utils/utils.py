import matplotlib.pyplot as plt
import numpy as np

def plotDifferenceMatrix(matrix1, matrix2, fileSaveName = "absolute_difference_matrix"):
    # Compute the absolute difference matrix
    diff_matrix = np.abs(matrix2 - matrix1)

    plt.figure(figsize=(10, 10))

    # Plot the absolute difference matrix as a heatmap
    # Define the values range for the colorbar
    # othervalues for cmap are 'hot', 'jet', 'gray', 'viridis', 'magma', 'inferno', 'plasma'
    plt.imshow(diff_matrix, cmap='hot', vmin=0, vmax=1)

    # plt.imshow(diff_matrix, cmap='hot')
    plt.title(fileSaveName)
    plt.colorbar(label='Absolute Difference')

    # Hide axis ticks and labels
    plt.xticks([])
    plt.yticks([])

    # Save the plot as a PNG image
    plt.savefig(f'../results/{fileSaveName}.png', bbox_inches='tight')
    plt.close("all")

def running_average(new_value, previous_average, n):
    return (previous_average * n + new_value) / (n + 1)

# Algorithm to draw i.i.d samples of p from a matrix
def sample_columns(matrix, num_samples = 6):
    num_columns = matrix.shape[1]
    sample_indices = np.random.choice(num_columns, num_samples, replace=False)
    sampled_matrix = matrix[:, sample_indices]
    return sampled_matrix