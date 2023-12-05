import os
import cv2
import numpy as np
from tqdm import tqdm
def calculate_mean_std(input_dir):
    # Initialize variables to accumulate pixel values

    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    r_values = None
    g_values = None
    b_values = None
    # Loop through each image and accumulate pixel values
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)  # Read the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # Accumulate pixel values
        if r_values is None:
            r_values = np.array(image[..., 0].flatten())
            g_values = np.array(image[..., 1].flatten())
            b_values = np.array(image[..., 2].flatten())
        else:
            r_values = np.append(r_values, image[..., 0].flatten())
            g_values = np.append(g_values, image[..., 1].flatten())
            b_values = np.append(b_values, image[..., 2].flatten())

    # Calculate the mean and standard deviation
    mean_values = [np.mean(r_values), np.mean(g_values), np.mean(b_values)]
    std_dev_values = [np.std(r_values), np.std(g_values), np.std(b_values)]
    return mean_values, std_dev_values
# Example usage:
input_directory = '/good/dataset/coco/train_saliency576_omni'
mean_values, std_dev_values = calculate_mean_std(input_directory)
# Print the results
print(f"Mean values for each channel: {mean_values}")
print(f"Standard deviation values for each channel: {std_dev_values}")