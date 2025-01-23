import os
import random
import numpy as np
import sdf
from tqdm import tqdm

def generate_box_sdf(width, height, depth, grid_size=64):
    """
    Generates an SDF for a box with the given dimensions.
    
    Parameters:
        width (float): Width of the box.
        height (float): Height of the box.
        depth (float): Depth of the box.
        grid_size (int): Resolution of the SDF grid.
    
    Returns:
        np.ndarray: 3D array containing SDF values.
    """
    # Create the box SDF
    box = sdf.Box([width / 2, height / 2, depth / 2])
    
    # Create a grid of points in 3D space
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1)
    
    # Evaluate the SDF at each grid point
    sdf_values = np.vectorize(lambda p: box.sdf(p))(grid)
    
    return sdf_values

def save_sdf_batch(output_dir, num_samples, grid_size=64):
    """
    Generates a batch of SDFs and saves them as NumPy files.
    
    Parameters:
        output_dir (str): Directory to save the SDF files.
        num_samples (int): Number of SDFs to generate.
        grid_size (int): Resolution of the SDF grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_samples), desc="Generating SDFs"):
        # Randomize box dimensions
        width = random.uniform(0.2, 1.0)
        height = random.uniform(0.2, 1.0)
        depth = random.uniform(0.2, 1.0)
        
        # Generate the SDF
        sdf_values = generate_box_sdf(width, height, depth, grid_size)
        
        # Save to a .npy file
        file_path = os.path.join(output_dir, f"box_sdf_{i:04d}.npy")
        np.save(file_path, sdf_values)

def main():
    # Parameters
    output_dir_train = "sdf_batches/train"
    output_dir_test = "sdf_batches/test"
    num_train_samples = 1000
    num_test_samples = 200
    grid_size = 64

    # Generate training data
    save_sdf_batch(output_dir_train, num_train_samples, grid_size)

    # Generate testing data
    save_sdf_batch(output_dir_test, num_test_samples, grid_size)

if __name__ == "__main__":
    main()
