import os
import random
import sdf
from tqdm import tqdm

def generate_and_save_boxes(output_dir, num_samples):
    """
    Generates random boxes with varying dimensions and saves their SDFs.
    
    Parameters:
        output_dir (str): Directory to save the SDF files.
        num_samples (int): Number of SDFs to generate.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_samples), desc="Generating SDF boxes"):
        # Generate random dimensions for the box
        width = random.uniform(0.2, 1.0)
        height = random.uniform(0.2, 1.0)
        depth = random.uniform(0.2, 1.0)
        
        # Create the SDF for the box
        box = sdf.box((width / 2, height / 2, depth / 2))
        
        # Save the SDF to a file
        file_path = os.path.join(output_dir, f"box_{i:04d}.stl")
        box.save(file_path)

def main():
    # Parameters
    output_dir_train = "sdf_boxes/train"
    output_dir_test = "sdf_boxes/test"
    num_train_samples = 10
    num_test_samples = 2

    # Generate training data
    generate_and_save_boxes(output_dir_train, num_train_samples)

    # Generate testing data
    generate_and_save_boxes(output_dir_test, num_test_samples)

if __name__ == "__main__":
    main()
