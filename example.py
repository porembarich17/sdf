import random
import os
import numpy as np
import torch
import json
from sdf import * 
import trimesh
import argparse

parser = argparse.ArgumentParser(description="Generate SDF samples for perforated plates.")
parser.add_argument('--holes', action='store_true', help='Add holes to the generated plates')
args = parser.parse_args()

# Function to generate random dimensions for the plate
def generate_plate_dimensions():
    height = random.uniform(0.01, 0.1) 
    width = random.uniform(height, 1.5)  
    length = random.uniform(height, 1.5) 
    return round(width, 2), round(length, 2), round(height, 2)

# Directory structure
output_dir = 'perforated_plates'
os.makedirs(output_dir, exist_ok=True)

# Metadata for .datasources.json
datasource_info = {
    "name": "Generated Plates Dataset",
    "version": "1.0",
    "description": "Dataset of perforated plates with SDF and surface samples.",
    "classes": ["perforated_plate"],
    "num_samples": 250
}

def perforated_plate_with_holes(dimensions, hole_radius=0.05, hole_spacing=0.15):
    """
    Creates an SDF of a box with cylindrical holes going through it.

    Parameters:
        dimensions (tuple): (width, height, length) of the plate
        hole_radius (float): radius of each cylindrical hole
        hole_spacing (float): distance between holes in a grid pattern

    Returns:
        function: SDF function representing the perforated plate
    """
    width, height, length = dimensions

    # Base box
    base = box((width, length, height))

    count_width = (width/2) // (hole_spacing)
    count_length = (length/2) // (hole_spacing)

    print(count_width)
    print(count_length)
    cy = cylinder(hole_radius).repeat(hole_spacing, (count_width,count_length,0))

    # Subtract the holes from the base
    perforated = base - cy
    perforated.save('out.stl')

    return perforated



def generate_sdf_samples(box_dimensions, resolution=100, with_holes=False):
    width, length, height = box_dimensions
    if with_holes:
        f = perforated_plate_with_holes((width, height, length))  # You define this function
    else:
        f = box((width, height, length))  # Simple box SDF

    # 3D Grid
    margin = 0.1
    x = np.linspace(-width/2 - margin, width/2 + margin, resolution)
    y = np.linspace(-height/2 - margin, height/2 + margin, resolution)
    z = np.linspace(-length/2 - margin, length/2 + margin, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T  # shape (N, 3)

    # Compute SDF values — slow method, but works
    distances = np.array([f(p.reshape(1, -1)) for p in points]).flatten()

    print(f"[{i}] SDF stats: min={distances.min():.4f}, max={distances.max():.4f}")

    # Create boolean masks
    pos_mask = distances > 0
    neg_mask = distances <= 0

    pos = points[pos_mask]
    neg = points[neg_mask]

    pos_sdf = distances[pos_mask]
    neg_sdf = distances[neg_mask]

    # Concatenate points with their sdf values → (N, 4)
    pos = np.hstack((pos, pos_sdf[:, np.newaxis]))
    neg = np.hstack((neg, neg_sdf[:, np.newaxis]))

    return pos, neg


# Main generation loop
for i in range(81):
    width, length, height = generate_plate_dimensions()
    
    # Generate SDF for the box
    pos_samples, neg_samples = generate_sdf_samples((width, length, height), 100, args.holes)

    # Save the SDF as a .npz file
    # Combine for training format
    all_points = np.vstack((pos_samples[:, :3], neg_samples[:, :3]))
    all_sdf = np.hstack((pos_samples[:, 3], neg_samples[:, 3]))

    # Save training data format
    train_filename = os.path.join(output_dir, f'perforated_plate_{i}_train.npz')
    np.savez_compressed(train_filename, pos=all_points.astype(np.float32), sdf=all_sdf.astype(np.float32))

    # Save testing data format
    test_filename = os.path.join(output_dir, f'perforated_plate_{i}_test.npz')
    np.savez_compressed(test_filename, pos=pos_samples.astype(np.float32), neg=neg_samples.astype(np.float32))

    # Create a surface mesh for the box
    #mesh = trimesh.creation.box(extents=[width, height, length])

    # Save the surface mesh as a .ply file
    #ply_filename = os.path.join(output_dir, f'perforated_plate_{i}.ply')
    #mesh.export(ply_filename)

    print(str(i) + " done")

# Create the .datasources.json metadata file
datasources_json_path = os.path.join(output_dir, '.datasources.json')
with open(datasources_json_path, 'w') as f:
    json.dump(datasource_info, f, indent=4)

print("Data generation complete!")
