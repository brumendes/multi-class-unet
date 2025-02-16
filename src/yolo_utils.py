import logging
import numpy as np
import cv2
from pathlib import Path


def yolo_label_to_mask(label_file, image_size):
    """
    Convert single YOLO segmentation label file to boolean mask
    
    Args:
        label_file (Path): Path to YOLO label file
        image_size (tuple): Size of original image (height, width)
        
    Returns:
        np.ndarray: Boolean mask (height, width)
    """
    height, width = image_size
    mask = np.zeros((height, width), dtype=bool)  # Changed to boolean type
    
    # Check if file exists
    if not label_file.exists():
        logging.warning(f"Label file not found: {label_file}")
        return mask
    
    try:
        # Read YOLO format points
        with open(label_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                class_id = int(values[0])
                
                points = np.array([float(x) for x in values[1:]], dtype=float)
                points = points.reshape(-1, 2)
                
                # Denormalize coordinates
                points[:, 0] *= width
                points[:, 1] *= height
                points = points.astype(np.int32)
                
                # Create temporary uint8 mask for fillPoly
                temp_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [points], color=1)
                
                # Convert to boolean and combine with existing mask
                mask = mask | (temp_mask > 0)
        
        logging.info(f"Created mask with {np.sum(mask)} positive pixels")
        return mask
        
    except Exception as e:
        logging.error(f"Error processing label file {label_file}: {str(e)}")
        return mask

def create_3d_mask(base_file, num_slices, image_size):
    """
    Create 3D mask from YOLO label files
    
    Args:
        base_file (Path): Path to base label file (one slice)
        num_slices (int): Number of slices in the volume
        image_size (tuple): Size of each slice (height, width)
        
    Returns:
        np.ndarray: 3D boolean mask (depth, height, width)
    """
    height, width = image_size
    mask_3d = np.zeros((num_slices, height, width), dtype=bool)
    
    # Get base filename pattern
    pattern = base_file.stem.rsplit('_', 1)[0]  # Remove slice number
    
    # Process each slice
    for i in range(num_slices):
        slice_file = base_file.parent / f"{pattern}_{i:03d}.txt"
        if slice_file.exists():
            mask_3d[i] = yolo_label_to_mask(slice_file, image_size)

    mask_3d = np.transpose(mask_3d, (0, 2, 1))
    mask_3d = np.swapaxes(mask_3d, 0, 2)
    logging.info(f"Created 3D mask with shape {mask_3d.shape}")
    logging.info(f"Total positive voxels: {np.sum(mask_3d)}")
    return mask_3d