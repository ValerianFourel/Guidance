import torch
import torchvision.utils as vutils
import os
from pathlib import Path

def tensor_to_grid_picture(tensor, output_folder, filename="grid_picture.png"):
    """
    Transform a tensor of shape (4, 3, 512, 512) into a grid picture of size 1024x1024
    and save it as a PNG file in the specified folder.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (4, 3, 512, 512)
    output_folder (str): Path to the folder where the image will be saved
    filename (str, optional): Name of the output file. Defaults to "grid_picture.png"
    
    Returns:
    str: Path to the saved image file
    """
    # Check input shape
    if tensor.shape != (4, 3, 512, 512):
        raise ValueError(f"Expected input shape (4, 3, 512, 512), but got {tensor.shape}")
    
    # Normalize the tensor to [0, 1] range if it's not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Create a 2x2 grid from the 4 images
    grid = vutils.make_grid(tensor, nrow=2, padding=0)
    
    # Ensure the output folder exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Full path for the output file
    output_file = output_path / filename
    
    # Save the grid picture as PNG
    vutils.save_image(grid, output_file)
    
    return str(output_file)

