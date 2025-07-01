# Import libraries 
import numpy as np
from PIL import Image, ImageOps  # Python Imaging Library (PIL)
Image.MAX_IMAGE_PIXELS = None  # Eliminate max pixel attribute for large images
# Gaussian blur, fill holes
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import threshold_otsu  # Otsu threshold
from skimage.measure import regionprops, label

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

def main():
    """
    Opens an image, inverts the colors, then converts to grayscale not in-place.
    Applies Gaussian blur, Otsu thresholding, 
    """
    # Open image
    path = "UNI/REINHARD_HE_IMAGES/train/ABMR/ABMR_06282023S3_Area1_reinhard_norm.tif"
    test = Image.open(path)
    print(f"Loaded slide {path}")

    # FINE MASK ===
    # Invert colors and convert to grayscale
    # Equivalent of taking the mean of the channels and inverting?
    test_grayscale = ImageOps.invert(test).convert("L")
    print("Converted RGB slide to grayscale")
    # Apply Gaussian blur and Otsu threshold (--> binary array)
    test_gaussian = gaussian_filter(input=test_grayscale, sigma=1)
    # test_gaussian_image = Image.fromarray(test_gaussian)
    threshold = threshold_otsu(test_gaussian)
    binary = test_gaussian > threshold
    # binary_pil = Image.fromarray((binary * 255).astype(np.uint8))
    # binary_pil.save("fine_mask.tif")
    print("Applied Gaussian blur and Otsu thresholding")
    

    # BACKGROUND SUBTRACTION ===
    # Create a copy of the RGB image array
    test_array = np.array(test)
    masked_image = test_array.copy()
    # Set background pixels (where mask is False) to white (255, 255, 255)
    masked_image[~binary] = [255, 255, 255]
    # Convert back to PIL Image
    # masked_pil = Image.fromarray(masked_image)
    # Save the masked image
    # masked_pil.save("subtract_bg.tif")
    print("Saved masked image with white background")


    # CREATE COARSE MASK FOR TILING ===

    
if __name__ == "__main__":
    main()