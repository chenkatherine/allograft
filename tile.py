# Import libraries 
import numpy as np
from PIL import Image, ImageOps  # Python Imaging Library (PIL)
Image.MAX_IMAGE_PIXELS = None  # Eliminate max pixel attribute for large images
# Gaussian blur, fill holes
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import threshold_otsu  # Otsu threshold
from skimage.measure import regionprops, label

import cv2

def main():
    """
    Opens an image, inverts the colors, then converts to grayscale not in-place.
    Applies Gaussian blur, Otsu thresholding, fills holes, and removes tissue
    fragments in preparation for tiling. 
    """
    # Open image
    path = "ABMR_06282023S1_Area1.tif"
    test = Image.open(path)
    print(f"Loaded slide {path}")
    

    # Invert colors and convert to grayscale
    test_grayscale = ImageOps.invert(test).convert("L")
    print("Converted RGB slide to grayscale")


    # Apply Gaussian blur and Otsu threshold (--> binary array)
    test_gaussian = gaussian_filter(input=test_grayscale, sigma=20)
    # test_gaussian_image = Image.fromarray(test_gaussian)
    threshold = threshold_otsu(test_gaussian)
    binary = test_gaussian > threshold
    binary_pil = Image.fromarray((binary * 255).astype(np.uint8))
    # binary_pil.save("ABMR_06282023S1_Area1_gaussian_20_otsu.tif")
    print("Applied Gaussian blur and Otsu thresholding")


    # Creates binary array from Gaussian/Otsu treated image and fills holes
    # Smooths out texture of slide
    binary_array = np.array(binary_pil)
    filled = binary_fill_holes(binary_array)
    filled_pil = Image.fromarray((filled * 255).astype(np.uint8))
    # filled_pil.save("ABMR_06282023S1_Area1_gaussian_otsu_filled.tif")
    print("Filled holes")
    

    # Removes recognized tissue fragments not in the main mask that are larger 
    # than the area of 3x3 tiles
    labeled = label(filled)
    cleaned = np.copy(labeled)  # type: np.ndarray
                                # shape: (17551, 68049)
    
    regions = regionprops(labeled)
    
    FRAGMENTS_MAX = (256 * 3) ** 2
    for region in regions:
        if region.area <= FRAGMENTS_MAX:
            for _, (r, c) in enumerate(region.coords):
                cleaned[r, c] = 0
    
    cleaned_binary = (cleaned > 0).astype(np.uint8) * 255
    # cleaned_pil = Image.fromarray((cleaned * 255).astype(np.uint8))
    # cleaned_pil.save("ABMR_06282023S1_Area1_gaussian_otsu_filled_cleaned.tif")
    cleaned_pil = Image.fromarray(cleaned_binary)
    # cleaned_pil.save("ABMR_06282023S1_Area1_gaussian_otsu_filled_cleaned.tif")
    print("Removed noise")


    # Tiles image into 256x256x3 tiles
    non_zero_coords = np.argwhere(cleaned > 0)
    min_row, min_col = np.min(non_zero_coords, axis=0)
    max_row, max_col = np.max(non_zero_coords, axis=0)
    print(f"Bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")

    TILE_SIZE = 256

    num_tiles_row = int(np.ceil((max_row - min_row) / TILE_SIZE))
    num_tiles_col = int(np.ceil((max_col - min_col) / TILE_SIZE))

    D_row = num_tiles_row * TILE_SIZE
    D_col = num_tiles_col * TILE_SIZE

    pad_top = int((D_row - (max_row - min_row)) / 2)
    pad_left = int((D_col - (max_col - min_col)) / 2)

    start_row = max(min_row - pad_top, 0)
    start_col = max(min_col - pad_left, 0)

    print(f"Tiling starting from row {start_row}, col {start_col}")
    print(type(cleaned_binary))

    saved_tiles = 0
    for i in range(num_tiles_row):
        for j in range(num_tiles_col):
            r0 = start_row + i * TILE_SIZE
            r1 = r0 + TILE_SIZE
            c0 = start_col + j * TILE_SIZE
            c1 = c0 + TILE_SIZE

            # Boundary check
            if r1 > cleaned_binary.shape[0] or c1 > cleaned_binary.shape[1]:
                continue

            tile = cleaned_binary[r0:r1, c0:c1]
            tile_img = Image.fromarray(tile)
            tile_img.save(f"tiles/tile_{i}_{j}.png")
            saved_tiles += 1
    print(f"Created {saved_tiles} tiles")

if __name__ == "__main__":
    main()
    print("Finished")
