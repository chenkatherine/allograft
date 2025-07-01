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

def tile(wsi_path):
    """
    Opens a WSI, inverts the colors, then converts to grayscale not in-place.
    Applies Gaussian blur, Otsu thresholding, and subtracts background (set to 255).
    Tiles background subtracted WSI into 256x256 tiles and saves tile details
    to json.
    """
    # OPEN WSI ===
    path = "UNI/REINHARD_HE_IMAGES/train/ABMR/ABMR_06282023S3_Area1_reinhard_norm.tif"
    wsi = Image.open(wsi_path)
    print(f"Loaded slide {path}")

    # FINE MASK ===
    # Invert colors and convert to grayscale
    # Equivalent of taking the mean of the channels and inverting?
    grayscale = ImageOps.invert(wsi).convert("L")
    print("Converted RGB slide to grayscale")
    # Apply Gaussian blur and Otsu threshold (--> binary array)
    fine_gaussian = gaussian_filter(input=grayscale, sigma=1)
    threshold = threshold_otsu(fine_gaussian)
    binary = fine_gaussian > threshold
    print("Applied Gaussian blur and Otsu thresholding")
    

    # BACKGROUND SUBTRACTION ===
    # Create a copy of the RGB image array
    test_array = np.array(wsi)
    masked_image = test_array.copy()
    # Set background pixels (where mask is False) to white (255, 255, 255)
    masked_image[~binary] = [255, 255, 255]
    cleaned_image = masked_image
    print("Subtracted background")


    # CREATE COARSE MASK FOR TILING ===
    # Convert to greyscale
    masked_image = ImageOps.invert(wsi).convert("L")
    # Apply Gaussian blur with a larger sigma and Otsu threshold (--> binary array)
    coarse_gaussian = gaussian_filter(input=masked_image, sigma=20)
    coarse_threshold = threshold_otsu(coarse_gaussian)
    coarse_binary = coarse_gaussian > coarse_threshold
    # Fill holes in binary mask
    coarse_filled = binary_fill_holes(coarse_binary)
    print("Created coarse mask for tiling")

    # REMOVE TISSUE FRAGMENTS NOT IN THE MAIN MASK THAT ARE SMALLER THAN THE AREA 
    # OF 3X3 TILES ===
    coarse_labeled = label(coarse_filled)
    coarse_cleaned = np.copy(coarse_labeled)
    
    regions = regionprops(coarse_labeled)
    
    FRAGMENTS_MAX = (256 * 3) ** 2
    for region in regions:
        if region.area <= FRAGMENTS_MAX:
            for _, (r, c) in enumerate(region.coords):
                coarse_cleaned[r, c] = 0
    cleaned_tile_mask = (coarse_cleaned > 0).astype(np.uint8) * 255
    cleaned_mask_image = Image.fromarray(cleaned_tile_mask)
    print("Removed artefacts")


    # TILING WSI ===
    # Tiles binary image into 256x256x3 tiles without overlap and saves only
    # tiles that are >=75% tissue to a list. Each tile information is 
    # saved as a dictionary in the list. List is ultimately saved as json.
    # Dictionary keys: dataset name, parent image path, path to tile, tile ID,
    # upper left coordinates, tile width, and tile height in that order.
    tiles = []

    DATASET = "PROOF_OF_CONCEPT"
    SLIDE = wsi_path.split('.')[0]
    WSI_WIDTH, WSI_HEIGHT = cleaned_mask_image.size

    non_zero_coords = np.argwhere(cleaned_tile_mask > 0)
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


    saved_tiles = 0
    id = 0
    for i in range(num_tiles_row):
        for j in range(num_tiles_col):
            r0 = start_row + i * TILE_SIZE
            r1 = r0 + TILE_SIZE
            c0 = start_col + j * TILE_SIZE
            c1 = c0 + TILE_SIZE

            if r1 > cleaned_tile_mask.shape[0] or c1 > cleaned_tile_mask.shape[1]:
                continue

            # Extract the mask tile to check tissue percentage
            mask_tile = cleaned_tile_mask[r0:r1, c0:c1]
            THRESHOLD = 0.75

            tile_white_pixels = np.sum(mask_tile > 0)
            tile_total_pixels = mask_tile.size
            tile_white_ratio = tile_white_pixels / tile_total_pixels

            # Save RGB tile from cleaned_image if tissue percentage is >= 75%
            if tile_white_ratio >= THRESHOLD:
                # Extract RGB tile from cleaned_image
                rgb_tile = cleaned_image[r0:r1, c0:c1]
                tile_img = Image.fromarray(rgb_tile.astype(np.uint8))
                tile_path = f"tiles/{SLIDE}_tile_{i}_{j}.tif"
                tile_img.save(tile_path)
                
                # Save tile information into a dictionary entry in a list
                tiles.append(dict(dataset=str(DATASET), parent_img=str(SLIDE),
                                path=str(tile_path), tile_id=int(id),
                                starting_point=(int(r0), int(c0)),
                                tile_width=int(TILE_SIZE), 
                                tile_height=int(TILE_SIZE),
                                image_width=int(WSI_WIDTH),
                                image_height=int(WSI_HEIGHT)))

                saved_tiles += 1
                id += 1
    print(f"Saved {saved_tiles} tiles")
    return tiles


def main():
    tiles = tile("bad.tif")
    with open("tiles.json", "w") as tile_json:
        json.dump(tiles, tile_json, indent=4)
        print("Saved tile list as json")
    print("Finished")
    
if __name__ == "__main__":
    main()
    