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


    # Tiles binary image into 256x256x3 tiles without overlap and saves tiles to
    # a list. Each tile information is saved as a dictionary in the list.
    # Dictionary keys: dataset name, parent image path, path to tile, tile ID,
    # upper left coordinates, tile width, and tile height in that order.
    tiles = []
    PATH = "ABMR_06282023S1_Area1.tif"
    DATASET = "allograft_proof_of_concept"
    test_width, test_height = cleaned_pil.size
    IMAGE_WIDTH = test_width
    IMAGE_HEIGHT = test_height

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


    saved_tiles = 0
    id = 0
    for i in range(num_tiles_row):
        for j in range(num_tiles_col):
            r0 = start_row + i * TILE_SIZE
            r1 = r0 + TILE_SIZE
            c0 = start_col + j * TILE_SIZE
            c1 = c0 + TILE_SIZE

            if r1 > cleaned_binary.shape[0] or c1 > cleaned_binary.shape[1]:
                continue

            # Save tile as tif if and only if the percentage of white pixels
            # in the tile is greater than or equal to 75%
            tile = cleaned_binary[r0:r1, c0:c1]
            THRESHOLD = 0.75

            tile_white_pixels = np.sum(tile > 0)
            tile_total_pixels = tile.size
            tile_white_ratio = tile_white_pixels / tile_total_pixels

            if tile_white_ratio >= THRESHOLD:
                tile_img = Image.fromarray(tile)
                tile_path = f"tiles/{PATH}_tile_{i}_{j}.png"
                tile_img.save(tile_path)
                
                # Save tile information into a dictionary entry in a list
                tiles.append(dict(dataset=str(DATASET), parent_img=str(PATH),
                                path=str(tile_path), tile_id=int(id),
                                starting_point=(int(r0), int(c0)),
                                tile_width=int(TILE_SIZE), 
                                tile_height=int(TILE_SIZE),
                                image_width=int(IMAGE_WIDTH),
                                image_height=int(IMAGE_HEIGHT)))

                saved_tiles += 1
                id += 1

    print(f"Created {saved_tiles} tiles")

    # Write the tiling scheme to disk
    # Binary image with grid overlaid in red
    # QC only
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(cleaned_binary, cmap='gray')

    # Only draw rectangles for tiles that were saved
    for tile_info in tiles:
        r0, c0 = tile_info["starting_point"]
        rect = patches.Rectangle((c0, r0), TILE_SIZE, TILE_SIZE, 
                                linewidth=1, edgecolor='red', 
                                facecolor='none')
        ax.add_patch(rect)

    plt.savefig("threshold_tiling_overlay_preview.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved tile scheme") 

    
    return tiles


if __name__ == "__main__":
    tiles_poc = main()
    with open("tiles.json", "w") as tile_json:
        json.dump(tiles_poc, tile_json, indent=4)
        print("Saved tile list as json")
    print("Finished")
