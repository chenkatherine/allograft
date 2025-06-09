# Import libraries 
import os
import numpy as np
from PIL import Image, ImageOps  # Python Imaging Library (PIL)
Image.MAX_IMAGE_PIXELS = None  # Eliminate max pixel attribute for large images
# Gaussian blur, fill holes
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import threshold_otsu  # Otsu threshold
from skimage.measure import regionprops, label


def tile(slide_path):
    """
    Tiles the original RGB H&E image into 256x256x3 tiles.
    Uses a binary tissue mask (from grayscale + Otsu) to filter for tiles with ≥75% tissue.
    """
    # Load RGB image
    img_rgb = Image.open(slide_path).convert("RGB")
    print(f"Loaded RGB slide: {slide_path}")

    # Create grayscale for mask generation
    test_grayscale = ImageOps.invert(img_rgb).convert("L")
    test_gaussian = gaussian_filter(input=test_grayscale, sigma=20)
    threshold = threshold_otsu(test_gaussian)
    binary = test_gaussian > threshold
    binary_filled = binary_fill_holes(binary)

    # Remove small fragments using mask
    labeled = label(binary_filled)
    cleaned = np.copy(labeled)
    FRAGMENTS_MAX = (256 * 3) ** 2
    for region in regionprops(labeled):
        if region.area <= FRAGMENTS_MAX:
            for r, c in region.coords:
                cleaned[r, c] = 0
    cleaned_binary = (cleaned > 0).astype(np.uint8) * 255
    print("Created tissue mask and removed small fragments")

    # Tile and filter using tissue mask
    TILE_SIZE = 256
    img_width, img_height = img_rgb.size

    tiles = []
    saved_tiles = 0

    num_tiles_x = img_width // TILE_SIZE
    num_tiles_y = img_height // TILE_SIZE

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            top = i * TILE_SIZE
            left = j * TILE_SIZE
            bottom = top + TILE_SIZE
            right = left + TILE_SIZE

            # Check mask coverage
            mask_tile = cleaned_binary[top:bottom, left:right]
            tissue_ratio = np.sum(mask_tile > 0) / mask_tile.size

            if tissue_ratio >= 0.75:
                rgb_tile = img_rgb.crop((left, top, right, bottom))

                filename = os.path.basename(slide_path)
                basename, _ = os.path.splitext(filename)
                output_dir = "HE_TILES/train/TCMR/"
                tile_path = os.path.join(output_dir, f"{basename}_tile_{i}_{j}.tif")
                rgb_tile.save(tile_path)

                tiles.append(dict(starting_point=(top, left)))
                saved_tiles += 1

    print(f"Saved {saved_tiles} RGB tiles with ≥75% tissue")
    print()

    # QC
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.imshow(img_rgb)

    # for tile_info in tiles:
        # r0, c0 = tile_info["starting_point"]
        # rect = patches.Rectangle((c0, r0), TILE_SIZE, TILE_SIZE,
                                 # linewidth=1, edgecolor='red', facecolor='none')
        # ax.add_patch(rect)

    # plt.savefig("rgb_tiling_overlay_filtered.png", bbox_inches='tight', dpi=300)
    # plt.close(fig)
    # print("Saved QC overlay")

    return tiles

def main():
    dir = "HE_IMAGES/train/TCMR"
    for filename in os.listdir(dir):
        if filename.lower().endswith('.tif'):
            image_path = os.path.join(dir, filename)
            tile(image_path)
    
    print("Finished")

if __name__ == "__main__":
    main()
    print("Finished")
