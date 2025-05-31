
import cv2
from pathlib import Path
import pandas as pd

def get_paths_and_positions():

    patches_dir = Path(__file__).parent / "patches"

    paths = list(patches_dir.glob("*.png"))

    patches_info_path = patches_dir / "info.csv"

    patches_info_df = pd.read_csv(patches_info_path)
    positions = [(row['x'], row['y']) for _, row in patches_info_df.iterrows()]

    return paths, positions

def load_patches(paths):
    """
    Load patches from the specified paths.

    Args:
        paths (list): List of paths to patch images.

    Returns:
        list: List of loaded patch images.
    """
    patches = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            patches.append(img)
        else:
            print(f"Warning: Could not load image at {path}")
    return patches

def get_patch_descriptors(patches, positions, patches_paths):

    orb = cv2.ORB_create(nfeatures=1000)

    patch_descriptors = []

    for i, patch_img in enumerate(patches):

        keypoints, descriptors = orb.detectAndCompute(patch_img, None)
        patch_descriptors.append({
            "keypoints": keypoints,
            "descriptors": descriptors,
            "name": patches_paths[i].name,
            "position": positions[i]
        })


def main():

    """
    Main function to process patches and extract descriptors.
    """

    paths, positions = get_paths_and_positions()


    # Load patches
    loaded_patches = load_patches(paths)

    # Get patch descriptors
    patch_descriptors = get_patch_descriptors(loaded_patches, positions, paths)

    return patch_descriptors

if __name__ == "__main__":

    main()