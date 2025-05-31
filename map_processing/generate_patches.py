from pathlib import Path
import cv2
import numpy as np
import json

def get_map_path() -> Path:
    """
    Returns the path to the map image file based on the root path and map name.
    """
    root_path = Path(__file__).parent.parent  # Adjust this path as needed
    map_name = "map.png"
    map_path = root_path / "data" / map_name

    if not map_path.exists():
        raise FileNotFoundError(f"Map file '{map_name}' not found in '{root_path / 'data'}'.")

    return root_path / "data" / "map.png"

def load_map_image(map_path: Path) -> np.ndarray:

    # Load map as grayscale
    map_img = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise FileNotFoundError(f"Map image not found at: {map_path}")
    
    return map_img

def split_map_into_patches(image, patch_height, patch_width, stride_y, stride_x):
    h, w = image.shape[:2]
    patches = []
    positions = []

    for y in range(0, h - patch_height + 1, stride_y):
        for x in range(0, w - patch_width + 1, stride_x):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)
            positions.append((x, y))
    

    # include last patch with different overlap
    if h % patch_height > 0:
        y = h - patch_height
        for x in range(0, w - patch_width + 1, stride_x):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)
            positions.append((x, y))
    if w % patch_width > 0:
        x = w - patch_width
        for y in range(0, h - patch_height + 1, stride_y):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)
            positions.append((x, y))

    return patches, positions

def draw_patch_grid(image, patch_height, patch_width, stride_y, stride_x):
    vis = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    for y in range(0, h - patch_height + 1, stride_y):
        for x in range(0, w - patch_width + 1, stride_x):
            cv2.rectangle(vis, (x, y), (x + patch_width, y + patch_height), (0, 255, 0), 1)
    
    if h % patch_height > 0:
        y = h - patch_height
        for x in range(0, w - patch_width + 1, stride_x):
            cv2.rectangle(vis, (x, y), (x + patch_width, y + patch_height), (0, 255, 0), 1)
    if w % patch_width > 0:
        x = w - patch_width
        for y in range(0, h - patch_height + 1, stride_y):
            cv2.rectangle(vis, (x, y), (x + patch_width, y + patch_height), (0, 255, 0), 1)
    return vis



def main(patch: dict, draw_patches: bool = False):

    map_path = get_map_path()
    map_img = load_map_image(map_path)

    patches, positions = split_map_into_patches(map_img, patch["height"], patch["width"], patch["stride_y"], patch["stride_x"])

    if draw_patches:
        grid_overlay = draw_patch_grid(map_img, patch["height"], patch["width"], patch["stride_y"], patch["stride_x"])
        cv2.imwrite(Path(__file__).parent / "patch_grid_overlay.png", grid_overlay)

    return patches, positions

def save_patches(patches, positions, patch_height, patch_width):

    output_dir = Path(__file__).parent / "patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove all files in the output_dir before saving new patches
    for file in output_dir.glob("*.png"):
        file.unlink()

    for i, patch in enumerate(patches):
        patch_filename = output_dir / f"patch_{i:04d}.png"
        cv2.imwrite(str(patch_filename), patch)

    print(f"Saved {len(patches)} patches to {output_dir}")
    
    # Save patch position data as JSON files
    save_patch_positions(positions, patch_height, patch_width)

def calculate_patch_centers(positions, patch_height, patch_width):
    """
    Calculate the center positions of patches relative to the map's top-left corner (0,0).
    
    Args:
        positions: List of (x, y) tuples representing top-left corners of patches
        patch_height: Height of each patch
        patch_width: Width of each patch
    
    Returns:
        List of dictionaries with center_x and center_y coordinates
    """
    centers = []
    for i, (x, y) in enumerate(positions):
        center_x = x + patch_width // 2
        center_y = y + patch_height // 2
        centers.append({
            "patch_id": i,
            "center_x": int(center_x),
            "center_y": int(center_y),
            "top_left_x": int(x),
            "top_left_y": int(y)
        })
    return centers

def save_patch_positions(positions, patch_height, patch_width):
    """
    Save patch center positions as JSON files.
    Creates both individual JSON files for each patch and a combined JSON file.
    """
    output_dir = Path(__file__).parent / "patches"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing JSON files
    for file in output_dir.glob("*.json"):
        file.unlink()
    
    centers = calculate_patch_centers(positions, patch_height, patch_width)
    
    # Save individual JSON files for each patch
    for center_data in centers:
        patch_id = center_data["patch_id"]
        json_filename = output_dir / f"patch_{patch_id:04d}.json"
        
        with open(json_filename, 'w') as f:
            json.dump(center_data, f, indent=2)
    
    # Save combined JSON file with all patch positions
    combined_filename = output_dir / "all_patch_positions.json"
    with open(combined_filename, 'w') as f:
        json.dump({
            "total_patches": len(centers),
            "patch_dimensions": {
                "height": patch_height,
                "width": patch_width
            },
            "patches": centers
        }, f, indent=2)
    
    print(f"Saved {len(centers)} patch position JSON files to {output_dir}")

if __name__ == "__main__":

    patch_sizes = [256, 384, 512, 640, 768, 1024, 1280, 1536, 1792, 2048]
    patch_height = patch_sizes[3]
    patch_width = int(patch_height * 4 / 3)

    stride_sizes = [1, 2, 4, 8]
    stride_size = stride_sizes[1]
    stride_y = patch_height // stride_size
    stride_x = patch_width // stride_size

    patch = {
        "height": patch_height,
        "width": patch_width,
        "stride_y": stride_y,
        "stride_x": stride_x,
    }

    patches, positions = main(patch, draw_patches=True)

    save_patches(patches, positions, patch_height, patch_width)