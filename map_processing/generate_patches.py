from pathlib import Path
import cv2
import numpy as np

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

def save_patches(patches):

    output_dir = Path(__file__).parent / "patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove all files in the output_dir before saving new patches
    for file in output_dir.glob("*.png"):
        file.unlink()

    for i, patch in enumerate(patches):
        patch_filename = output_dir / f"patch_{i:04d}.png"
        cv2.imwrite(str(patch_filename), patch)

    print(f"Saved {len(patches)} patches to {output_dir}")

if __name__ == "__main__":

    patch_sizes = [256, 384, 512, 640]
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

    save_patches(patches)