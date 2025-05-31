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
    max_col = 0
    max_row = 0

    def create_patch_name(row, col):
        row = str(row)
        col = str(col)
        i = row if len(row) > 1 else f"0{row}"
        j = col if len(col) > 1 else f"0{col}"
        return f"patch_{i}_{j}.png"

    # main grid of patches
    for i, y in enumerate(range(0, h - patch_height + 1, stride_y)):
        for j, x in enumerate(range(0, w - patch_width + 1, stride_x)):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append({
                "image": patch,
                "position": (x, y),
                "name": create_patch_name(i, j),
                "center": (x + patch_width // 2, y + patch_height // 2),
                "grid": (i, j)
            })

            max_col = max(max_col, j)
            max_row = max(max_row, i)

    # last row of patches if height is not divisible by patch_height
    if h % patch_height > 0:
        y = h - patch_height
        for j, x in enumerate(range(0, w - patch_width + 1, stride_x)):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append({
                "image": patch,
                "position": (x, y),
                "name": create_patch_name(max_row + 1, j),
                "center": (x + patch_width // 2, y + patch_height // 2),
                "grid": (max_row + 1, j)
            })
    
    # last column of patches if width is not divisible by patch_width
    if w % patch_width > 0:
        x = w - patch_width
        for i, y in enumerate(range(0, h - patch_height + 1, stride_y)):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append({
                "image": patch,
                "position": (x, y),
                "name": create_patch_name(i, max_col + 1),
                "center": (x + patch_width // 2, y + patch_height // 2),
                "grid": (i, max_col + 1)
            })
    
    # last patch if both height and width are not divisible
    if h % patch_height > 0 and w % patch_width > 0:
        y = h - patch_height
        x = w - patch_width
        patch = image[y:y + patch_height, x:x + patch_width]
        patches.append({
            "image": patch,
            "position": (x, y),
            "name": create_patch_name(max_row + 1, max_col + 1),
            "center": (x + patch_width // 2, y + patch_height // 2),
            "grid": (max_row + 1, max_col + 1)
        })
    
    # sort patches by name
    patches.sort(key=lambda p: p["name"])

    return patches

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


def save_patches(patches, patch_height, patch_width):

    output_dir = Path(__file__).parent / "patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove all files in the output_dir before saving new patches
    for file in output_dir.glob("*.png"):
        file.unlink()

    ## SAVE PATCHES AS IMAGES

    for i, patch in enumerate(patches):
        patch_filename = output_dir / patch["name"]
        cv2.imwrite(str(patch_filename), patch["image"])

    ## SAVE ALL PATCH POSITIONS

    data = {
        "total_patches": len(patches),
        "patch_dimensions": {
            "height": patch_height,
            "width": patch_width
        },
        "patches": []
    }

    for i, patch in enumerate(patches):
        data["patches"].append({
            "patch_id": i,
            "name": patch["name"],
            "top_left_x": patch["position"][0],
            "top_left_y": patch["position"][1],
            "center_x": patch["center"][0],
            "center_y": patch["center"][1],
            "grid_row": patch["grid"][0],
            "grid_col": patch["grid"][1]
        })
    
    # Save patch positions as a JSON file
    positions_filename = output_dir / "all_patch_positions.json"
    with open(positions_filename, 'w') as f:
        json.dump(data, f, indent=2)



def main(patch: dict, draw_patches: bool = False):

    map_path = get_map_path()
    map_img = load_map_image(map_path)

    patches = split_map_into_patches(map_img, patch["height"], patch["width"], patch["stride_y"], patch["stride_x"])

    if draw_patches:
        grid_overlay = draw_patch_grid(map_img, patch["height"], patch["width"], patch["stride_y"], patch["stride_x"])
        cv2.imwrite(Path(__file__).parent / "patch_grid_overlay.png", grid_overlay)

    return patches

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

    patches = main(patch, draw_patches=True)

    save_patches(patches, patch["height"], patch["width"])
