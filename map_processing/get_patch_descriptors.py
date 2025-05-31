
import cv2

def main(patches, patches_paths):

    orb = cv2.ORB_create(nfeatures=1000)

    patch_descriptors = []

    for i, patch_img in enumerate(patches):

        keypoints, descriptors = orb.detectAndCompute(patch_img, None)
        patch_descriptors.append({
            "keypoints": keypoints,
            "descriptors": descriptors,
            "name": patches_paths[i].name
        })
