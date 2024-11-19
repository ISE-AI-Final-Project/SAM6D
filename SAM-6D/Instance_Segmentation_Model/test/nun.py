import os
import time

# import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

start_time = time.time()


def intersec_mask_rgbd(detection_rgb, detection_depth):

    mask_combined = torch.cat((detection_rgb["masks"], detection_depth["masks"]), dim=0)

    blank_canva = torch.zeros_like(mask_combined[0], device=detection_rgb.device)

    mask_sizes = dict()

    for i, mask in enumerate(mask_combined):
        id = i + 1

        to_check = blank_canva[mask]
        unique_values, unique_counts = torch.unique(to_check, return_counts=True)
        mask_num_pixel = mask.sum()

        # Case 0: all 0
        if torch.all(to_check == 0):
            blank_canva[mask] = id
            mask_sizes[id] = mask_num_pixel.item()

        # Case 1: New all in one old  (small in big pass)
        elif len(unique_values) == 1:
            pass

        # # Case 2:
        else:
            insert_id = id

            # sort from large size to small
            sorted_counts, sorted_indices = torch.sort(unique_counts, descending=True)
            sorted_unique_values = unique_values[sorted_indices]

            for intersec_mask_id, intersec_mask_size in zip(
                sorted_unique_values, sorted_counts
            ):
                if intersec_mask_id == 0:
                    continue

                if intersec_mask_size >= 0.8 * mask_num_pixel:
                    insert_id = intersec_mask_id
                    continue

                if intersec_mask_size >= 0.2 * mask_sizes.get(intersec_mask_id, 0):
                    blank_canva[blank_canva == intersec_mask_id] = insert_id
                    mask_sizes[intersec_mask_id.item()] = 0
                else:
                    mask_sizes[intersec_mask_id.item()] -= intersec_mask_size.item()

            blank_canva[mask] = insert_id
            mask_sizes[insert_id.item()] = torch.sum(blank_canva == insert_id).item()

    detection = canva_to_detection(blank_canva)

    return detection


def canva_to_detection(blank_canva):

    # Get unique IDs in the canvas, excluding background (0)
    unique_ids = torch.unique(blank_canva)
    # unique_ids = unique_ids[unique_ids != 0]

    masks_list = []
    boxes_list = []

    # Save each mask for each unique ID
    for mask_id in unique_ids:
        if mask_id == 0:
            continue

        # Create a binary mask for the current group
        group_mask = (blank_canva == mask_id).to(
            torch.float32
        )  # Convert directly to float32 for consistency

        nonzero_indices = torch.nonzero(group_mask)
        if nonzero_indices.size(0) > 0:  # Ensure there are non-zero points
            y_min, x_min = nonzero_indices.min(dim=0)[0]  # min of y and x
            y_max, x_max = nonzero_indices.max(dim=0)[0]  # max of y and x
            boxes_list.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
            masks_list.append(group_mask)

    # Convert lists to tensors directly
    masks_tensor = (
        torch.stack(masks_list) if masks_list else torch.empty(0)
    )  # Create a tensor from masks list
    boxes_tensor = (
        torch.tensor(boxes_list, device=blank_canva.device)
        if boxes_list
        else torch.empty(0, 4, device=blank_canva.device)
    )
    # print(len(masks_tensor))
    # print(masks_tensor, len(masks_tensor))
    # print(boxes_tensor, len(boxes_tensor))
    return {"masks": masks_tensor, "boxes": boxes_tensor}


def rescale_depth(arr):
    # Find the minimum and maximum values
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Rescale to 0–255
    rescaled_arr = (arr - min_val) / (max_val - min_val) * 255

    # Convert to integer values if needed
    rescaled_arr = rescaled_arr.astype(np.uint8)

    return rescaled_arr


def depth_image_process(depth_image):

    # depth_image[depth_image == 0] = 10000
    depth_image = rescale_depth(depth_image)

    # depth_image = cv2.equalizeHist(depth_image)

    # Define the threshold to detect the darkest areas (shadows)
    # Threshold value 30 can be adjusted based on the shadow intensity
    _, shadow_mask = cv2.threshold(depth_image, 30, 255, cv2.THRESH_BINARY_INV)

    # Ensure the mask is in the correct format (8-bit single-channel)
    shadow_mask = np.uint8(shadow_mask)

    # Inpainting to fill in the shadow areas
    inpainted_image = cv2.inpaint(depth_image, shadow_mask, 0, cv2.INPAINT_NS)

    # Apply a sharpening filter to enhance the image's sharpness
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Apply the kernel to the inpainted image
    sharpened_image = cv2.filter2D(inpainted_image, -1, sharpening_kernel)

    return sharpened_image


start_time = time.time()

# Load the depth image in unchanged mode (grayscale depth data)
depth_image = cv2.imread(
    "/Users/nunny/Desktop/final project1/0000.png", cv2.IMREAD_UNCHANGED
)

# Run the processing function
processed_depth = depth_image_process(depth_image)

print(processed_depth.shape)
# Save the processed and sharpened image
inpainted_output_path = "/Users/nunny/Desktop/final project1/inpainted_NS4_image2.png"
# cv2.imwrite(inpainted_output_path, sharpened_image)
plt.imsave(inpainted_output_path, processed_depth, cmap="gray")
print("Inpainted image saved at:", inpainted_output_path)

end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")

rgb_mask_folder = "/workspace/nunny/run_intersect_numpy/01/0000/rgb_0000"
depth_mask_folder = "/workspace/nunny/run_intersect_numpy/01/0000/inpainted_NS_sharp"
group_folder_output = "/workspace/nunny/run_intersect_numpy/01/0000/group_masks"
combine_pic_path = (
    "/workspace/nunny/run_intersect_numpy/01/0000/blank_canva_combineAll.jpg"
)
blank_canva, mask_sizes, masks_tensor, boxes_tensor = (
    check_label_intersections_multiple_folder(
        [rgb_mask_folder, depth_mask_folder], combine_pic_path, group_folder_output
    )
)


end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")
