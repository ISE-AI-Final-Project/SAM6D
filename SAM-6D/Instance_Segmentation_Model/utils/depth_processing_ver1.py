import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def intersec_mask_rgbd(detection_rgb, detection_depth):
    """
    Intersect RGB Detection with RGB Mask
    """

    mask_combined = detection_rgb["masks"].to(torch.bool)

    blank_canva = torch.zeros_like(mask_combined[0], device=mask_combined.device).to(
        torch.int64
    )

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
        elif len(unique_values) == 1 or len(unique_values) >= 4:
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
                    insert_id = intersec_mask_id.item()
                    continue

                if intersec_mask_size >= 0.2 * mask_sizes.get(intersec_mask_id, 0):
                    blank_canva[blank_canva == intersec_mask_id] = insert_id
                    mask_sizes[intersec_mask_id.item()] = 0
                else:
                    mask_sizes[intersec_mask_id.item()] -= intersec_mask_size.item()

            blank_canva[mask] = insert_id
            mask_sizes[insert_id] = torch.sum(blank_canva == insert_id).item()
    
    """
    Intersect RGB Detection with Depth Mask
    """

    mask_depth_combined = detection_depth["masks"].to(torch.bool)

    group_id = dict()
    j = 1
    for i, mask in enumerate(mask_depth_combined):

        # Get IDs overlapping with the current depth mask
        overlapping_ids, counts = torch.unique(blank_canva[mask], return_counts=True)
        # Filter overlapping IDs based on a 20% threshold
        valid_overlapping_ids = []
        for i, overlap_id in enumerate(overlapping_ids):
            if overlap_id > 0:  # Exclude background ID (0)
                overlap_percentage = counts[i] / mask.sum()  # Calculate overlap percentage
                if overlap_percentage >= 0.12:  # Check threshold (10%)
                    valid_overlapping_ids.append(overlap_id)

        if valid_overlapping_ids:
        # Check if any valid ID already has a group
            existing_groups = [group_id[valid_id] for valid_id in valid_overlapping_ids if valid_id in group_id]
            if existing_groups:
                # Use the first existing group and merge others
                new_group = existing_groups[0]
                merge_groups(group_id, valid_overlapping_ids, new_group)
            else:
                new_group = f"group_{j}"
                merge_groups(group_id, valid_overlapping_ids, new_group)
                j += 1
    group_to_ids = {}
    for obj_id, group in group_id.items():
        if group not in group_to_ids:
            group_to_ids[group] = []
        group_to_ids[group].append(obj_id)

    # Combine masks in blank_canva for each group
    for group, ids in group_to_ids.items():
        combined_mask = torch.zeros_like(blank_canva, dtype=bool)
        for obj_id in ids:
            combined_mask |= (blank_canva == obj_id)

        # Assign a single ID (e.g., the first ID in the group) to the combined mask
        new_id = ids[0]
        blank_canva[combined_mask] = new_id

    detection = canva_to_detection(blank_canva)  
    return detection


def merge_groups(group_id, valid_ids, new_group):
    """
    Ensures that all IDs in valid_ids are assigned to the same group.
    If an ID already belongs to a group, that group is merged with new_group.
    """
    for valid_id in valid_ids:
        if valid_id in group_id:
            existing_group = group_id[valid_id]
            # Reassign all IDs in existing_group to new_group
            for key, value in group_id.items():
                if value == existing_group:
                    group_id[key] = new_group
        else:
            group_id[valid_id] = new_group



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
    """
    One channel depth image to processed RGB image
    """

    # Rescale Depth
    depth_image = rescale_depth(depth_image)

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

    return cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR)


def save_mask_as_image(mask_tensor, path):
    # Create a blank canvas to overlay the masks
    canvas = np.zeros((mask_tensor.shape[1], mask_tensor.shape[2], 3), dtype=np.uint8)

    # Assign unique colors to each mask
    colors = [
        (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )
        for _ in range(mask_tensor.shape[0])
    ]

    # Overlay each mask on the canvas
    for i in range(mask_tensor.shape[0]):
        mask = mask_tensor[i].cpu().numpy()
        # Add the mask with its unique color
        canvas[mask == 1] = colors[i]

    cv2.imwrite(path, canvas)

    print(f"Image saved at {path}")


if __name__ == "__main__":

    start_time = time.time()

    depth_path = os.path.join(
        "/home/icetenny/senior-1/Linemod_preprocessed/data",
        "01",
        "depth",
        f"{'0000'}.png",
    )
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    print(depth_image.shape)
    # Run the processing function
    processed_depth = depth_image_process(depth_image)

    print(processed_depth.shape)

    # Save the processed and sharpened image
    # inpainted_output_path = "inpainted_NS4_image2.png"
    # cv2.imwrite(inpainted_output_path, sharpened_image)
    # print("Inpainted image saved at:", inpainted_output_path)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")

    cv2.imshow("hi", processed_depth)
    cv2.waitKey()
    cv2.destroyAllWindows()
