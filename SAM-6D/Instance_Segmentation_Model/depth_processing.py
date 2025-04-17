import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation, label


def separate_non_adjacent_objects_torch(binary_mask):
    """
    Separates non-adjacent objects in a binary mask.

    """
    binary_mask_np = binary_mask.cpu().numpy()
    labeled_mask, num_features = label(binary_mask_np)
    # Create a tensor of shape (N, H, W) where each slice corresponds to one object
    separated_masks = np.array([(labeled_mask == i) for i in range(1, num_features + 1)])
    separated_masks_tensor = torch.from_numpy(separated_masks).to(torch.bool) 
    return separated_masks_tensor


def intersec_mask_rgbd(detection_rgb, detection_depth):
    """
    Intersect RGB Detection with RGB Mask
    """

    mask_combined = detection_rgb["masks"].to(torch.bool)
    blank_canva = torch.zeros_like(mask_combined[0], device=mask_combined.device).to(
        torch.int64
    )

    mask_sizes = dict()
    id = 1
    for i, ori_mask in enumerate(mask_combined):
        separated_masks_tensor = separate_non_adjacent_objects_torch(ori_mask)
        for mask in separated_masks_tensor:
            mask = mask.to(mask_combined.device)            
            id = id + 1

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

        valid_overlapping_dict = {}
        for i, overlap_id in enumerate(overlapping_ids):
            if overlap_id > 0:  # Exclude background ID (0)
                overlap_percentage = counts[i] / mask.sum()  # Calculate overlap percentage
                if overlap_percentage >= 0.12:  # Check threshold (10%)
                    valid_overlapping_dict[overlap_id] = overlap_percentage

        if valid_overlapping_dict:
            # Check if any valid ID already has a group
            existing_groups = [
                group_id[valid_id]['group'] for valid_id in valid_overlapping_dict.keys() if valid_id in group_id
            ]
            if existing_groups:
                # Use the first existing group and merge others
                new_group = existing_groups[0]
                merge_groups(group_id, valid_overlapping_dict.keys(), new_group)
            else:
                new_group = f"group_{j}"
                merge_groups(group_id, valid_overlapping_dict.keys(), new_group)
                j += 1

            # Update the group_id with overlap percentages
            for valid_id, overlap_percentage in valid_overlapping_dict.items():
                group_id[valid_id] = {
                    "group": new_group,
                    "overlap_percentage": overlap_percentage
                }
    
    temp_group = {}
    final_group = {}

    # Organize group_id by their groups
    group_to_ids = {}
    for idx, data in group_id.items():
        group_name = data['group']
        if group_name not in group_to_ids:
            group_to_ids[group_name] = []
        group_to_ids[group_name].append(idx)

    # print("group_to_ids")
    # print(group_to_ids)

    # Iterate through each group
    for group_name, ids in group_to_ids.items():
        temp_group[group_name] = []

        # Process each ID in the group
        while ids:
            current_id = ids.pop(0)
            added_to_existing_list = False

            # Check adjacency with existing sublists
            for sublist in temp_group[group_name]:
                for id_in_list in sublist:
                    if are_adjacent((blank_canva == id_in_list), (blank_canva == current_id)):
                        # print(f"Adjacent: {id_in_list} and {current_id}")
                        sublist.append(current_id)
                        added_to_existing_list = True
                        break
                    # else:
                    #     print(f"Not Adjacent: {id_in_list} and {current_id}")
                if added_to_existing_list:
                    break

            # If not added to any existing sublist, create a new one
            if not added_to_existing_list:
                temp_group[group_name].append([current_id])

        # Finalize the group: Keep the sublist with the highest overlap percentage
        max_overlap_mask = max(
            [idx for sublist in temp_group[group_name] for idx in sublist],
            key=lambda idx: group_id[idx]['overlap_percentage']
        )
        # Find the sublist containing the max_overlap_mask
        final_group[group_name] = next(
            sublist for sublist in temp_group[group_name] if max_overlap_mask in sublist
        )

    # # print("final_group")
    # print(final_group)

    for group, ids in final_group.items():
        # print(f"Combining masks for {group}: {ids}")
        combined_mask = torch.zeros_like(blank_canva, dtype=bool)
        for obj_id in ids:
            combined_mask |= (blank_canva == obj_id)

        # Assign a single ID (e.g., the first ID in the group) to the combined mask
        new_id = ids[0]
        blank_canva[combined_mask] = new_id

    all_detection = canva_to_detection(blank_canva)  
    final_detection = union_detections(detection_rgb, all_detection)
    return final_detection


def are_adjacent(mask1, mask2):
    mask1_np = mask1.cpu().numpy() if mask1.is_cuda else mask1.numpy()
    mask2_np = mask2.cpu().numpy() if mask2.is_cuda else mask2.numpy()
    dilated_mask1 = binary_dilation(mask1_np)
    return np.any(dilated_mask1 & mask2_np)


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
            if y_max - y_min <= 0 or x_max - x_min <= 0:
                continue
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


def union_detections(detection_rgb, all_detection):
    """
    Unions the masks and boxes from detection_rgb and all_detection,
    skipping those in all_detection that already exist in detection_rgb.
    
    """
    # Get masks and boxes from both detections
    masks_rgb = detection_rgb["masks"]
    boxes_rgb = detection_rgb["boxes"]
    masks_final = all_detection["masks"]
    boxes_final = all_detection["boxes"]

    # Start with RGB detection masks and boxes
    combined_masks = [masks_rgb]
    combined_boxes = [boxes_rgb]

    # Skip duplicates
    for i, mask_final in enumerate(masks_final):
        is_duplicate = False
        for mask_rgb in masks_rgb:
            if torch.equal(mask_rgb, mask_final):
                is_duplicate = True
                break

        if not is_duplicate:
            # Add non-duplicate mask and corresponding box
            combined_masks.append(mask_final.unsqueeze(0))  # Add as a single slice
            combined_boxes.append(boxes_final[i].unsqueeze(0))  # Add as a single slice

    # Stack all combined masks and boxes
    combined_masks_tensor = torch.cat(combined_masks, dim=0) if combined_masks else torch.empty(0, *masks_rgb.shape[1:], dtype=torch.bool, device=masks_rgb.device)
    combined_boxes_tensor = torch.cat(combined_boxes, dim=0) if combined_boxes else torch.empty(0, 4, dtype=torch.float32, device=boxes_rgb.device)

    # Return the final union detection
    return {"masks": combined_masks_tensor, "boxes": combined_boxes_tensor}



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
