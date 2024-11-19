import os
import numpy as np
import cv2
import torch
import time
start_time = time.time()

def intersec_mask_rgbd(detection_rgb, detection_depth):

    mask_combined = torch.cat((detection_rgb['masks'], detection_depth['masks'), dim=0)

    blank_canva = np.zeros_like(cv2.imread(all_image_files[0], cv2.IMREAD_GRAYSCALE))
    mask_sizes = dict()

    for i, image_file in enumerate(all_image_files):
        image1 = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        mask = (image1 > 0)
        id = i + 1

        to_check = blank_canva[mask]
        unique_to_check = np.unique(to_check, return_counts=True)
        mask_num_pixel = mask.sum()

        # Case 0: all 0
        if np.all(to_check ==0):
            blank_canva[mask] = id
            mask_sizes[id] = mask_num_pixel

        # Case 1: New all in one old  (small in big pass)
        elif len(unique_to_check[0]) == 1:
            pass

        # # Case 2:
        else:
            insert_id = id

            unique_values, counts = unique_to_check

            #sort from large size to small
            sorted_indices = np.argsort(-counts)
            sorted_unique_values = unique_values[sorted_indices]
            sorted_counts = counts[sorted_indices]

            for intersec_mask_id, intersec_mask_size in zip(sorted_unique_values, sorted_counts):
                if intersec_mask_id == 0:
                    continue

                if intersec_mask_size >= 0.8 * mask_num_pixel:
                    insert_id = intersec_mask_id
                    continue

                if intersec_mask_size >= 0.2 * mask_sizes.get(intersec_mask_id):
                    blank_canva[blank_canva == intersec_mask_id] = insert_id
                    mask_sizes[intersec_mask_id] = 0
                else:
                    mask_sizes[intersec_mask_id] -= intersec_mask_size

            blank_canva[mask] = insert_id
            mask_sizes[insert_id] = np.sum(blank_canva == insert_id)
    # print(np.unique(blank_canva, return_counts=True), len(np.unique(blank_canva, return_counts=True)[0]))
    # print(mask_sizes)
    cv2.imwrite(output_path, blank_canva)

    masks_tensor, boxes_tensor = save_group_masks(blank_canva, group_folder)
    return blank_canva,mask_sizes, masks_tensor, boxes_tensor
    
def save_group_masks(blank_canva, group_folder):
    os.makedirs(group_folder, exist_ok=True)
    
    # Get unique IDs in the canvas, excluding background (0)
    unique_ids = np.unique(blank_canva)
    # unique_ids = unique_ids[unique_ids != 0]
    
    masks_list = []
    boxes_list = []
    
    # Save each mask for each unique ID
    for mask_id in unique_ids:
        # Create a binary mask for the current group
        group_mask = (blank_canva == mask_id).astype(np.float32)  # Convert directly to float32 for consistency
        masks_list.append(group_mask)
       
        # Find bounding box coordinates
        y_indices, x_indices = np.where(group_mask)
        x_min = x_indices.min()
        x_max = x_indices.max()
        y_min = y_indices.min()
        y_max = y_indices.max()
        boxes_list.append([x_min, y_min, x_max, y_max])
        
        # Save the mask as an image (optional)
        output_path = os.path.join(group_folder, f'group_mask_{mask_id}.png')
        cv2.imwrite(output_path, (group_mask * 255).astype(np.uint8))  # Scale to 255 for visibility
    
    # Convert lists to numpy arrays before tensor conversion
    masks_array = np.array(masks_list)  # Convert list to numpy array first
    boxes_array = np.array(boxes_list)

    # Convert numpy arrays to tensors and use CPU as the device
    masks_tensor = torch.tensor(masks_array, device='cuda:0')
    boxes_tensor = torch.tensor(boxes_array, device='cuda:0')
    # print(len(masks_tensor))
    # print(masks_tensor, len(masks_tensor))
    # print(boxes_tensor, len(boxes_tensor))
    return masks_tensor, boxes_tensor

rgb_mask_folder = '/workspace/nunny/run_intersect_numpy/01/0000/rgb_0000'  
depth_mask_folder = '/workspace/nunny/run_intersect_numpy/01/0000/inpainted_NS_sharp' 
group_folder_output = "/workspace/nunny/run_intersect_numpy/01/0000/group_masks"
combine_pic_path = '/workspace/nunny/run_intersect_numpy/01/0000/blank_canva_combineAll.jpg'
blank_canva,mask_sizes, masks_tensor, boxes_tensor = check_label_intersections_multiple_folder(
                    [rgb_mask_folder, 
                    depth_mask_folder], 
                    combine_pic_path, 
                    group_folder_output)


end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")    