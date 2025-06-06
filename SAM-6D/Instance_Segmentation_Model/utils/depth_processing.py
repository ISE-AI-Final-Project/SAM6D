import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def intersec_mask_rgbd(detection_rgb, detection_depth):
    """
    Intersec RGB Detection with Depth Detection
    """

    mask_combined = torch.cat(
        (detection_rgb["masks"], detection_depth["masks"]), dim=0
    ).to(torch.bool)

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
        if torch.all(to_check.cpu() == 0):
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

                if intersec_mask_size >= 0.1 * mask_sizes.get(intersec_mask_id, 0):
                    blank_canva[blank_canva == intersec_mask_id] = insert_id
                    mask_sizes[intersec_mask_id.item()] = 0
                else:
                    mask_sizes[intersec_mask_id.item()] -= intersec_mask_size.item()

            blank_canva[mask] = insert_id
            mask_sizes[insert_id] = torch.sum(blank_canva == insert_id).item()

    detection = canva_to_detection(blank_canva)

    # save_mask_as_image(detection["masks"], "test.png")
    # save_mask_as_image(detection_rgb["masks"], "test_rgb.png")
    # save_mask_as_image(detection_depth["masks"], "test_d.png")

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


def depth_image_process_smooth(depth_image):
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

    smooth_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    smooth_kernel /= smooth_kernel.sum()

    # Apply the kernel to the inpainted image
    smoothed_image = cv2.filter2D(inpainted_image, -1, smooth_kernel)
    return smoothed_image


def get_adj_contour(mask):
    kernel = np.ones((3, 3), np.uint8)  # 3x3 structuring element
    offset_mask = cv2.dilate(mask, kernel, iterations=1)
    adj_mask = offset_mask - mask

    return adj_mask


def depth_guide_merge(depth_image, rgb_detection, device=0):
    processed_depth_image = depth_image_process_smooth(depth_image=depth_image)

    canva = np.zeros_like(depth_image).astype(np.uint16)
    detected_dict = dict()

    output = []

    for depth in range(255):
        depth_img_bin = cv2.inRange(processed_depth_image, depth, depth)

        num_labels, labels_im = cv2.connectedComponents(depth_img_bin)

        if num_labels <= 1:
            continue

        index, count = np.unique(labels_im, return_counts=True)

        passed_index, passed_count = index[count > 50][1:], count[count > 50][1:]

        if len(passed_index) < 1:
            continue

        new_canva = np.zeros_like(depth_image).astype(np.uint16)

        new_detected_dict = dict()

        for big_index, big_count in zip(passed_index, passed_count):

            big_index_mask = (labels_im == big_index).astype(np.uint16)
            adj_big_index_mask = get_adj_contour(big_index_mask)

            last_adj_index_list, last_adj_count_list = np.unique(
                canva[adj_big_index_mask == 1], return_counts=True
            )

            for last_adj_index, last_adj_count in zip(
                last_adj_index_list, last_adj_count_list
            ):
                if last_adj_index == 0:
                    continue

                if last_adj_count < adj_big_index_mask.sum() * 0.2:
                    continue

                if detected_dict[last_adj_index] == -1:  # Never Detect
                    detected_dict[last_adj_index] = big_index

                    new_canva[canva == last_adj_index] = big_index

                else:  # Already Detect
                    new_canva[canva == last_adj_index] = detected_dict[last_adj_index]

            new_canva[labels_im == big_index] = big_index
            new_detected_dict[big_index] = -1

        for detect_index, detect_value in detected_dict.items():
            if detect_value == -1:
                if (
                    np.sum(canva == detect_index)
                    > 0.003 * canva.shape[0] * canva.shape[1]
                ):
                    output.append((canva == detect_index).astype(np.uint16))

        canva = new_canva.copy()
        detected_dict = new_detected_dict.copy()

    # print(len(output))

    depth_bin = (np.asarray(output) > 0).astype(np.bool_)  # shape: (A, H, W)
    res_bin = (rgb_detection > 0).astype(np.bool_)  # shape: (B, H, W)

    # print(depth_bin.shape, res_bin.shape)

    # Expand dims for broadcasting
    depth_exp = depth_bin[:, np.newaxis, :, :]  # (A, 1, H, W)
    res_exp = res_bin[np.newaxis, :, :, :]  # (1, B, H, W)

    # Compute intersection and union
    intersection = np.logical_and(depth_exp, res_exp).sum(axis=(2, 3))  # (A, B)
    union = np.logical_or(depth_exp, res_exp).sum(axis=(2, 3))  # (A, B)

    # Compute IoU
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = intersection / union
        iou[union == 0] = 0

    # Create final fused masks
    depth_fused_output = []
    for i in range(iou.shape[0]):
        selected = iou[i] > 0.03  # select res masks with sufficient overlap
        if np.any(selected):
            combined_mask = np.any(res_bin[selected], axis=0).astype(np.uint8)
            if combined_mask.sum() > 0:
                depth_fused_output.append(combined_mask)

    # print(len(depth_fused_output))

    # Combine with original rgb_segmentation
    combined_segmentation = np.concatenate(
        [rgb_detection, np.array(depth_fused_output)]
    )

    masks_tensor = torch.tensor(combined_segmentation, device=device).to(torch.float32)

    # Convert to Detection
    boxes_list = []

    # Save each mask for each unique ID
    for mask in masks_tensor:
        nonzero_indices = torch.nonzero(mask)
        if nonzero_indices.size(0) > 0:  # Ensure there are non-zero points
            y_min, x_min = nonzero_indices.min(dim=0)[0]  # min of y and x
            y_max, x_max = nonzero_indices.max(dim=0)[0]  # max of y and x
            boxes_list.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])

    # Convert lists to tensors directly
    boxes_tensor = (
        torch.tensor(boxes_list, device=device)
        if boxes_list
        else torch.empty(0, 4, device=device)
    )

    return {"masks": masks_tensor, "boxes": boxes_tensor}


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
