import argparse
import glob
import logging
import os
import sys
import time

# Add the path to the Instance_Segmentation_Model directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Instance_Segmentation_Model"))

import random

import cv2
import distinctipy
import imageio
import numpy as np
import torch
import trimesh
import yaml
from omegaconf import OmegaConf
from PIL import Image
from sen_ism_inferencer import SEN_ISM, lm_batch_input_data, load_yaml


def batch_input_data(depth, K, device):
    """
    Get Batch Input info to be use for inference

    Input:
        depth: int32 np array
        K: camera K list of len 9
        device: cuda device
    """

    batch = {}
    # depth = np.array(imageio.v2.imread(depth_path)).astype(np.int32)
    cam_K = np.array(K).reshape((3, 3))
    depth_scale = np.array(1.0)

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch["depth_scale"] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch


def overlay_masks_boxes(image, masks, bboxes, scores, score_threshold=0.5):
    image = image.copy()

    for i in range(len(masks)):
        if scores[i] < score_threshold:
            continue

        mask = masks[i].astype(np.uint8)
        color = [random.randint(0, 255) for _ in range(3)]  # Random color

        # Create colored mask overlay
        colored_mask = np.zeros_like(image)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]

        # Blend mask with image
        image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bboxes[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Put score text
        label = f"{scores[i]:.2f}"
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return image

def overlay_masks_boxes_no_scores(image, masks, bboxes):
    image = image.copy()

    for i in range(len(masks)):
        mask = masks[i].astype(np.uint8)
        color = [random.randint(0, 255) for _ in range(3)]  # Random color

        # Create colored mask overlay
        colored_mask = np.zeros_like(image)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]

        # Blend mask with image
        image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bboxes[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image


if __name__ == "__main__":

    K_1080 = [
        547.7678833007812,
        0.0,
        477.81231689453125,
        0.0,
        547.7678833007812,
        271.7215270996094,
        0.0,
        0.0,
        1.0,
    ]

    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/inference/run_inference_linemod.yaml",
        help="Path to inference config yaml file",
    )

    args = parser.parse_args()

    config = load_yaml(args.config)

    TARGET_OBJ = "purple"

    OBJ_TEMPLATE_DIR = f"/home/icetenny/senior-2/senior_dataset/{TARGET_OBJ}/templates"
    CAD_PATH = (
        f"/home/icetenny/senior-2/senior_dataset/{TARGET_OBJ}/{TARGET_OBJ}_centered.ply"
    )

    RGB_PATH = "/home/icetenny/senior-2/results/run_1744009981830583473/1744010291531799752_rgb.png"
    DEPTH_PATH = "/home/icetenny/senior-2/results/run_1744009981830583473/1744010291531799752_depth.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rgb_img = cv2.cvtColor(cv2.imread(RGB_PATH), cv2.COLOR_BGR2RGB)
    depth_img = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)

    depth_img[depth_img > 2000] = 0

    cam_batch_info = batch_input_data(
        depth=np.array(imageio.v2.imread(DEPTH_PATH)).astype(np.int32),
        K=K_1080,
        device=device,
    )

    # init sem ism
    sen_ism = SEN_ISM(
        config=config, device=device, path_parent="Instance_Segmentation_Model/"
    )

    # init template
    sen_ism.init_template(obj_template_dir=OBJ_TEMPLATE_DIR, cad_path=CAD_PATH)

    detections, segmented_detections, depth_detections = sen_ism.run_inference(
        rgb_img=rgb_img, depth_img=depth_img, batch_info=cam_batch_info, return_all=True
    )

    # print(detections.masks, detections.boxes, detections.scores)

    # sen_ism.save_vis_image(rgb_img, detections, save_path="detection_ism.png")

    print(segmented_detections.masks.shape, depth_detections.masks.shape, detections.masks.shape)


    output_segmented_image = overlay_masks_boxes_no_scores(
        image=cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
        masks=segmented_detections.masks,
        bboxes=segmented_detections.boxes,
    )

    cv2.imwrite(
        "/home/icetenny/senior-2/results/run_1744009981830583473/1744010291531799752_all_mask_rgb.png",
        output_segmented_image,
    )

    output_depth_segmented_image = overlay_masks_boxes_no_scores(
        image=cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
        masks=depth_detections.masks,
        bboxes=depth_detections.boxes,
    )

    cv2.imwrite(
        "/home/icetenny/senior-2/results/run_1744009981830583473/1744010291531799752_all_mask_depth.png",
        output_depth_segmented_image,
    )

    output_combined_segmented_image = overlay_masks_boxes_no_scores(
        image=cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
        masks=detections.masks,
        bboxes=detections.boxes,
    )

    cv2.imwrite(
        "/home/icetenny/senior-2/results/run_1744009981830583473/1744010291531799752_all_mask_combined.png",
        output_combined_segmented_image,
    )
    torch.cuda.empty_cache()


    output_image = overlay_masks_boxes(
        image=cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),
        masks=detections.masks,
        bboxes=detections.boxes,
        scores=detections.scores,
        score_threshold=0,
    )
    cv2.imwrite(
        f"/home/icetenny/senior-2/results/run_1744009981830583473/1744010291531799752_results_{TARGET_OBJ}.png",
        output_image,
    )


    best_score = np.argmax(detections.scores)
    best_mask = detections.masks[best_score]

    cv2.imwrite(
        f"/home/icetenny/senior-2/results/run_1744009981830583473/1744010291531799752_best_{TARGET_OBJ}.png",
        best_mask.astype(np.uint8) * 255,
    )


    #     # Create Folder
    # obj_output_dir = os.path.join(output_dir, "sam6d_results", obj_id)
    # if not os.path.exists(obj_output_dir):
    #     os.makedirs(obj_output_dir)

    # save_path = os.path.join(obj_output_dir, f"detection_ism_{image_id}")
