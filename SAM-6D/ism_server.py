import argparse
import glob
import logging
import os
import sys
import time

from my_custom_socket import MyServer

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
        default="Instance_Segmentation_Model/configs/inference/run_inference_sam.yaml",
        help="Path to inference config yaml file",
    )

    args = parser.parse_args()
    config = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init sem ism
    sen_ism = SEN_ISM(
        config=config, device=device, path_parent="Instance_Segmentation_Model/"
    )
    # print(sen_ism)
    # print("HII")

    # Init server
    server = MyServer(host="127.0.0.1", port=11111, server_name="ISM Server")
    server.start()

    while True:
        # Wait for msg
        recv_msg = server.wait_for_msg()
        if recv_msg is not None:
            # Receive Message
            rgb_img, depth_img, target_obj, dataset_path_prefix = recv_msg

            obj_template_dir = os.path.join(dataset_path_prefix, target_obj, "templates")

            cad_path = os.path.join(
                dataset_path_prefix, target_obj, f"{target_obj}_centered.ply"
            )

            cam_batch_info = batch_input_data(depth=depth_img, K=K_1080, device=device)

            # Clip Depth for better depth segment
            depth_img[depth_img > 2000] = 0

            print(f"Received RGB Image with shape: {rgb_img.shape}")
            print(f"Received Depth Image with shape: {depth_img.shape}")
            print(f"Finding: {target_obj}")

            # init template
            sen_ism.init_template(obj_template_dir=obj_template_dir, cad_path=cad_path)

            # Inference
            # detections = sen_ism.run_inference(
            #     rgb_img=rgb_img, depth_img=depth_img, batch_info=cam_batch_info
            # )

            detections, segmented_detections, depth_detections = sen_ism.run_inference(
                rgb_img=rgb_img,
                depth_img=depth_img,
                batch_info=cam_batch_info,
                return_all=True,
            )

            print(
                segmented_detections.masks.shape,
                depth_detections.masks.shape,
                detections.masks.shape,
            )

            output_combined_segmented_image = overlay_masks_boxes(
                image=rgb_img,
                masks=detections.masks,
                bboxes=detections.boxes,
                scores=detections.scores,
                score_threshold=0,
            )


            best_score = np.argmax(detections.scores)
            best_mask = detections.masks[best_score].astype(np.uint8)
            best_box = detections.boxes[best_score]

            print(f"Best Score: {best_score} at bbox: {best_box}")
            print(f"Best Mask type: {best_mask.dtype}")

            # Send response
            server.send_response(
                msg_type_out=["numpyarray", "numpyarray", "float", "numpyarray"],
                msg_out=[
                    best_mask,
                    best_box,
                    best_score,
                    output_combined_segmented_image,
                ],
            )

            print(f"[{server.server_name}] Response Sent. Restarting.")
            server.restart()

            torch.cuda.empty_cache()
        else:
            print(f"[{server.server_name}] Connection Lost. Restarting.")
            server.restart()
