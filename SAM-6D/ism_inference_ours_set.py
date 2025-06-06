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

    # K_1080 = [
    #     547.7678833007812,
    #     0.0,
    #     477.81231689453125,
    #     0.0,
    #     547.7678833007812,
    #     271.7215270996094,
    #     0.0,
    #     0.0,
    #     1.0,
    # ]

    # K_2K = [
    #     542.290771484375,
    #     0.0,
    #     549.8480834960938,
    #     0.0,
    #     542.290771484375,
    #     312.20489501953125,
    #     0.0,
    #     0.0,
    #     1.0,
    # ]

    K_Fused = [
        541.045166015625,
        0.0,
        549.8463745117188,
        0.0,
        541.045166015625,
        312.2047119140625,
        0.0,
        0.0,
        1.0
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


    ran_dict = {
        "acnewash": -1,
        "sunscreen": -1,
        "lactose": -1,
        "orange": -1,
        "purple": 24,
        # "contactcleaning": -1,
        # "cereal": 103,
    }

    OBJ_TEMPLATE_FOLDER = f"/home/icetenny/senior-2/senior_dataset/"

    RGB_FOLDER = "/home/icetenny/senior-2/results/senior_dataset_fuse_0205/rgb"
    DEPTH_FOLDER = "/home/icetenny/senior-2/results/senior_dataset_fuse_0205/depth"
    DEPTH_FUSED_FOLDER = "/home/icetenny/senior-2/results/senior_dataset_fuse_0205/depth_fused"
    OUTPUT_FOLDER = "/home/icetenny/senior-2/dataset/inference-fused"

    with torch.no_grad():
        # init sem ism
        sen_ism = SEN_ISM(
            config=config, device=device, path_parent="Instance_Segmentation_Model/"
        )

        target_obj_list = [i for i in os.listdir(OBJ_TEMPLATE_FOLDER) if "." not in i]
        image_list = list(sorted(os.listdir(RGB_FOLDER)))
        total_image = len(image_list)

        print(f"Object List includes {target_obj_list}")

        for target_obj in target_obj_list:
            start_index = ran_dict.get(target_obj, 0)
            if start_index == -1:
                continue

            print(f"Inferencing {target_obj}")

            os.makedirs(os.path.join(OUTPUT_FOLDER, target_obj, "sam"), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_FOLDER, target_obj, "ours"), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_FOLDER, target_obj, "ours_fused"), exist_ok=True)

            obj_template_dir = os.path.join(OBJ_TEMPLATE_FOLDER, target_obj, "templates")

            cad_path = f"/home/icetenny/senior-2/senior_dataset/{target_obj}/{target_obj}_centered.ply"
            # init template
            sen_ism.init_template(obj_template_dir=obj_template_dir, cad_path=cad_path)
            print(f"Init {target_obj} template finished.")

            for i, image_id in enumerate(image_list[start_index:]):
                print(f"\t{target_obj} : {i+start_index+1} / {total_image}")

                rgb_path = os.path.join(RGB_FOLDER, image_id)
                depth_path = os.path.join(DEPTH_FOLDER, image_id)
                depth_fused_path = os.path.join(DEPTH_FUSED_FOLDER, image_id)

                rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_fused_img = cv2.imread(depth_fused_path, cv2.IMREAD_UNCHANGED)

                depth_img[depth_img > 2000] = 0
                depth_fused_img[depth_fused_img > 2000] = 0

                cam_batch_info = batch_input_data(
                    depth=np.array(imageio.v2.imread(depth_path)).astype(np.int32),
                    K=K_Fused,
                    device=device,
                )

                cam_batch_info_fused = batch_input_data(
                    depth=np.array(imageio.v2.imread(depth_fused_path)).astype(np.int32),
                    K=K_Fused,
                    device=device,
                )


                # Inference Normal
                normal_detections = sen_ism.run_inference_no_depth(
                    rgb_img=rgb_img, batch_info=cam_batch_info
                )

                best_score_normal = np.argmax(normal_detections.scores)
                best_mask_normal = normal_detections.masks[best_score_normal].astype(
                    np.uint8
                )
                best_box_normal = normal_detections.boxes[best_score_normal]

                print(
                    f"\t\tBest Score Normal: {best_score_normal} at bbox: {best_box_normal}"
                )
                normal_write_path = os.path.join(OUTPUT_FOLDER, target_obj, "sam", image_id)
                cv2.imwrite(normal_write_path, best_mask_normal * 255)
                torch.cuda.empty_cache()


                # Inference Ours
                detections, segmented_detections, depth_detections = sen_ism.run_inference(
                    rgb_img=rgb_img,
                    depth_img=depth_img,
                    batch_info=cam_batch_info,
                    return_all=True,
                )

                best_score = np.argmax(detections.scores)
                best_mask = detections.masks[best_score].astype(np.uint8)
                best_box = detections.boxes[best_score]

                print(f"\t\tBest Score With Depth: {best_score} at bbox: {best_box}")

                ours_write_path = os.path.join(OUTPUT_FOLDER, target_obj, "ours", image_id)
                cv2.imwrite(ours_write_path, best_mask * 255)

                torch.cuda.empty_cache()

                # # Inference Ours: Depth Fused
                # detections_fused, segmented_detections_fused, depth_detections_fused = sen_ism.run_inference(
                #     rgb_img=rgb_img,
                #     depth_img=depth_fused_img,
                #     batch_info=cam_batch_info_fused,
                #     return_all=True,
                # )

                # best_score_fused = np.argmax(detections_fused.scores)
                # best_mask_fused = detections.masks[best_score_fused].astype(np.uint8)
                # best_box_fused = detections.boxes[best_score_fused]

                # print(f"\t\tBest Score With Depth Fused: {best_score_fused} at bbox: {best_box_fused}")

                # fused_write_path = os.path.join(OUTPUT_FOLDER, target_obj, "ours_fused", image_id)
                # cv2.imwrite(fused_write_path, best_mask_fused * 255)

                # torch.cuda.empty_cache()
