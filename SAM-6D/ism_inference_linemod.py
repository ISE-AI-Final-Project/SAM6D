import argparse
import glob
import logging
import os
import sys
import time

# Add the path to the Instance_Segmentation_Model directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Instance_Segmentation_Model"))

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

if __name__ == "__main__":

    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/inference/run_inference_linemod.yaml",
        help="Path to inference config yaml file",
    )

    args = parser.parse_args()

    config = load_yaml(args.config)

    # RPD
    # OBJ_TEMPLATE_DIR = "/workspace/SAM6D/SAM-6D/Data/templates/01/templates"
    # CAD_PATH = "/workspace/Linemod_preprocessed/models/obj_01.ply"

    # RGB_PATH = "/workspace/Linemod_preprocessed/data/01/rgb/0000.png"
    # DEPTH_PATH = "/workspace/Linemod_preprocessed/data/01/depth/0000.png"

    # CAM_INFO_PATH = "/workspace/Linemod_preprocessed/data/01/info.yml"

    OBJ_TEMPLATE_DIR = "/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/linemod-ism-eval/templates/01/templates"
    CAD_PATH = "/home/icetenny/senior-1/Linemod_preprocessed/models/obj_01.ply"

    RGB_PATH = "/home/icetenny/senior-1/Linemod_preprocessed/data/01/rgb/0000.png"
    DEPTH_PATH = "/home/icetenny/senior-1/Linemod_preprocessed//data/01/depth/0000.png"

    CAM_INFO_PATH = "/home/icetenny/senior-1/Linemod_preprocessed/data/01/info.yml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cam_batch_info = lm_batch_input_data(
        depth_path=DEPTH_PATH, cam_path=CAM_INFO_PATH, device=device, image_id="0000"
    )
    rgb_img = cv2.cvtColor(cv2.imread(RGB_PATH), cv2.COLOR_BGR2RGB)
    depth_img = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)

    # init sem ism
    sen_ism = SEN_ISM(
        config=config, device=device, path_parent="Instance_Segmentation_Model/"
    )

    # init template
    sen_ism.init_template(obj_template_dir=OBJ_TEMPLATE_DIR, cad_path=CAD_PATH)

    detections = sen_ism.run_inference(
        rgb_img=rgb_img, depth_img=depth_img, batch_info=cam_batch_info
    )

    print(detections.masks, detections.boxes, detections.boxes)

    # sen_ism.save_vis_image(rgb_img, detections, save_path="detection_ism.png")

    torch.cuda.empty_cache()

    #     # Create Folder
    # obj_output_dir = os.path.join(output_dir, "sam6d_results", obj_id)
    # if not os.path.exists(obj_output_dir):
    #     os.makedirs(obj_output_dir)

    # save_path = os.path.join(obj_output_dir, f"detection_ism_{image_id}")
