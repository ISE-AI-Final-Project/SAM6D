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

if __name__ == "__main__":

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

    ran_dict = {"01":-1, "02":-1, "04":-1} #2:851, 4:773

    OBJ_TEMPLATE_FOLDER = (
        f"/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/linemod-ism-eval/templates/"
    )

    LINEMOD_FOLDER = "/home/icetenny/senior-1/Linemod_preprocessed/"

    OUTPUT_FOLDER = "/home/icetenny/senior-2/dataset/linemod_dg_v2"

    target_obj_list = [
        # "01",
        # "02",
        # "04",
        # "05",
        # "06",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]
    print(f"Object List includes {target_obj_list}")

    with torch.no_grad():
        # init sem ism
        sen_ism = SEN_ISM(
            config=config, device=device, path_parent="Instance_Segmentation_Model/"
        )

        for target_obj in target_obj_list:
            start_index = ran_dict.get(target_obj, 0)
            if start_index == -1:
                continue

            print(f"Inferencing {target_obj}")

            rgb_folder = os.path.join(LINEMOD_FOLDER, "data", target_obj, "rgb")
            depth_folder = os.path.join(LINEMOD_FOLDER, "data", target_obj, "depth")

            cam_info_path = os.path.join(LINEMOD_FOLDER, "data", target_obj, "info.yml")

            image_list = list(sorted(os.listdir(rgb_folder)))
            total_image = len(image_list)

            os.makedirs(os.path.join(OUTPUT_FOLDER, target_obj, "sam"), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_FOLDER, target_obj, "ours"), exist_ok=True)

            obj_template_dir = os.path.join(
                OBJ_TEMPLATE_FOLDER, target_obj, "templates"
            )

            cad_path = os.path.join(LINEMOD_FOLDER, "models", f"obj_{target_obj}.ply")

            # init template
            sen_ism.init_template(obj_template_dir=obj_template_dir, cad_path=cad_path)
            print(f"Init {target_obj} template finished.")

            for i, image_id in enumerate(image_list[start_index:]):
                print(f"\t{target_obj} : {i+start_index+1} / {total_image}")

                rgb_path = os.path.join(rgb_folder, image_id)
                depth_path = os.path.join(depth_folder, image_id)

                rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

                cam_batch_info = lm_batch_input_data(
                    depth_path=depth_path,
                    cam_path=cam_info_path,
                    device=device,
                    image_id=image_id.split(".")[0],
                )

                # Inference Normal ############################################
                normal_detections, rgb_segmentation = sen_ism.run_inference_no_depth(
                    rgb_img=rgb_img, batch_info=cam_batch_info, return_all=True
                )

                best_score_normal = np.argmax(normal_detections.scores)
                best_mask_normal = normal_detections.masks[best_score_normal].astype(
                    np.uint8
                )
                best_box_normal = normal_detections.boxes[best_score_normal]

                print(
                    f"\t\tBest Score Normal: {best_score_normal} at bbox: {best_box_normal}"
                )
                normal_write_path = os.path.join(
                    OUTPUT_FOLDER, target_obj, "sam", image_id
                )
                cv2.imwrite(normal_write_path, best_mask_normal * 255)
                torch.cuda.empty_cache()

                # print(normal_detections.masks.shape)
                # np.save('result_0000.npy', normal_detections.masks)

                # # Inference Ours ##############################################
                # detections, segmented_detections, depth_detections = (
                #     sen_ism.run_inference(
                #         rgb_img=rgb_img,
                #         depth_img=depth_img,
                #         batch_info=cam_batch_info,
                #         return_all=True,
                #     )
                # )

                # best_score = np.argmax(detections.scores)
                # best_mask = detections.masks[best_score].astype(np.uint8)
                # best_box = detections.boxes[best_score]

                # print(f"\t\tBest Score With Depth: {best_score} at bbox: {best_box}")

                # ours_write_path = os.path.join(
                #     OUTPUT_FOLDER, target_obj, "ours", image_id
                # )
                # cv2.imwrite(ours_write_path, best_mask * 255)

                # torch.cuda.empty_cache()


                # Inference Ours v2 ##############################################

                # print(normal_detections.masks.shape)
                detections = (
                    sen_ism.run_inference_depth_guide(
                        rgb_img=rgb_img,
                        depth_img=depth_img,
                        batch_info=cam_batch_info,
                        rgb_segmentation=normal_detections,
                        return_all=True,
                    )
                )

                best_score = np.argmax(detections.scores)
                best_mask = detections.masks[best_score].astype(np.uint8)
                best_box = detections.boxes[best_score]

                print(f"\t\tBest Score With Depth: {best_score} at bbox: {best_box}")

                ours_write_path = os.path.join(
                    OUTPUT_FOLDER, target_obj, "ours", image_id
                )
                cv2.imwrite(ours_write_path, best_mask * 255)

                torch.cuda.empty_cache()

