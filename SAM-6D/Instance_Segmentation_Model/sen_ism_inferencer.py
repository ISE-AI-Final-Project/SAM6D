import argparse
import glob
import logging
import os
import sys
import time

import cv2
import distinctipy
import imageio
import numpy as np
import torch
import trimesh
import yaml
from hydra import compose, initialize
from hydra.utils import instantiate
from model.utils import Detections, convert_npz_to_json
from omegaconf import OmegaConf
from PIL import Image
from segment_anything.utils.amg import rle_to_mask
from skimage.feature import canny
from skimage.morphology import binary_dilation
from utils.bbox_utils import CropResizePad
from utils.depth_processing import (
    depth_guide_merge,
    depth_image_process,
    intersec_mask_rgbd,
)
from utils.poses.pose_utils import (
    get_obj_poses_from_template_level,
    load_index_level_in_level2,
)

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "Instance_Segmentation_Model")

sys.path.append(os.path.join(ROOT_DIR, "provider"))
sys.path.append(os.path.join(ROOT_DIR, "segment_anything"))
sys.path.append(os.path.join(ROOT_DIR, "model"))
sys.path.append(os.path.join(ROOT_DIR, "model", "layers"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "utils", "poses"))
sys.path.append(os.path.join(ROOT_DIR, "utils", "poses", "predefined_poses"))


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    start = time.time()  # time estimate start

    def show(j):
        x = int(size * j / count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60)  # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(
            f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}",
            end="\r",
            file=out,
            flush=True,
        )

    show(0.1)  # avoid div/0
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def lm_batch_input_data(depth_path, cam_path, device, image_id="0001"):
    """
    Get LM Batch Input info to be use for inference

    Input:
        cam_path: path to cam yaml file
    """

    batch = {}
    cam_dict = load_yaml(cam_path)
    depth = np.array(imageio.v2.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_dict[int(image_id)]["cam_K"]).reshape((3, 3))
    depth_scale = np.array(cam_dict[int(image_id)]["depth_scale"])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch["depth_scale"] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch


class SEN_ISM:
    # Senior ISM Model

    def __init__(self, config, device, path_parent=""):
        self.device = device
        self.model = self.init_sam6d(
            config["SEGMENTOR_MODEL"],
            stability_score_thresh=config["STABILITY_SCORE_THRESH"],
            path_parent=path_parent,
        )

    def init_sam6d(self, segmentor_model, stability_score_thresh, path_parent=""):
        # Init SAM 6D Model
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="run_inference.yaml")

        if segmentor_model == "sam":
            with initialize(version_base=None, config_path="configs/model"):
                cfg.model = compose(config_name="ISM_sam.yaml")
            cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh

            if path_parent != "":
                cfg.model.segmentor_model.sam.checkpoint_dir = (
                    cfg.model.segmentor_model.sam.checkpoint_dir.replace(
                        "./", path_parent
                    )
                )
                cfg.model.descriptor_model.checkpoint_dir = (
                    cfg.model.descriptor_model.checkpoint_dir.replace("./", path_parent)
                )

        elif segmentor_model == "fastsam":
            with initialize(version_base=None, config_path="configs/model"):
                cfg.model = compose(config_name="ISM_fastsam.yaml")

            if path_parent != "":
                cfg.model.segmentor_model.fastsam.checkpoint_dir = (
                    cfg.model.segmentor_model.fastsam.checkpoint_dir.replace(
                        "./", path_parent
                    )
                )
                cfg.model.descriptor_model.checkpoint_dir = (
                    cfg.model.descriptor_model.checkpoint_dir.replace("./", path_parent)
                )
        else:
            raise ValueError(
                "The segmentor_model {} is not supported now!".format(segmentor_model)
            )

        logging.info("Initializing model")
        model = instantiate(cfg.model)

        model.descriptor_model.model = model.descriptor_model.model.to(self.device)
        model.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(model.segmentor_model, "predictor"):
            model.segmentor_model.predictor.model = (
                model.segmentor_model.predictor.model.to(self.device)
            )
        else:
            model.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

        return model

    def init_template(self, obj_template_dir, cad_path):
        """
        Init Template

        input
            obj_template_dir: path to folder contains mask_{}.png, rgb_{}.png, xyz_{}.npz
            cad_path: path to cad (.ply)
        """
        # Init Template with obj id
        logging.info(f"Initializing template from:{obj_template_dir}")
        num_templates = len(glob.glob(f"{obj_template_dir}/*.npy"))
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(
                os.path.join(obj_template_dir, "rgb_" + str(idx) + ".png")
            )
            mask = Image.open(
                os.path.join(obj_template_dir, "mask_" + str(idx) + ".png")
            )
            boxes.append(mask.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))

        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)

        self.model.ref_data = {}
        self.model.ref_data["descriptors"] = (
            self.model.descriptor_model.compute_features(
                templates, token_name="x_norm_clstoken"
            )
            .unsqueeze(0)
            .data
        )
        self.model.ref_data["appe_descriptors"] = (
            self.model.descriptor_model.compute_masked_patch_feature(
                templates, masks_cropped[:, 0, :, :]
            )
            .unsqueeze(0)
            .data
        )

        # Load Mesh
        mesh = trimesh.load_mesh(cad_path)
        model_points = mesh.sample(2048).astype(np.float32) / 1000.0
        self.model.ref_data["pointcloud"] = (
            torch.tensor(model_points).unsqueeze(0).data.to(self.device)
        )

    def run_inference(self, rgb_img, depth_img, batch_info, return_all=False):
        """
        Run inference, make sure to init_template first.

        Input:
            rgb_img: read with cv2.cvtColor( cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth_img: read with cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            batch_info: dict with keys: "depth", "cam_intrinsic", "depth_scale"
            return_all: If true, return detections, segmented_detections, depth_detections
                        Else, return only detections.
        Return:
            detections: Final Detection with keys: "masks" , "boxes" , "scores" , "object_ids"
            segmented_detections: RGB Segmented with keys: "masks" , "boxes"
            depth_detections: Depth Segmented with keys: "masks" , "boxes"
        """
        # Run inference
        start = time.time()

        # run inference
        detections_rgb = self.model.segmentor_model.generate_masks(np.array(rgb_img))

        # Depth segment
        processed_depth = depth_image_process(depth_img)
        detections_depth = self.model.segmentor_model.generate_masks(
            np.array(processed_depth)
        )

        # Combine RGB and Depth
        detections_combined = intersec_mask_rgbd(detections_rgb, detections_depth)

        detections_combined_with_rgb = {
            "masks": torch.cat(
                (detections_combined["masks"], detections_rgb["masks"]), dim=0
            ),
            "boxes": torch.cat(
                (detections_combined["boxes"], detections_rgb["boxes"]), dim=0
            ),
        }

        # log(0)
        # detections = Detections(detections_combined)
        detections = Detections(detections_combined_with_rgb)
        # detections.to_numpy()

        # Clone all masks detection to return
        segmented_detections = Detections(detections_rgb)
        segmented_detections.to_numpy()

        depth_detections = Detections(detections_depth)
        depth_detections.to_numpy()

        # Forward Descriptor
        query_decriptors, query_appe_descriptors = self.model.descriptor_model.forward(
            rgb_img, detections
        )

        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.model.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor = self.model.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors
        )

        # Get obj pose
        template_poses = get_obj_poses_from_template_level(
            level=2, pose_distribution="all"
        )
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        self.model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]
        image_uv = self.model.project_template_to_image(
            best_template, pred_idx_objects, batch_info, detections.masks
        )

        # Geo score
        geometric_score, visible_ratio = self.model.compute_geometric_score(
            image_uv,
            detections,
            query_appe_descriptors,
            ref_aux_descriptor,
            visible_thred=self.model.visible_thred,
        )

        # final score
        final_score = (
            semantic_score + appe_scores + geometric_score * visible_ratio
        ) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))
        detections.to_numpy()

        print(f"Finished inference in {time.time()-start}")
        if return_all:
            return detections, segmented_detections, depth_detections
        else:
            return detections

    def run_inference_no_depth(self, rgb_img, batch_info, return_all=False):
        """
        Run inference, make sure to init_template first.

        Input:
            rgb_img: read with cv2.cvtColor( cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            batch_info: dict with keys: "depth", "cam_intrinsic", "depth_scale"
            return_all: If true, return detections, and segmented_detections
                Else, return only detections.

        Return:
            detections: keys: "masks" , "boxes" , "scores" , "object_ids"
        """
        # Run inference
        start = time.time()

        # run inference
        detections_rgb = self.model.segmentor_model.generate_masks(np.array(rgb_img))

        # log(0)
        detections = Detections(detections_rgb)

        # Clone all masks detection to return
        segmented_detections = Detections(detections_rgb)
        segmented_detections.to_numpy()

        # Forard Descriptor
        query_decriptors, query_appe_descriptors = self.model.descriptor_model.forward(
            rgb_img, detections
        )

        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.model.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor = self.model.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors
        )

        # Get obj pose
        template_poses = get_obj_poses_from_template_level(
            level=2, pose_distribution="all"
        )
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        self.model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]
        image_uv = self.model.project_template_to_image(
            best_template, pred_idx_objects, batch_info, detections.masks
        )

        # Geo score
        geometric_score, visible_ratio = self.model.compute_geometric_score(
            image_uv,
            detections,
            query_appe_descriptors,
            ref_aux_descriptor,
            visible_thred=self.model.visible_thred,
        )

        # final score
        final_score = (
            semantic_score + appe_scores + geometric_score * visible_ratio
        ) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))
        detections.to_numpy()

        print(f"Finished inference in {time.time()-start}")
        if return_all:
            return detections, segmented_detections
        else:
            return detections
        
    def run_inference_depth_guide(self, rgb_img, depth_img, batch_info, rgb_segmentation, return_all=False):
        """
        Run inference, make sure to init_template first.

        Input:
            rgb_img: read with cv2.cvtColor( cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth_img: read with cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            batch_info: dict with keys: "depth", "cam_intrinsic", "depth_scale"
            rgb_segmentation: Detection
            return_all: If true, return detections, segmented_detections, depth_detections
                        Else, return only detections.
        Return:
            detections: Final Detection with keys: "masks" , "boxes" , "scores" , "object_ids"
            segmented_detections: RGB Segmented with keys: "masks" , "boxes"
            depth_detections: Depth Segmented with keys: "masks" , "boxes"
        """
        # Run inference
        start = time.time()

        # # run inference
        # rgb_segmentation.to_torch()

        # # Depth segment
        # processed_depth = depth_image_process(depth_img)
        # # detections_depth = self.model.segmentor_model.generate_masks(
        # #     np.array(processed_depth)
        # # )

        # # Combine RGB and Depth
        # detections_combined = intersec_mask_rgbd(detections_rgb, detections_depth)

        # detections_combined_with_rgb = {
        #     "masks": torch.cat(
        #         (detections_combined["masks"], detections_rgb["masks"]), dim=0
        #     ),
        #     "boxes": torch.cat(
        #         (detections_combined["boxes"], detections_rgb["boxes"]), dim=0
        #     ),
        # }

        detections_combined_with_rgb = depth_guide_merge(depth_image=depth_img, rgb_detection=rgb_segmentation.masks, device=self.device)

        # log(0)
        # detections = Detections(detections_combined)
        detections = Detections(detections_combined_with_rgb)
        # detections.to_numpy()

        # # Clone all masks detection to return
        # segmented_detections = Detections(detections_rgb)
        # segmented_detections.to_numpy()

        # depth_detections = Detections(detections_depth)
        # depth_detections.to_numpy()

        # Forward Descriptor
        query_decriptors, query_appe_descriptors = self.model.descriptor_model.forward(
            rgb_img, detections
        )

        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.model.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor = self.model.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors
        )

        # Get obj pose
        template_poses = get_obj_poses_from_template_level(
            level=2, pose_distribution="all"
        )
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        self.model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]
        image_uv = self.model.project_template_to_image(
            best_template, pred_idx_objects, batch_info, detections.masks
        )

        # Geo score
        geometric_score, visible_ratio = self.model.compute_geometric_score(
            image_uv,
            detections,
            query_appe_descriptors,
            ref_aux_descriptor,
            visible_thred=self.model.visible_thred,
        )

        # final score
        final_score = (
            semantic_score + appe_scores + geometric_score * visible_ratio
        ) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))
        detections.to_numpy()

        print(f"Finished inference in {time.time()-start}")
        if return_all:
            return detections
        else:
            return detections

    def save_detection(self, detections, save_path):
        detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)

    def save_vis_image(self, rgb, detections, save_path="tmp.png"):
        img = rgb.copy()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        colors = distinctipy.get_colors(len(detections))
        alpha = 0.33

        best_score = 0.0
        best_id = 0
        for mask_idx, det_score in enumerate(detections.scores):
            if best_score < det_score:
                best_score = det_score
                best_id = mask_idx

        best_det_mask = detections.masks[best_id]
        mask = rle_to_mask(best_det_mask)
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = detections["object_ids"][best_id]
        temp_id = obj_id - 1

        r = int(255 * colors[temp_id][0])
        g = int(255 * colors[temp_id][1])
        b = int(255 * colors[temp_id][2])
        img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
        img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
        img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]
        img[edge, :] = 255

        img = Image.fromarray(np.uint8(img))
        img.save(save_path)
        prediction = Image.open(save_path)

        # concat side by side in PIL
        img = np.array(img)
        concat = Image.new("RGB", (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        concat.save(save_path)


if __name__ == "__main__":

    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/inference/run_inference_linemod.yaml",
        help="Path to inference config yaml file",
    )

    parser.add_argument("--cuda", default=0, help="Cuda Device, default=0", type=int)

    args = parser.parse_args()

    config = load_yaml(args.config)

    # os.makedirs(os.path.join(config["OUTPUT_DIR"], "sam6d_results"), exist_ok=True)

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

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    cam_batch_info = lm_batch_input_data(
        depth_path=DEPTH_PATH, cam_path=CAM_INFO_PATH, device=device, image_id="0000"
    )
    rgb_img = Image.open(RGB_PATH).convert("RGB")
    depth_img = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)

    # init sem ism
    sen_ism = SEN_ISM(config=config, device=device)

    # init template
    sen_ism.init_template(obj_template_dir=OBJ_TEMPLATE_DIR, cad_path=CAD_PATH)

    detections = sen_ism.run_inference(
        rgb_img=rgb_img, depth_img=depth_img, batch_info=cam_batch_info
    )

    print(detections.boxes, detections.scores, detections.object_ids)

    torch.cuda.empty_cache()

    #     # Create Folder
    # obj_output_dir = os.path.join(output_dir, "sam6d_results", obj_id)
    # if not os.path.exists(obj_output_dir):
    #     os.makedirs(obj_output_dir)

    # save_path = os.path.join(obj_output_dir, f"detection_ism_{image_id}")
