import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import (
    get_obj_poses_from_template_level,
    load_index_level_in_level2,
)
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23
import yaml
import sys
import time


def progressbar(it, prefix="", size=60, out=sys.stdout):  # Python3.6+
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


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.0
    for mask_idx, det in enumerate(detections):
        if best_score < det["score"]:
            best_score = det["score"]
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
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
    return concat


def batch_input_data(depth_path, cam_path, device, image_id="0001"):
    batch = {}
    cam_dict = load_yaml(cam_path)
    depth = np.array(imageio.v2.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_dict[int(image_id)]["cam_K"]).reshape((3, 3))
    depth_scale = np.array(cam_dict[int(image_id)]["depth_scale"])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch["depth_scale"] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch


def init_sam6d(segmentor_model, stability_score_thresh):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="run_inference.yaml")

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name="ISM_sam.yaml")
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name="ISM_fastsam.yaml")
    else:
        raise ValueError(
            "The segmentor_model {} is not supported now!".format(segmentor_model)
        )

    logging.info("Initializing model")
    model = instantiate(cfg.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")

    return model, device


def init_template(model, template_dir, obj_id="01"):
    logging.info(f"Initializing template OBJ:{obj_id}")
    template_dir = os.path.join(template_dir, obj_id, "templates")
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, "rgb_" + str(idx) + ".png"))
        mask = Image.open(os.path.join(template_dir, "mask_" + str(idx) + ".png"))
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
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = (
        model.descriptor_model.compute_features(templates, token_name="x_norm_clstoken")
        .unsqueeze(0)
        .data
    )
    model.ref_data["appe_descriptors"] = (
        model.descriptor_model.compute_masked_patch_feature(
            templates, masks_cropped[:, 0, :, :]
        )
        .unsqueeze(0)
        .data
    )


def run_inference(
    model, device, output_dir, data_dir, cad_dir, obj_id="01", image_id="0001"
):
    start = time.time()

    # print()
    def log(i, p=False):
        if p:
            print(i, time.time() - start)
            return time.time()

    # run inference
    # logging.info(f"Running inference OBJ:{obj_id} IMAGE:{image_id}")

    rgb = Image.open(f"{data_dir}/{obj_id}/rgb/{image_id}.png").convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    # log(0)
    detections = Detections(detections)
    start = log("Segment")
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(
        np.array(rgb), detections
    )
    start = log("Forward Desciptor")

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    start = log("Compute sem score")

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor = model.compute_appearance_score(
        best_template, pred_idx_objects, query_appe_descriptors
    )
    start = log("Compute app score")

    # compute the geometric score
    depth_path = f"{data_dir}/{obj_id}/depth/{image_id}.png"
    cam_path = f"{data_dir}/{obj_id}/info.yml"
    batch = batch_input_data(depth_path, cam_path, device, image_id=image_id)
    start = log("Batch depth input")

    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

    start = log("Get obj pose")
    mesh = trimesh.load_mesh(f"{cad_dir}/obj_{obj_id}.ply")
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = (
        torch.tensor(model_points).unsqueeze(0).data.to(device)
    )

    start = log("Load mesh")

    image_uv = model.project_template_to_image(
        best_template, pred_idx_objects, batch, detections.masks
    )

    start = log("Project template")

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv,
        detections,
        query_appe_descriptors,
        ref_aux_descriptor,
        visible_thred=model.visible_thred,
    )
    start = log("Compute geo score")

    # final score
    # logging.info(f"Saving results OBJ:{obj_id} IMAGE:{image_id}")

    final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (
        1 + 1 + visible_ratio
    )

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))

    detections.to_numpy()

    # Create Folder
    obj_output_dir = f"{output_dir}/sam6d_results/{obj_id}"
    if not os.path.exists(obj_output_dir):
        os.makedirs(obj_output_dir)

    save_path = f"{obj_output_dir}/detection_ism_{image_id}"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    # detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
    # save_json_bop23(save_path + ".json", detections)

    start = log("Save output")

    if int(image_id) % 100 == 0:
        detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])

        vis_img = visualize(rgb, detections, f"{obj_output_dir}/vis_ism_{image_id}.png")
        vis_img.save(f"{obj_output_dir}/vis_ism_{image_id}.png")


if __name__ == "__main__":

    # cam_path = f'/home/icetenny/senior-1/Linemod_preprocessed/data/01/info.yml'
    # cam_dict = load_yaml(cam_path)
    # print(cam_dict)
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname",
        default="configs/run_inference_linemod.yaml",
        help="Path to inference config yaml file",
    )

    args = parser.parse_args()

    config = load_yaml(args.fname)

    os.makedirs(f"{config['OUTPUT_DIR']}/sam6d_results", exist_ok=True)

    sam6d_model, device = init_sam6d(
        config["SEGMENTOR_MODEL"],
        stability_score_thresh=config["STABILITY_SCORE_THRESH"],
    )

    # [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    for obj in config['OBJ_ID']:
        obj_id = str(obj).zfill(2)
        num_image = len(os.listdir(f'{config["DATA_DIR"]}/{obj_id}/rgb'))
        # print(num_image)

        init_template(sam6d_model, template_dir=config["TEMPLATE_DIR"], obj_id=obj_id)

        for img in progressbar(range(0, num_image), f"Inferencing OBJ {obj}: ", 40):
            image_id = str(img).zfill(4)
            run_inference(
                sam6d_model,
                device,
                config["OUTPUT_DIR"],
                config["DATA_DIR"],
                config["CAD_DIR"],
                obj_id=obj_id,
                image_id=image_id,
            )

            torch.cuda.empty_cache()
