import warnings

warnings.filterwarnings("ignore")

import argparse
import importlib
import json
import os
import os.path as osp
import random
import sys

import cv2
import gorilla
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as transforms
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from my_custom_socket import MyServer
from Pose_Estimation_Model.lib.knn_torch import one_nn

os.chdir("./")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "Pose_Estimation_Model")

sys.path.append(os.path.join(ROOT_DIR, "provider"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "model"))
sys.path.append(os.path.join(ROOT_DIR, "model", "pointnet2"))


from data_utils import (
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
    load_im,
)
from draw_utils import draw_detections

knn = one_nn.apply
rgb_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


"""
camera.json
{"cam_K": [607.060302734375, 0.0, 639.758056640625, 0.0, 607.1031494140625, 363.29052734375, 0.0, 0.0, 1.0], "depth_scale": 0.001}
[fx, 0, cx, 0, fy, cy, 0, 0, 1]

Add lib knn
Add config inference.yaml
Add checkpoints
"""


def get_parser():
    parser = argparse.ArgumentParser(description="Pose Estimation")
    # pem
    parser.add_argument("--gpus", type=str, default="0", help="path to pretrain model")
    parser.add_argument(
        "--model", type=str, default="pose_estimation_model", help="path to model file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Pose_Estimation_Model/config/inference.yaml",
        help="path to config file, different config.yaml use different config",
    )
    parser.add_argument(
        "--iter", type=int, default=600000, help="epoch num. for testing"
    )
    parser.add_argument("--exp_id", type=int, default=0, help="")
    args_cfg = parser.parse_args()

    return args_cfg


def init_cfg():
    args = get_parser()
    exp_name = (
        args.model
        + "_"
        + osp.splitext(args.config.split("/")[-1])[0]
        + "_id"
        + str(args.exp_id)
    )
    log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus = args.gpus
    cfg.model_name = args.model
    cfg.log_dir = log_dir
    cfg.test_iter = args.iter

    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    return cfg


def sample_point_cloud(points_np, sample_size=1000):
    # Input : numpy pointclouds (N points x 3), number of pointcloud to sample (M points, default 1000)
    # Output : numpy pointclouds (M points x 3 if M<N, else N points x 3)
    num_points = points_np.shape[0]
    actual_sample_size = min(sample_size, num_points)
    sampled_points = points_np[
        np.random.choice(num_points, actual_sample_size, replace=False)
    ]
    return sampled_points


def mask_to_rle_pytorch(tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def chamfer_distance(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer Distance between two point clouds.

    :param point_cloud1: Nx3 numpy array representing the first point cloud.
    :param point_cloud2: Mx3 numpy array representing the second point cloud.
    :return: Chamfer Distance (float)
    """
    # Build KDTree for fast nearest neighbor search
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)

    # Nearest neighbor distances from point_cloud1 to point_cloud2
    distances1, _ = tree1.query(point_cloud2, k=1)
    # Nearest neighbor distances from point_cloud2 to point_cloud1
    distances2, _ = tree2.query(point_cloud1, k=1)

    # Average distances in both directions
    chamfer_dist = np.mean(distances1) + np.mean(distances2)
    return chamfer_dist


def get_scene(rgb, depth, K, image_size):
    # Recieve RGB, Depth, Cam_Intrinsic and return o3d pcd scene for visualization
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth)

    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_size[0], image_size[1], fx, fy, cx, cy
    )
    camera_intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    intrinsic.intrinsic_matrix = camera_intrinsic_matrix

    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, cam.intrinsic, cam.extrinsic
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def mask_to_cloud(depth_img, mask, K):
    # convert Binary Mask and Depth to segmented Clouds
    h, w = depth_img.shape
    i, j = np.indices((h, w))
    valid = (mask > 0) & (depth_img > 0)

    z = depth_img[valid]
    x = (j[valid] - K[0, 2]) * z / K[0, 0]
    y = (i[valid] - K[1, 2]) * z / K[1, 1]

    # Rescale based on the depth scale
    points = np.stack((x, y, z), axis=-1) / 1000

    # convert numpy points to clouds (Ready to be visualized)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([0, 0, 1])
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return point_cloud


def rle_to_mask(rle) -> np.ndarray:
    # Convert run length from SAM6D to binary mask:
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()


def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(0, 255, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    return None


def find_distance(transformed_points, to_pcl, is_sym):
    # Distance compared to the target clouds (From ground truth rotation and translation)
    if is_sym:
        pred = torch.from_numpy(transformed_points.astype(np.float32)).cuda()
        target = torch.from_numpy(to_pcl.astype(np.float32)).cuda()

        inds = knn(target, pred)
        target = torch.index_select(target, 0, inds)

        dis = torch.mean(torch.norm((pred - target), dim=1), dim=0)
    else:
        dis = np.mean(np.linalg.norm(transformed_points - to_pcl, axis=1))
    return dis


def get_xyz(depth_image, binary_mask):
    fx = 547.7678833007812
    fy = 547.7678833007812
    cx = 477.81231689453125
    cy = 271.7215270996094

    if depth_image.dtype != np.uint16:
        depth_image = depth_image.astype(np.uint16)
    mask_coords = np.argwhere(binary_mask > 0)

    valid_points = []
    for y, x in mask_coords:
        z = depth_image[y, x] / 1000.0
        if z > 0:
            x_real = (x - cx) * z / fx
            y_real = (y - cy) * z / fy
            valid_points.append([x_real, y_real, z])

    if len(valid_points) == 0:
        raise ValueError("No valid depth points found in the mask")

    avg_real_world_point = np.mean(valid_points, axis=0)
    x, y, z = avg_real_world_point.tolist()

    return [x, y, z]


def _get_template(path, cfg, tem_index=1):
    # Where path is the path to templates file ex. object1/templates/
    rgb_path = os.path.join(path, "rgb_" + str(tem_index) + ".png")
    mask_path = os.path.join(path, "mask_" + str(tem_index) + ".png")
    xyz_path = os.path.join(path, "xyz_" + str(tem_index) + ".npy")

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:, :, ::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask > 0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(
            np.arange(len(choose)), cfg.n_sample_template_point
        )
    else:
        choose_idx = np.random.choice(
            np.arange(len(choose)), cfg.n_sample_template_point, replace=False
        )
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz


def get_templates(path, cfg):
    # Where path is the path to templates file ex. object1/templates/
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
    return all_tem, all_tem_pts, all_tem_choose


def get_data(rgb, depth, cad_path, cam_path, segmented_img_list, cfg):
    cam_info = json.load(open(cam_path))
    K = np.array(cam_info["cam_K"]).reshape(3, 3)

    whole_image = rgb.astype(np.uint8)
    if len(whole_image.shape) == 2:
        whole_image = np.concatenate(
            [whole_image[:, :, None], whole_image[:, :, None], whole_image[:, :, None]],
            axis=2,
        )
    whole_depth = depth.astype(np.float32) * cam_info["depth_scale"] / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []

    for mask in segmented_img_list:
        tensor_mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
        rle_mask = mask_to_rle_pytorch(tensor_mask)

        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue

        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(
                np.arange(len(choose)), cfg.n_sample_observed_point
            )
        else:
            choose_idx = np.random.choice(
                np.arange(len(choose)), cfg.n_sample_observed_point, replace=False
            )
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        rgb = whole_image.copy()[y1:y2, x1:x2, :][:, :, ::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(
            rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR
        )
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        inst = {"scene": 0, "segmentation": rle_mask[0], "score": 1}

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(0)
        all_dets.append(inst)

    ret_dict = {}
    ret_dict["pts"] = torch.stack(all_cloud).cuda()
    ret_dict["rgb"] = torch.stack(all_rgb).cuda()
    ret_dict["rgb_choose"] = torch.stack(all_rgb_choose).cuda()
    ret_dict["score"] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict["pts"].size(0)
    ret_dict["model"] = (
        torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    )
    ret_dict["K"] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, whole_image, model_points, all_dets, all_cloud, 2 * radius


def feed_icp(
    from_pcl,
    to_pcl,
    cloud_to_process,
    pred_rot=None,
    pred_trans=None,
    init_rot=None,
    threshold=0.02,
):
    """
    Perform ICP (Iterative Closest Point) refinement between two point clouds.

    Parameters:
        from_pcl (np.ndarray): Source point cloud (N x 3).
        to_pcl (np.ndarray): Target point cloud (M x 3).
        cloud_to_process (np.ndarray): Cloud to transform after alignment.
        pred_rot (np.ndarray): Initial rotation (3x3).
        pred_trans (np.ndarray): Initial translation (3,).
        init_rot (np.ndarray): Optional extra rotation to apply after ICP.
        threshold (float): ICP convergence threshold.

    Returns:
        icped_points (np.ndarray): Transformed cloud after ICP.
        rotational_components (np.ndarray): Final rotation matrix.
        translational_components (np.ndarray): Final translation vector.
    """
    from_clouds = o3d.geometry.PointCloud()
    from_clouds.points = o3d.utility.Vector3dVector(from_pcl)

    to_clouds = o3d.geometry.PointCloud()
    to_clouds.points = o3d.utility.Vector3dVector(to_pcl)

    # Default transformation
    trans_init = np.eye(4)

    if pred_rot is not None:
        pred_rot = np.asarray(pred_rot)
    if pred_trans is not None:
        pred_trans = np.asarray(pred_trans)

    if pred_rot is not None:
        if pred_rot.ndim == 3:
            pred_rot = pred_rot[0]
    if pred_trans is not None:
        if pred_trans.ndim == 2:
            pred_trans = pred_trans[0]

    if pred_rot is not None:
        trans_init[:3, :3] = pred_rot
    if pred_trans is not None:
        trans_init[:3, 3] = pred_trans

    # Evaluate initial alignment
    evaluation = o3d.pipelines.registration.evaluate_registration(
        from_clouds, to_clouds, threshold, trans_init
    )

    if evaluation.fitness < 0.9:
        # Perform ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            from_clouds,
            to_clouds,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000),
        )

        icp_transformation = reg_p2p.transformation
        R_icp = icp_transformation[:3, :3]
        t_icp = icp_transformation[:3, 3]

        if init_rot is not None:
            # Optional initial rotation applied before ICP result
            icped_points = np.dot(cloud_to_process, init_rot)
            icped_points = np.dot(icped_points, R_icp.T) + t_icp
        else:
            icped_points = np.dot(cloud_to_process, R_icp.T) + t_icp

        return icped_points, R_icp, t_icp

    else:
        # If fitness is high enough, skip ICP and return initial transform
        R_init = trans_init[:3, :3]
        t_init = trans_init[:3, 3]
        icped_points = np.dot(cloud_to_process, R_init.T) + t_init
        return icped_points, R_init, t_init


def get_pose(
    cfg,
    template_path,
    model,
    rgb_image,
    depth_image,
    cad_path,
    cam_path,
    segmented_img_list,
):
    print("=> extracting templates")
    all_tem, all_tem_pts, all_tem_choose = get_templates(
        template_path, cfg.test_dataset
    )
    with torch.no_grad():
        all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(
            all_tem, all_tem_pts, all_tem_choose
        )

    print("=> loading input data")
    input_data, img, model_points, detections, segmented_clouds, diameter = get_data(
        rgb_image, depth_image, cad_path, cam_path, segmented_img_list, cfg.test_dataset
    )

    ninstance = input_data["pts"].size(0)
    print("=> running model")
    with torch.no_grad():
        input_data["dense_po"] = all_tem_pts.repeat(ninstance, 1, 1)
        input_data["dense_fo"] = all_tem_feat.repeat(ninstance, 1, 1)
        out = model(input_data)

    if "pred_pose_score" in out.keys():
        pose_scores = out["pred_pose_score"] * out["score"]
    else:
        pose_scores = out["score"]
    pose_scores = pose_scores.detach().cpu().numpy()
    pred_rot = out["pred_R"].detach().cpu().numpy()
    pred_trans = out["pred_t"].detach().cpu().numpy()

    return (
        model_points,
        segmented_clouds,
        detections,
        diameter,
        pose_scores,
        pred_rot,
        pred_trans,
        input_data,
        img,
    )


def surface_distance(cad_path, seg_cloud, estimated_rotation, estimated_translation):
    mesh = o3d.io.read_triangle_mesh(cad_path)
    mesh.rotate(estimated_rotation, center=(0, 0, 0))
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    rays = scene.create_rays_pinhole(
        fov_deg=120,
        center=[0, 0, 0],
        eye=[0, 0, -300],
        up=[0, 1, 0],
        width_px=600,
        height_px=600,
    )
    ans = scene.cast_rays(rays)

    hit = ans["t_hit"].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
    pcd = o3d.t.geometry.PointCloud(points)
    points_np = pcd.point.positions.numpy() / 1000

    visible_cloud = points_np + estimated_translation
    return chamfer_distance(visible_cloud, seg_cloud)


def second_refine(cad_path, pose_path, seg_cloud, mul_cloud, num_point_cloud=1000):
    res_list = []
    poses = np.load(pose_path)
    for pose in poses:
        rot = pose[:3, :3]
        # mesh.rotate(rot.T, center=(0, 0, 0)) Original
        # cam_path = pose_path.T and pose_path = cam_path.T
        mesh = o3d.io.read_triangle_mesh(cad_path)
        mesh.rotate(rot, center=(0, 0, 0))
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tensor_mesh)

        rays = scene.create_rays_pinhole(
            fov_deg=120,
            center=[0, 0, 0],
            eye=[0, 0, -300],
            up=[0, 1, 0],
            width_px=600,
            height_px=600,
        )
        ans = scene.cast_rays(rays)

        hit = ans["t_hit"].isfinite()
        points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape(
            (-1, 1)
        )
        pcd = o3d.t.geometry.PointCloud(points)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        points_np = pcd.point.positions.numpy() / 1000

        sampled_points = sample_point_cloud(points_np, sample_size=num_point_cloud)

        average_xyz = np.mean(seg_cloud, axis=0)

        sampled_points_copy = np.dot(sampled_points, rot) + average_xyz
        tf_points, rotation, translation = feed_icp(
            sampled_points_copy, seg_cloud, sampled_points_copy
        )

        R = np.dot(rot, rotation.T)
        T = np.dot(average_xyz, rotation.T) + translation
        tf_points = np.dot(points_np, R) + T

        seg_cloud_sampled = sample_point_cloud(seg_cloud, tf_points.shape[0])

        dis = chamfer_distance(tf_points, seg_cloud_sampled)

        # samp_points = sample_point_cloud(load_pc_numpy(cad_path) / 1000, sample_size=2000)
        transform_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rotation_test = np.dot(np.dot(rot.T, transform_matrix), R).T
        translation_test = T
        # test_points = np.dot(samp_points, rotation_test) + T

        res_list.append((dis, rotation_test, translation_test))

    min_index = min(range(len(res_list)), key=lambda i: res_list[i][0])
    calculated_dis, vpta_rot, vpta_trans = res_list[min_index]
    # print(calculated_dis)
    # Where hypo cloud is the hypothesis of my own that it might be correct

    calculated_clouds = np.dot(mul_cloud, vpta_rot.T) + vpta_trans
    return calculated_clouds, vpta_rot, vpta_trans


if __name__ == "__main__":
    result_path = "/home/icetenny/senior-2/results"
    sam_checkpoint = "Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth"
    pose_path = (
        "Instance_Segmentation_Model/utils/poses/predefined_poses/obj_poses_level0.npy"
    )

    cam_path = "zed2i.json"

    cfg = init_cfg()

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    with open(cam_path, "r") as file:
        camera_data = json.load(file)
    cam_K = camera_data["cam_K"]
    depth_scale = camera_data["depth_scale"]
    K = np.array(cam_K).reshape((3, 3))

    # Initialize Pose Estimation Model
    print("=> creating model")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    model = model.cuda()
    model.eval()
    gorilla.solver.load_checkpoint(model=model, filename=sam_checkpoint)

    server = MyServer(host="127.0.0.1", port=22222, server_name="SAM6D PEM Server")
    server.start()

    while True:
        recv_msg = server.wait_for_msg()
        if recv_msg is not None:
            best_mask, rgb_image, depth_image, target_obj, dataset_path_prefix = (
                recv_msg
            )

            print(f"[{server.server_name}] Target object: {target_obj}")
            print(f"[{server.server_name}] Received mask shape: {best_mask.shape}")
            print(
                f"[{server.server_name}] Received RGB shape: {rgb_image.shape}, dtype: {rgb_image.dtype}"
            )
            print(
                f"[{server.server_name}] Received depth shape: {depth_image.shape}, dtype: {depth_image.dtype}"
            )

            template_path = os.path.join(dataset_path_prefix, target_obj, "templates")
            cad_path = os.path.join(
                dataset_path_prefix, target_obj, f"{target_obj}_centered.ply"
            )

            res = get_pose(
                cfg,
                template_path,
                model,
                rgb_image,
                depth_image,
                cad_path,
                cam_path,
                segmented_img_list=[best_mask],
            )
            (
                model_points,
                segmented_clouds,
                detections,
                diameter,
                pose_scores,
                pred_rot,
                pred_trans,
                input_data,
                img,
            ) = res

            Rotation = pred_rot[0]
            Translation = pred_trans[0]
            segmented_cloud = segmented_clouds[0].detach().cpu().numpy()
            transformed_cloud = np.dot(model_points, Rotation.T) + Translation
            chamfer_dis = chamfer_distance(transformed_cloud, segmented_cloud)

            print("Object's Diameter : {0}".format(diameter))
            print("Chamfer Before : {0}".format(chamfer_dis))

            dis = surface_distance(
                cad_path=cad_path,
                seg_cloud=segmented_cloud,
                estimated_rotation=pred_rot[0],
                estimated_translation=pred_trans[0],
            )
            print("Surface distance : {0}".format(dis))

            icp_rotation = Rotation
            icp_translation = Translation
            if dis > diameter * 0.2:
                eval_points, final_rot, final_trans = second_refine(
                    cad_path=cad_path,
                    pose_path=pose_path,
                    seg_cloud=segmented_cloud,
                    mul_cloud=model_points,
                    num_point_cloud=1000,
                )
                eval_points, icp_rotation, icp_translation = feed_icp(
                    model_points,
                    segmented_cloud,
                    cloud_to_process=model_points,
                    pred_rot=final_rot,
                    pred_trans=final_trans,
                    threshold=0.001,
                )
            else:
                eval_points, icp_rotation, icp_translation = feed_icp(
                    model_points,
                    segmented_cloud,
                    cloud_to_process=model_points,
                    pred_rot=pred_rot,
                    pred_trans=pred_trans,
                    threshold=0.001,
                )

            icp_cloud = np.dot(model_points, icp_rotation.T) + icp_translation
            chamfer_dis = chamfer_distance(icp_cloud, segmented_cloud)

            print("Chamfer_After : {0}".format(chamfer_dis))
            detections[0]["score"] = float(pose_scores[0])
            detections[0]["R"] = list(icp_rotation.tolist())
            detections[0]["t"] = list(icp_translation.tolist())

            # Draw Result
            valid_masks = pose_scores == pose_scores.max()
            K = input_data["K"].detach().cpu().numpy()[valid_masks]
            result_image = draw_detections(
                img,
                pred_rot[valid_masks],
                pred_trans[valid_masks],
                model_points,
                K,
                color=(0, 255, 0),
            )

            Rounded_Rotation = np.round(icp_rotation, 6)
            Rounded_Translation = np.round(icp_translation, 6)

            server.send_response(
                msg_type_out=["numpyarray", "numpyarray", "numpyarray"],
                msg_out=[Rounded_Rotation, Rounded_Translation, result_image],
            )

            print(f"[{server.server_name}] Response Sent. Restarting.")
            server.restart()

            torch.cuda.empty_cache()
        else:
            print(f"[{server.server_name}] Connection Lost. Restarting.")
            server.restart()
