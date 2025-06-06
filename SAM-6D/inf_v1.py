import warnings
warnings.filterwarnings("ignore")

import os
import gorilla
import argparse
import random
import os.path as osp
import sys
import numpy as np
import torch
import importlib
import torchvision.transforms as transforms
import cv2
from PIL import Image
from Pose_Estimation_Model.lib.knn_torch import one_nn
from PIL import Image
import trimesh
import json
import open3d as o3d

os.chdir("./")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, 'Pose_Estimation_Model')

sys.path.append(os.path.join(ROOT_DIR, 'provider'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, "Instance_Segmentation_Model"))


from data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from draw_utils import draw_detections

knn = one_nn.apply
rgb_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


'''
camera.json
{"cam_K": [607.060302734375, 0.0, 639.758056640625, 0.0, 607.1031494140625, 363.29052734375, 0.0, 0.0, 1.0], "depth_scale": 0.001}
[fx, 0, cx, 0, fy, cy, 0, 0, 1]

Add lib knn
Add config inference.yaml
Add checkpoints
'''


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")
    # pem
    parser.add_argument("--gpus", type=str, default="0", help="path to pretrain model")
    parser.add_argument("--model", type=str, default="pose_estimation_model", help="path to model file")
    parser.add_argument("--config", type=str, default="/workspace/SAM6D/SAM-6D/Pose_Estimation_Model/config/base.yaml", help="path to config file, different config.yaml use different config")
    parser.add_argument("--iter", type=int, default=600000, help="epoch num. for testing")
    parser.add_argument("--exp_id", type=int, default=0, help="")
    args_cfg = parser.parse_args()

    return args_cfg

def init_cfg():
    args = get_parser()
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus     = args.gpus
    cfg.model_name = args.model
    cfg.log_dir  = log_dir
    cfg.test_iter = args.iter

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    return  cfg

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

def get_scene(rgb, depth, K, image_size):
    # Recieve RGB, Depth, Cam_Intrinsic and return o3d pcd scene for visualization
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth)

    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(image_size[0], image_size[1], fx, fy, cx, cy)
    camera_intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    intrinsic.intrinsic_matrix = camera_intrinsic_matrix

    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = np.array([[1., 0., 0., 0.], [0.,1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, cam.intrinsic, cam.extrinsic)
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
    cx = 325.26110
    cy = 242.04899
    fx = 572.41140
    fy = 573.57043

    if depth_image.dtype != np.uint16:
        depth_image = depth_image.astype(np.uint16)
    mask_coords = np.argwhere(binary_mask > 0)

    valid_points = []
    for (y, x) in mask_coords:
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
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
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
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    whole_image = rgb.astype(np.uint8)
    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    whole_depth = depth.astype(np.float32) * cam_info['depth_scale'] / 1000.0
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
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        inst = {'scene':0, 'segmentation':rle_mask[0], 'score':1}

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(0)
        all_dets.append(inst)


    ret_dict = {}
    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, whole_image, model_points, all_dets, all_cloud, 2*radius

def icp(from_pcl, to_pcl, pred_rot=None, pred_trans=None, threshold=0.005):
    from_clouds = o3d.geometry.PointCloud()
    from_clouds.points = o3d.utility.Vector3dVector(from_pcl)

    to_clouds = o3d.geometry.PointCloud()
    to_clouds.points = o3d.utility.Vector3dVector(to_pcl)

    if ((pred_rot is None) and (pred_trans is None)):
        trans_init = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    elif pred_rot is None:
        trans_init = np.array([[1., 0., 0., pred_trans[0]], [0., 1., 0., pred_trans[1]], [0., 0., 1., pred_trans[2]], [0., 0., 0., 1.]])
    elif pred_trans is None:
        trans_init = np.asarray([[pred_rot[0][0], pred_rot[0][1], pred_rot[0][2], 0], 
                                [pred_rot[1][0], pred_rot[1][1], pred_rot[1][2], 0], 
                                [pred_rot[2][0], pred_rot[2][1], pred_rot[2][2], 0], 
                                [0, 0, 0, 1]])
    else:
        trans_init = np.asarray([[pred_rot[0][0], pred_rot[0][1], pred_rot[0][2], pred_trans[0]], 
                                [pred_rot[1][0], pred_rot[1][1], pred_rot[1][2], pred_trans[1]], 
                                [pred_rot[2][0], pred_rot[2][1], pred_rot[2][2], pred_trans[2]], 
                                [0, 0, 0, 1]])

    evaluation = o3d.pipelines.registration.evaluate_registration(from_clouds, to_clouds, threshold, trans_init)

    if evaluation.fitness < 1:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            from_clouds, to_clouds, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), 
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))

        icp_transformation = reg_p2p.transformation

        rotational_components = icp_transformation[:3, :3]
        translational_components = icp_transformation[:3, 3]
        return rotational_components, translational_components


if __name__ == "__main__":
    result_path = '/workspace/SAM6D/SAM-6D/'
    sam_checkpoint = '/workspace/SAM6D/SAM-6D/Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth'

    main_dir = '/workspace/SAM6D/SAM-6D/'
    
    # Defining important inputs
    cam_path = '/workspace/SAM6D/SAM-6D/camera.json'
    template_path = '/workspace/SAM6D/SAM-6D/Data/templates/'+ '08' +'/templates'
    cad_path = '/workspace/Linemod_preprocessed/models/obj_' + '08' + '.ply'

    # Things feeded from each Scene
    rgb_path = '/workspace/Linemod_preprocessed/data/' + '08' + '/rgb/0100.png'
    depth_path = '/workspace/Linemod_preprocessed/data/' + '08' + '/depth/0100.png'
    segmented_path_list = ['/workspace/Linemod_preprocessed/data/' + '08' + '/mask/0100.png']

    # Assuming rgb, depth, and segmented masks are feeded as numpy arrays
    rgb_image = load_im(rgb_path)
    depth_image = load_im(depth_path)

    segmented_img_list = []
    for directory in segmented_path_list:
        mask_image = Image.open(directory).convert("1")
        mask = np.array(mask_image, dtype=np.uint8)
        segmented_img_list.append(mask)


    cfg = init_cfg()

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    with open(cam_path, 'r') as file:
        camera_data = json.load(file)
    cam_K = camera_data['cam_K']
    depth_scale = camera_data['depth_scale']
    K = np.array(cam_K).reshape((3, 3))

    # Initialize Pose Estimation Model
    # print("=> creating model")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    model = model.cuda()
    model.eval()
    gorilla.solver.load_checkpoint(model=model, filename=sam_checkpoint)

    # Feature Extraction of templates
    # print("=> extracting templates")
    # tem_path = 'Path to template file ex. object1/templates/'
    all_tem, all_tem_pts, all_tem_choose = get_templates(template_path, cfg.test_dataset)
    with torch.no_grad():
        all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    # Retrieving Scene data for Pose Estimation Model
    # print("=> loading input data")
    input_data, img, model_points, detections, segmented_clouds, diameter = get_data(
        rgb_image, depth_image, cad_path, cam_path, 
        segmented_img_list, cfg.test_dataset)
    
    ninstance = input_data['pts'].size(0)

    # Pose Estimation
    # print("=> running model")
    with torch.no_grad():
        input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
        input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
        out = model(input_data)

    if 'pred_pose_score' in out.keys():
        pose_scores = out['pred_pose_score'] * out['score']
    else:
        pose_scores = out['score']
    pose_scores = pose_scores.detach().cpu().numpy()
    pred_rot = out['pred_R'].detach().cpu().numpy()
    pred_trans = out['pred_t'].detach().cpu().numpy()

    for idx, det in enumerate(detections):
        Rotation = pred_rot[idx]
        Translation = pred_trans[idx]

        segmented_cloud = segmented_clouds[idx].detach().cpu().numpy()
        transformed_cloud = np.dot(model_points, Rotation.T) + Translation
        distance = find_distance(transformed_cloud, segmented_cloud, False)

        if distance >= diameter:
            avg_cloud = get_xyz(depth_image, segmented_img_list[idx])
            icp_Rotation, icp_Translation = icp(from_pcl=model_points, to_pcl=segmented_cloud, pred_trans=avg_cloud)
        else:
            icp_Rotation, icp_Translation = icp(from_pcl=model_points, to_pcl=segmented_cloud, pred_rot=Rotation, pred_trans=Translation)

        # icp_cloud = np.dot(model_points, icp_Rotation.T) + icp_Translation
        # distance = find_distance(icp_cloud, segmented_cloud, False)

        detections[idx]['score'] = float(pose_scores[idx])
        detections[idx]['R'] = list(icp_Rotation.tolist())
        detections[idx]['t'] = list(icp_Translation.tolist())

        # print(detections[idx]['R'])
        # print(detections[idx]['t'])

    # # Save as JSON
    # print("=> saving results")
    # os.makedirs(result_path, exist_ok=True)
    # with open(os.path.join(result_path, 'detection_pem.json'), "w") as f:
    #     json.dump(detections, f)

    # Save Predicted Image
    # print("=> Saving Image ...")
    save_path = os.path.join(result_path, 'vis_pem.png')
    valid_masks = pose_scores == pose_scores.max()
    K = input_data['K'].detach().cpu().numpy()[valid_masks]
    vis_img = visualize(img, pred_rot[valid_masks], pred_trans[valid_masks]*1000, model_points*1000, K, save_path)

    # # Visualize
    # rgb_img = np.array(Image.open(rgb_path))
    # depth_img = np.array(Image.open(depth_path))

    # scene = get_scene(rgb_img, depth_img, K, (640, 480))
    # interested_clouds = o3d.geometry.PointCloud()
    # # predicted_clouds = o3d.geometry.PointCloud()
    # predicted_meshes = [] 

    # print("Number of proposal : "+str(len(detections)))
    # for data in detections:
    #     mask_img = rle_to_mask(data['segmentation'])
    #     interested_cloud = mask_to_cloud(depth_img, mask_img, K)

    #     R = np.array(data['R'])
    #     t = np.array(data['t'])

    #     predicted_mesh = o3d.io.read_triangle_mesh(cad_path)
    #     predicted_mesh.scale(1 / 1000, center=(0, 0, 0))

    #     transformation_matrix = np.eye(4)
    #     transformation_matrix[:3, :3] = R
    #     transformation_matrix[:3, 3] = t 

    #     predicted_mesh.transform(transformation_matrix)
    #     predicted_mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #     predicted_meshes.append(predicted_mesh)

    #     # predicted_points = model_points.dot(R.T) + t

    #     # predicted_cloud = o3d.geometry.PointCloud()
    #     # predicted_cloud.points = o3d.utility.Vector3dVector(predicted_points)
    #     # predicted_cloud.paint_uniform_color([1, 0, 0])
    #     # predicted_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #     interested_clouds += interested_cloud

    # # o3d.visualization.draw_geometries([scene, interested_clouds] + predicted_meshes)
    # o3d.visualization.draw_geometries([scene] + predicted_meshes)