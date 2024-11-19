import blenderproc as bproc

"Import bproc first"
import os

import bpy

print("Bpy version", bpy.app.version_string)
import cv2
import numpy as np
import trimesh

# set relative path of Data folder
cad_path = "/home/icetenny/senior-1/Linemod_preprocessed/models"
output_dir = "/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/vrp_lm/templates"
bproc.init()

render_dir = os.path.dirname(os.path.abspath(__file__))


def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1 / (2 * radius)


for idx, model_file in enumerate(os.listdir(cad_path)):
    if ".ply" not in model_file:
        continue

    model_name = model_file.rstrip(".ply")
    print(
        "---------------------------"
        + str(model_name)
        + "    "
        + str(idx)
        + "-------------------------------------"
    )

    save_fpath = os.path.join(output_dir, model_name)
    if not os.path.exists(save_fpath):
        os.makedirs(save_fpath)

    obj_fpath = os.path.join(cad_path, model_file)
    if not os.path.exists(obj_fpath):
        continue

    scale = get_norm_info(obj_fpath)

    bproc.clean_up()

    obj = bproc.loader.load_obj(obj_fpath, use_legacy_obj_import=True)[0]
    obj.set_scale([scale, scale, scale])
    obj.set_cp("category_id", idx)

    # set light
    light1 = bproc.types.Light()
    light1.set_type("POINT")
    light1.set_location([0, 0, 0])
    light1.set_energy(1000)

    location = [
        [1.732, 0, 0],
        [-1.732, 0, 0],
        [0, 1.732, 0],
        [0, -1.732, 0],
        [0, 0, 1.732],
        [0, 0, -1.732],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ]

    for img_id, loc in enumerate(location):
        loc = [l * 1.25 for l in loc]

        light1.set_location([l * 2.5 for l in loc])

        # compute rotation based on vector going from location towards the location of object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            obj.get_location() - loc
        )
        # add homog cam pose based on location and rotation
        cam2world_matrix = bproc.math.build_transformation_mat(loc, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix, frame=0)

        bproc.renderer.set_max_amount_of_samples(50)
        # render the whole pipeline
        data = bproc.renderer.render()
        # render nocs
        data.update(bproc.renderer.render_nocs())

        # # save rgb images
        # for img_id in range(len(data["colors"])):

        color_bgr = data["colors"][0]
        color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath, "rgb_" + str(img_id) + ".png"), color_bgr)

        # save masks
        mask = data["nocs"][0][..., -1]
        cv2.imwrite(
            os.path.join(save_fpath, "mask_" + str(img_id) + ".png"), mask * 255
        )

        # save nocs
        # xyz = 2 * (data["nocs"][0][..., :3] - 0.5)
        # np.save(
        #     os.path.join(save_fpath, "xyz_" + str(img_id) + ".npy"),
        #     xyz.astype(np.float16),
        # )
