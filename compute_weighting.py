import os
from multiprocessing import Pool
from pathlib import Path
from pickle import NONE
from typing import List

import numpy as np
import pyexr
import torch
from tqdm import tqdm

from preparation.front3d.frustum import generate_frustum, generate_frustum_volume, frustum_transform, frustum_culling, \
    coords_multiplication
from preparation.front3d.prepare_3d_frustum_mask import Pad, Mask
from preparation.front3d.prepare_3d_geometry import read_sparse_distance_field_to_dense
from lib.utils.debugger import Debugger


def main():
    PATH = Path("")
    output_path = PATH / "/datasets/panoptic-reconstruction/front3d_weighting/"
    output_path.mkdir(exist_ok=True, parents=True)

    image_root_path = PATH / "/datasets/panoptic-reconstruction/front3d_2d"


    scenes = [folder for folder in tqdm(image_root_path.iterdir(), desc="Collect scenes") if folder.is_dir()]

    # scan frames
    frames: List[Path] = []
    for scene in tqdm(scenes, desc="Collect frames"):
        files = [file for file in scene.iterdir() if "distance_field" in file.name and len(file.suffixes) == 1 and not (output_path / file.parent.name / f"{file.stem.replace('distance_field', 'weighting')}.npz").exists()]
        frames.extend(files)
    print("Total frames (job)", len(frames))

    # load frustum mask
    mask = torch.from_numpy(np.load(str(PATH / "datasets/panoptic-reconstruction/front3d/frustum_mask.npz"))["mask"])
    mask = torch.logical_not(mask)

    infos = [(frame, mask, output_path) for frame in frames]

    # process
    # for info in tqdm(infos):
    #     process_sample(info)
    with Pool(12) as pool:
        for _ in tqdm(pool.imap_unordered(process_sample, infos, chunksize=1), total=len(infos), desc="Processing 3d geometry"):
            continue


def process_sample(info):
    frame, mask, output_path = info

    surface_weight = 5.0
    occluded_weight = 2.0

    # get surface
    geometry = read_sparse_distance_field_to_dense(frame, 12)
    geometry = geometry[::-1].copy()  # flip to make it consistent
    geometry = torch.from_numpy(geometry)

    surface_mask = geometry < 3
    voxel_size = 0.03

    weighting = torch.ones_like(geometry)
    weighting[surface_mask] = surface_weight

    # occlusion?
    # load depth
    depth_root_path = PATH = Path("") / "datasets/panoptic-reconstruction/front3d/"
    depth_path = depth_root_path / frame.parent.name / f"{frame.stem.replace('distance_field', 'depth')}.exr"
    try:
        depth = pyexr.read(str(depth_path)).squeeze()
        depth = torch.from_numpy(depth[::-1, ::-1].copy()).float()
    except Exception as e:
        print("ERROR while parsing", str(depth_path))
        print(e)
        return

    # define intrinsic
    intrinsic = torch.tensor([[277.1281435,   0.       , 159.5,  0.],
                                 [  0.       , 277.1281435, 119.5,  0.],
                                 [  0.       ,   0.       ,   1. ,  0.],
                                 [  0.       ,   0.       ,   0. ,  1.]])

    intrinsic_inv = np.linalg.inv(intrinsic)
    frustum_in_camera_space = generate_frustum((320, 240), intrinsic_inv, 0.4, 6.0)
    (dim_x, dim_y, dim_z), camera_to_frustum_volume = generate_frustum_volume(frustum_in_camera_space, voxel_size)
    frustum_volume_to_camera = np.linalg.inv(camera_to_frustum_volume)
    frustum = frustum_transform(frustum_in_camera_space, camera_to_frustum_volume)
    volume_coordinates = np.stack(np.ones((dim_x, dim_y, dim_z)).nonzero(), 1)
    volume_coordinates = torch.from_numpy(frustum_culling(volume_coordinates, frustum)).long()

    frustum_volume_to_image_plane = np.dot(intrinsic, frustum_volume_to_camera)
    image_coords = coords_multiplication(frustum_volume_to_image_plane, volume_coordinates)
    image_coords_x = torch.from_numpy(np.clip(np.round(image_coords[:, 0] / image_coords[:, 2]), 0, 320 - 1)).long()
    image_coords_y = torch.from_numpy(np.clip(np.round(image_coords[:, 1] / image_coords[:, 2]), 0, 240 - 1)).long()
    image_coords_z = torch.from_numpy(image_coords[:, 2])

    occlusion_mask = torch.zeros(dim_x, dim_y, dim_z).int()

    # in front of depth -> visible
    depth = depth[image_coords_y, image_coords_x]  # - depth_min
    visible = image_coords_z <= depth
    visible_coordinates_x = volume_coordinates[visible, 0]
    visible_coordinates_y = volume_coordinates[visible, 1]
    visible_coordinates_z = volume_coordinates[visible, 2]

    occlusion_mask[visible_coordinates_x, visible_coordinates_y, visible_coordinates_z] = 1

    # behind depth -> occluded
    occlusion = image_coords_z > depth
    occlusion_coordinates_x = volume_coordinates[occlusion, 0]
    occlusion_coordinates_y = volume_coordinates[occlusion, 1]
    occlusion_coordinates_z = volume_coordinates[occlusion, 2]

    occlusion_mask[occlusion_coordinates_x, occlusion_coordinates_y, occlusion_coordinates_z] = -1

    occlusion_surface_mask = (occlusion_mask == -1) & surface_mask
    occlusion_freespace_mask = (occlusion_mask == -1) & (surface_mask == False)

    weighting[occlusion_surface_mask] *= occluded_weight
    weighting[occlusion_freespace_mask] = occluded_weight

    # pad, mask
    transform = Pad([256, 256, 256], padding_value=1)
    weighting_padded = transform(weighting)
    masking = Mask(mask, 1)
    weighting_masked = masking(weighting_padded).short().numpy()

    output_file = output_path / frame.parent.name / f"{frame.stem.replace('distance_field', 'weighting')}.npz"
    output_file.parent.mkdir(exist_ok=True, parents=True)

    np.savez_compressed(str(output_file), data=weighting_masked)


if __name__ == '__main__':
    main()
