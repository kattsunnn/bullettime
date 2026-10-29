from pathlib import Path
from person_re_identification.osnet import OSNet
import numpy as np
import img_utils as iu

from bullettime import (
    load_croprecords_json,
    load_extrinsics,
    filter_pairs_by_camera_id,
    create_reconstruction_record,
    generate_bullettime_images,
    spherical_clustering_3d_points  
)

def generate_bulletime_by_multiple_gaze_points_clustering(
        src_imgs: list[np.ndarray], 
        output_path: Path,
        crop_records_path: Path,
        extrinsics_path: Path, 
        length_idx: list[int] = [9, 7, 5, 6, 8, 10],
        length_scale: float=0.5,
        min_samples: int=3,
        sim_th: float=0.75,
        fov_w: float=60,
        fov_h: float=60,
        scale_dist: float=2.0
        ):

    # read data
    crop_records = load_croprecords_json(str(crop_records_path))
    extrinsics_data = load_extrinsics(extrinsics_path)
    src_h, src_w = src_imgs[0].shape[:2]
    cropped_img_paths = [record.crop_img_path for record in crop_records]
    
    # find similar pairs
    person_pairs = OSNet.find_top_n_similar_pairs(cropped_img_paths, top_n=-1, similarity_threshold=sim_th)
    if not person_pairs:
        return
    valid_pairs = filter_pairs_by_camera_id(person_pairs, crop_records)
    if not valid_pairs:
        return
    
    # 3d reconstruction
    reconstruction_records = []
    for valid_pair in valid_pairs:
        valid_pair_path = [valid_pair[0], valid_pair[1]]
        reconstruct_record = create_reconstruction_record(
            valid_pair_path, 
            crop_records, 
            extrinsics_data, 
            src_w,
            src_h, 
            length_idx
        ) 
        reconstruction_records.append(reconstruct_record)
    
    # clustering
    radius = reconstruction_records[0].total_length * length_scale 
    points = np.vstack([record.gaze_point_3d for record in reconstruction_records])
    clusters, outliers = spherical_clustering_3d_points(points, radius, min_samples=min_samples)

    # retrieve paths and estimate new reconstruction records for each cluster
    new_reconstruction_records = []
    for cluster in clusters:
        cluster_paths = list(dict.fromkeys(
            path
            for idx in cluster.point_indices
            for path in reconstruction_records[idx].paths
        ))
        cluster_record = create_reconstruction_record(
            cluster_paths,
            crop_records,
            extrinsics_data,
            src_w,
            src_h,
            length_idx,
        )
        new_reconstruction_records.append(cluster_record)

    # generate bullettime images
    gaze_points = [record.gaze_point_3d for record in new_reconstruction_records]
    for person_idx, pt_3d in enumerate(gaze_points):
        bullettime_imgs = generate_bullettime_images(
            world_point=pt_3d,
            extrinsics=extrinsics_data,
            src_imgs=src_imgs,
            fov_w=fov_w,
            fov_h=fov_h,
            scale_dist=scale_dist
        )
        if len(bullettime_imgs) > 0:
            iu.save_imgs(bullettime_imgs, f"{output_path}/05_bullettime", f"person_{person_idx:02d}_{{}}")
            
    return