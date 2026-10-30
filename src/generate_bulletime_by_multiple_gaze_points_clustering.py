import argparse
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
    spherical_clustering_3d_points, 
    visualize_cameras_and_3d_points,
    save_json,
    is_behind_any_camera,
)

def generate_bulletime_by_multiple_gaze_points_clustering(
        src_imgs: list[np.ndarray], 
        output_path: Path,
        crop_records_path: Path,
        extrinsics_path: Path, 
        length_idx: list[int] = [9, 7, 5, 6, 8, 10],
        length_scale: float = 0.5,
        min_samples: int = 3,
        sim_th: float = 0.75,
        fov_w: float = 60.0,
        fov_h: float = 60.0,
        scale_dist: float = 2.0
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
    gaze_cand_records = []
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
        gaze_cand_records.append(reconstruct_record)
    
    # clustering
    radius = gaze_cand_records[0].total_length * length_scale 
    points = np.vstack([record.gaze_point_3d for record in gaze_cand_records])
    clusters, outliers = spherical_clustering_3d_points(points, radius, min_samples=min_samples)

    # save pairs with their indices and 3d gaze points
    pairs_data = [
        {
            "index": idx,
            "pair": [valid_pair[0], valid_pair[1]],
            "similarity": float(valid_pair[2]),
            "gaze_point_3d": record.gaze_point_3d,
        }
        for idx, (valid_pair, record) in enumerate(zip(valid_pairs, gaze_cand_records))
    ]
    save_json(f"{output_path}/02_data/person_pairs_gaze_points.json", pairs_data)

    # retrieve paths and estimate new reconstruction records for each cluster
    gaze_records = []
    gaze_records_data = []
    for cluster_id, cluster in enumerate(clusters):
        cluster_paths = list(dict.fromkeys(
            path
            for idx in cluster.point_indices
            for path in gaze_cand_records[idx].paths
        ))
        gaze_record = create_reconstruction_record(
            cluster_paths,
            crop_records,
            extrinsics_data,
            src_w,
            src_h,
            length_idx,
        )
        if is_behind_any_camera(gaze_record.gaze_point_3d, extrinsics_data): continue

        gaze_records.append(gaze_record)
        gaze_records_data.append({
            "cluster_number": cluster_id,
            "gaze_point_3d": gaze_record.gaze_point_3d,
            "centroid": cluster.centroid.tolist() if isinstance(cluster.centroid, np.ndarray) else cluster.centroid,
            "point_indices": cluster.point_indices,
            "variance": float(cluster.variance),
            "paths": cluster_paths,
        })

    # save clustering results
    outlier_paths = list(dict.fromkeys(
        path
        for idx in outliers
        for path in gaze_cand_records[idx].paths
    ))
    cluster_results_data = {
        "num_clusters": len(clusters),
        "num_outliers": len(outliers),
        "clusters": gaze_records_data,
        "outlier_indices": outliers,
        "outlier_paths": outlier_paths,
    }
    save_json(f"{output_path}/02_data/cluster_results.json", cluster_results_data)

    visualize_cameras_and_3d_points(extrinsics_data, points)

    # generate bullettime images
    gaze_points = [record.gaze_point_3d for record in gaze_records]
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

def main():
    parser = argparse.ArgumentParser(
        description="複数注視点のクラスタリングによるBullet-Time画像の生成"
    )
    parser.add_argument("-i", "--input", required=True, help="入力画像ディレクトリのパス")
    parser.add_argument("-o", "--output", required=True, help="出力先ディレクトリのパス")
    parser.add_argument("-c", "--crop_records", required=True, help="all_crop_records.jsonのパス")
    parser.add_argument("-e", "--extrinsics", required=True, help="カメラ外部パラメータJSONのパス")
    parser.add_argument(
        "--length_idx",
        type=int,
        nargs="+",
        default=[9, 7, 5, 6, 8, 10],
        help="人物の長さ計測に使用するキーポイントインデックス",
    )
    parser.add_argument(
        "--length_scale",
        type=float,
        default=0.5,
        help="クラスタリング半径のスケール係数",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=3,
        help="DBSCANクラスタリングの最小サンプル数",
    )
    parser.add_argument(
        "--sim_th",
        type=float,
        default=0.75,
        help="Re-ID類似度ペアの閾値",
    )
    parser.add_argument(
        "--fov_w",
        type=float,
        default=60.0,
        help="Bullet-Time画像の画角(横)",
    )
    parser.add_argument(
        "--fov_h",
        type=float,
        default=60.0,
        help="Bullet-Time画像の画角(縦)",
    )
    parser.add_argument(
        "--scale_dist",
        type=float,
        default=2.0,
        help="注視点からのカメラ距離スケール係数",
    )
    args = parser.parse_args()

    src_imgs = iu.load_imgs(Path(args.input))
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    generate_bulletime_by_multiple_gaze_points_clustering(
        src_imgs=src_imgs,
        output_path=output_path,
        crop_records_path=Path(args.crop_records),
        extrinsics_path=Path(args.extrinsics),
        length_idx=args.length_idx,
        length_scale=args.length_scale,
        min_samples=args.min_samples,
        sim_th=args.sim_th,
        fov_w=args.fov_w,
        fov_h=args.fov_h,
        scale_dist=args.scale_dist,
    )


if __name__ == "__main__":
    main()
