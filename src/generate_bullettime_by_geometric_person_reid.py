# 外部ライブラリ
import argparse
from pathlib import Path
import numpy as np

# 自作ライブラリ
import img_utils as iu
from person_re_identification.osnet import OSNet

# bullettimeパッケージからのインポート
# pyrefly: ignore [missing-import]
from bullettime import (
    generate_bullettime_images,
    load_croprecords_json,
    save_json,
    save_geometric_reid_process_json,
    load_extrinsics,
    create_reconstruction_record,
    calculate_gaze_rays,
    filter_pairs_by_camera_id,
    remove_records_by_paths,
    find_rays_within_distance,
    visualize_geometric_reid,
    is_behind_any_camera,
)

def generate_bullettime_by_geometric_person_reid(
        src_imgs: list[np.ndarray], output_path: Path, extrinsics_path: Path, crop_records_path: Path,
        min_rays: int=3, length_idx: list[int] = [9, 7, 5, 6, 8, 10] , sim_th: float=0.75, sim_th_factor: float=0.5,
        fov_w: float=60, fov_h: float=60, scale_dist: float=2.0):
    # 設定された全てのパラメータを保存
    params = {
        "min_rays": min_rays,
        "length_idx": length_idx,
        "sim_th": sim_th,
        "sim_th_factor": sim_th_factor,
        "fov_w": fov_w,
        "fov_h": fov_h,
        "scale_dist": scale_dist,
        "extrinsics_path": str(extrinsics_path),
        "crop_records_path": str(crop_records_path),
    }
    save_json(f"{output_path}/02_data/parameters.json", params)

    # クロップレコード，外部パラメータ読み込み
    all_crop_records = load_croprecords_json(str(crop_records_path))
    crop_records = list(all_crop_records)
    extrinsics_data = load_extrinsics(extrinsics_path)
    src_h, src_w = src_imgs[0].shape[:2]
    gaze_points_3d = []
    geometric_reid_process = []

    while len(crop_records) >= min_rays:
        # 最も類似するペア特定
        cropped_img_paths = [record.crop_img_path for record in crop_records]
        person_pairs = OSNet.find_top_n_similar_pairs(cropped_img_paths, top_n=-1, similarity_threshold=sim_th)
        if not person_pairs:
            break
        valid_pairs = filter_pairs_by_camera_id(person_pairs, crop_records)
        if not valid_pairs:
            break
        most_similar_pair = max(valid_pairs, key=lambda x: x[2])
        most_similar_pair_path = [most_similar_pair[0], most_similar_pair[1]]
        # 最も類似するペアを保存
        iter_data = {
            "status": "processing",
            "most_similar_pair": {
                "path": most_similar_pair_path,
                "similarity": float(most_similar_pair[2])
            },
        }
        geometric_reid_process.append(iter_data)
        save_geometric_reid_process_json(f"{output_path}/02_data/geometric_reid_process.json", geometric_reid_process)

        # 3次元復元
        reconstruct_record = create_reconstruction_record(most_similar_pair_path, all_crop_records, extrinsics_data, src_w, src_h, length_idx) 
        ref_point = reconstruct_record.gaze_point_3d 
        # カメラ後方判定
        is_behind = is_behind_any_camera(ref_point, extrinsics_data)
        if is_behind:
            print(f"Warning: Ref point {ref_point} is behind at least one camera. Skipping this pair.")
            # is_behindであれば保存
            iter_data["status"] = "skipped_behind_camera"
            save_geometric_reid_process_json(f"{output_path}/02_data/geometric_reid_process.json", geometric_reid_process)
            crop_records = remove_records_by_paths(crop_records, most_similar_pair_path)
            continue
        ref_point_camera_ids = reconstruct_record.cameras
        dist_th = reconstruct_record.total_length * sim_th_factor
        # 注視点候補を保存
        iter_data["ref_point"] = ref_point
        iter_data["ref_point_camera_ids"] = ref_point_camera_ids
        iter_data["dist_th"] = dist_th
        save_geometric_reid_process_json(f"{output_path}/02_data/geometric_reid_process.json", geometric_reid_process)

        # 同一人物の直線を探索
        gaze_rays = calculate_gaze_rays(crop_records, extrinsics_data)
        close_rays = find_rays_within_distance(ref_point, ref_point_camera_ids, dist_th, gaze_rays)
        # person_raysのパスをリスト化して結合
        close_rays_path = [ray.path for ray in close_rays]
        person_paths = most_similar_pair_path + close_rays_path
        # 同一人物と思われる直線群を保存
        iter_data["person_paths"] = person_paths
        save_geometric_reid_process_json(f"{output_path}/02_data/geometric_reid_process.json", geometric_reid_process)

        # person_pathsの数がmin_rays未満であればスキップ
        if len(person_paths) < min_rays:
            print(f"Warning: Number of person paths ({len(person_paths)}) is less than min_rays ({min_rays}). Skipping this pair.")
            iter_data["status"] = "skipped_less_than_min_rays"
            save_geometric_reid_process_json(f"{output_path}/02_data/geometric_reid_process.json", geometric_reid_process)
            crop_records = remove_records_by_paths(crop_records, most_similar_pair_path)
            continue

        # 注視点の3次元復元
        reconstruct_record = create_reconstruction_record(person_paths, all_crop_records, extrinsics_data, src_w, src_h, length_idx)
        gaze_points_3d.append(reconstruct_record.gaze_point_3d)
        # 注視点を保存
        iter_data["gaze_3d_point"] = reconstruct_record.gaze_point_3d
        iter_data["status"] = "success"
        save_geometric_reid_process_json(f"{output_path}/02_data/geometric_reid_process.json", geometric_reid_process)

        # 可視化
        most_similar_records = [r for r in all_crop_records if r.crop_img_path in most_similar_pair_path]
        most_similar_rays = calculate_gaze_rays(most_similar_records, extrinsics_data)
        visualize_geometric_reid(
            most_similar_rays=most_similar_rays,
            close_rays=close_rays,
            ref_point=ref_point,
            final_gaze_point=reconstruct_record.gaze_point_3d
        )


        # 使用済みレコード削除
        crop_records = remove_records_by_paths(crop_records, person_paths)
    
    # バレットタイム映像用画像を生成し保存
    for person_idx, pt_3d in enumerate(gaze_points_3d):
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

    return gaze_points_3d

def main():
    parser = argparse.ArgumentParser(description="全方位画像から幾何学的ReID手法によりバレットタイム映像を生成する")
    parser.add_argument("-i", "--input", required=True, help="入力画像のパス")
    parser.add_argument("-o", "--output", required=True, help="出力先ディレクトリ")
    parser.add_argument("-p", "--pose", required=True, help="カメラ姿勢のJSONファイルのパス (images_aggregated_pose.json)")
    parser.add_argument("-c", "--crop_records", required=True, help="CropレコードのJSONファイルのパス")
    parser.add_argument("--min_rays", type=int, default=3, help="注視点探索に必要な最小視線数 (デフォルト: 3)")
    parser.add_argument("--sim_th", type=float, default=0.75, help="類似度閾値 (デフォルト: 0.75)")
    parser.add_argument("--sim_th_factor", type=float, default=0.5, help="注視点探索の距離閾値係数 (デフォルト: 0.5)")
    parser.add_argument("--fov_w", type=float, default=60.0, help="バレットタイム画像の視野角(横) (デフォルト: 60.0)")
    parser.add_argument("--fov_h", type=float, default=60.0, help="バレットタイム画像の視野角(縦) (デフォルト: 60.0)")
    parser.add_argument("--scale_dist", type=float, default=2.0, help="スケール距離 (デフォルト: 2.0)")
    parser.add_argument("--length_idx", type=int, nargs="+", default=[9, 7, 5, 6, 8, 10], help="3次元復元に使用する骨格インデックス (デフォルト: 9 7 5 6 8 10)")
    args = parser.parse_args()

    src_imgs = iu.load_imgs(Path(args.input)) 
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extrinsics_path = Path(args.pose)
    crop_records_path = Path(args.crop_records)

    generate_bullettime_by_geometric_person_reid(
        src_imgs=src_imgs,
        output_path=output_path,
        extrinsics_path=extrinsics_path,
        crop_records_path=crop_records_path,
        min_rays=args.min_rays,
        length_idx=args.length_idx,
        sim_th=args.sim_th,
        sim_th_factor=args.sim_th_factor,
        fov_w=args.fov_w,
        fov_h=args.fov_h,
        scale_dist=args.scale_dist,
    )
    
if __name__ == "__main__":
    main()