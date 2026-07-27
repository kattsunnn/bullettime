# 外部ライブラリ
from PIL import ExifTags
import argparse
import copy
import numpy as np
import json
from pathlib import Path

# 自作ライブラリ
import img_utils as iu
from person_re_identification.osnet import OSNet
from omni_directional_img_utils.e2p import E2P

# bullettimeパッケージからのインポート
# pyrefly: ignore [missing-import]
from bullettime import (
    PD_YOLO,
    CropRecord,
    ReconstructionRecord,
    generate_front_ppis,
    generate_ppi_from_world_point,
    create_ppirecords,
    distance_based_nms,
    convert_ppi_record_to_crop,
    save_json,
    save_ppirecords_json,
    save_croprecords_json,
    load_croprecords_json,
    save_person_cluster_json,
    save_reconstruction_results_json,
    save_geometric_reid_process_json,
    load_extrinsics,
    validate_and_filter_clusters,
    create_reconstruction_record,
    cluster_gaze_points,
    save_gaze_rays_json,
    calculate_gaze_rays,
    filter_pairs_by_camera_id,
    remove_records_by_paths,
    find_rays_within_distance,
    visualize_geometric_reid,
)

def create_crop_records(camera_id: int, src_img, output_path,
                        *, fov_w=60, fov_h=60, overlap=0.25, range_w=90 , range_h=60,
                        input_size=1504, pd_conf=0.5, ppi_conf=0.25, gaze_idx=0,
                        dist_th=5.0) -> list[CropRecord]:
    file_name_pattern = f"camera_{camera_id:02d}"                   
    # 正面方向の透視投影画像を生成
    ppis = generate_front_ppis(src_img, fov_w, fov_h, overlap, range_w, range_h)
    ppis_raw = [ppi.get_ppi() for ppi in ppis] 
    iu.save_imgs(ppis_raw, f"{output_path}/00_ppi", f"{file_name_pattern}_{{}}")
    # 骨格検出
    pose_detector = PD_YOLO(model_path = "C:\\Users\\naoki\\Prog\\bullettime\\yolo26x-pose.pt") 
    detection_results = pose_detector.detect_pose(ppis_raw, input_size=input_size, pd_conf=pd_conf)
    plotted_ppis = pose_detector.plot_detected_poses()
    iu.save_imgs(plotted_ppis, f"{output_path}/01_plotted_ppi", f"{file_name_pattern}_{{}}")
    # 注視点の抽出（クロップ・全方位角度変換まで実行）
    ppi_records = create_ppirecords(ppis, detection_results, gaze_idx=gaze_idx, ppi_conf=ppi_conf)
    if not ppi_records: return []
    ppi_records_before_nms = copy.deepcopy(ppi_records) # コピーが残らないから必要？
    ppi_records_filterd_by_nms = distance_based_nms(ppi_records, dist_th=dist_th)
    # NMS処理前後のレコードをそれぞれ保存
    save_ppirecords_json(
        f"{output_path}/02_data/{file_name_pattern}_ppi_records_before_nms.json",
        ppi_records_before_nms,
    )
    save_ppirecords_json(
        f"{output_path}/02_data/{file_name_pattern}_ppi_records_after_nms.json",
        ppi_records_filterd_by_nms,
    )
    #局所画像出力
    crop_records: list[CropRecord] = []
    for i, ppi_record in enumerate(ppi_records_filterd_by_nms):
        save_path = f"{output_path}/03_crop/{file_name_pattern}_{i:03d}.jpg"
        ppi = ppis[ppi_record.ppi_id]
        crop_record = convert_ppi_record_to_crop(camera_id, ppi, ppi_record, save_path)
        crop_records.append(crop_record)
    return crop_records

def create_all_crop_records(src_imgs: list[np.ndarray], output_path: Path,
                          fov_w=60.0, fov_h=60.0, ppi_overlap=0.25, ppi_range_w=90, ppi_range_h=60,
                          pd_input_size=1504, pd_conf=0.5, ppi_conf=0.25, gaze_idx=0,
                          nms_dist_th=5.0) -> list[CropRecord]:
    # 局所画像群生成
    all_crop_records: list[CropRecord] = []
    for camera_id, src_img in enumerate(src_imgs):
        crop_records = create_crop_records(
            camera_id=camera_id,
            src_img=src_img,
            output_path=output_path,
            fov_w=fov_w,
            fov_h=fov_h,
            overlap=ppi_overlap,
            range_w=ppi_range_w,
            range_h=ppi_range_h,
            input_size=pd_input_size,
            pd_conf=pd_conf,
            ppi_conf=ppi_conf,
            gaze_idx=gaze_idx,
            dist_th=nms_dist_th,
        )
        all_crop_records.extend(crop_records)
    # すべての CropRecord を保存
    save_croprecords_json(
        f"{output_path}/02_data/all_crop_records.json",
        all_crop_records
    )
    return all_crop_records


def create_bullettime_from_croprecords(src_imgs: list[np.ndarray], output_path: Path, extrinsics: np.ndarray,
                                            all_crop_records: list[CropRecord],
                                            reid_eps=0.4, reid_min_sample=3, k=5, fov_w=60.0, fov_h=60.0, scale_dist=2.0,
                                            length_idx: list[int] = [9, 7, 5, 6, 8, 10]) -> list[CropRecord]:
    # Person ReId
    cropped_img_paths = [record.crop_img_path for record in all_crop_records]
    if len(cropped_img_paths) < 4:
        return all_crop_records
    person_cluster = OSNet.dbscan_by_k_reciprocal_jaccard(cropped_img_paths, eps=reid_eps, min_samples=reid_min_sample, k=k)
    save_person_cluster_json(f"{output_path}/02_data/person_cluster.json", person_cluster) # Person ReId生結果保存
    # クラスタの確認と不整合データの除外（ノイズクラスタも除外）
    person_cluster = validate_and_filter_clusters(person_cluster, all_crop_records, extrinsics)
    save_person_cluster_json(f"{output_path}/02_data/person_cluster_filtered.json", person_cluster) # フィルタ適用後のクラスタ結果保存
    # 3D Reconstruction
    reconstruction_results = {}
    src_h, src_w = src_imgs[0].shape[:2]
        
    for label, paths in person_cluster.items():
        reconstruction_results[str(label)] = create_reconstruction_record(
            paths=paths,
            all_crop_records=all_crop_records,
            extrinsics=extrinsics,
            src_w=src_w,
            src_h=src_h,
            length_idx=length_idx
        )
    save_reconstruction_results_json(f"{output_path}/02_data/reconstruction_3d.json", reconstruction_results)   
    # gaze_point_3d のクラスタリングを実行し、代表値を保存
    representative_gaze_points = cluster_gaze_points(reconstruction_results)
    save_json(f"{output_path}/02_data/representative_gaze_points.json", representative_gaze_points)

    # バレットタイム映像用画像を生成し保存
    for person_idx, pt_3d in enumerate(representative_gaze_points):
        bullettime_imgs = []
        world_point = np.array(pt_3d, dtype=float)
        for camera_id, (src_img, extrinsic) in enumerate(zip(src_imgs, extrinsics)):
            ppi = generate_ppi_from_world_point(
                world_point=world_point,
                extrinsic=extrinsic,
                src_img=src_img,
                fov_w=fov_w,
                fov_h=fov_h,
                scale_distance=scale_dist
            )
            bullettime_imgs.append(ppi)
            
        if len(bullettime_imgs) > 0:
            iu.save_imgs(bullettime_imgs, f"{output_path}/05_bullettime", f"person_{person_idx:02d}_{{}}")
    
    return all_crop_records


# 入力パラメータの設定，Osnet，スケーリング等
def generate_bullettime(src_imgs: list[np.ndarray], output_path: Path, extrinsics: np.ndarray,
                        reid_eps=0.4, reid_min_sample=3, k=5, fov_w=60.0, fov_h=60.0, scale_dist=2.0,
                        ppi_overlap=0.25, ppi_range_w=90, ppi_range_h=60, pd_input_size=1504, pd_conf=0.5,
                        ppi_conf=0.25, gaze_idx=0, length_idx=[9, 7, 5, 6, 8, 10], nms_dist_th=5.0) -> list[CropRecord]:
    # バリデーション
    if len(src_imgs) != len(extrinsics):
        raise ValueError(
            f"Mismatch between number of source images ({len(src_imgs)}) "
            f"and extrinsics ({len(extrinsics)})."
        )
    # 設定された全てのパラメータを保存
    params = {
        "reid_eps": reid_eps,
        "reid_min_sample": reid_min_sample,
        "k": k,
        "fov_w": fov_w,
        "fov_h": fov_h,
        "scale_dist": scale_dist,
        "ppi_overlap": ppi_overlap,
        "ppi_range_w": ppi_range_w,
        "ppi_range_h": ppi_range_h,
        "pd_input_size": pd_input_size,
        "pd_conf": pd_conf,
        "ppi_conf": ppi_conf,
        "gaze_idx": gaze_idx,
        "length_idx": length_idx,
        "nms_dist_th": nms_dist_th,
    }
    save_json(str(output_path / "02_data" / "parameters.json"), params)

    all_crop_records = create_all_crop_records(
        src_imgs=src_imgs,
        output_path=output_path,
        fov_w=fov_w,
        fov_h=fov_h,
        ppi_overlap=ppi_overlap,
        ppi_range_w=ppi_range_w,
        ppi_range_h=ppi_range_h,
        pd_input_size=pd_input_size,
        pd_conf=pd_conf,
        ppi_conf=ppi_conf,
        gaze_idx=gaze_idx,
        nms_dist_th=nms_dist_th
    )

    return create_bullettime_from_croprecords(
        src_imgs=src_imgs,
        output_path=output_path,
        extrinsics=extrinsics,
        all_crop_records=all_crop_records,
        reid_eps=reid_eps,
        reid_min_sample=reid_min_sample,
        k=k,
        fov_w=fov_w,
        fov_h=fov_h,
        scale_dist=scale_dist,
        length_idx=length_idx
    )


def generate_bullettime_from_croprecords(src_imgs: list[np.ndarray], output_path: Path, extrinsics: np.ndarray,
                                         crop_records_path: Path,
                                         reid_eps=0.4, reid_min_sample=3, k=5, fov_w=60.0, fov_h=60.0, scale_dist=2.0,
                                         length_idx: list[int] = [9, 7, 5, 6, 8, 10]) -> list[CropRecord]:
    # バリデーション
    if len(src_imgs) != len(extrinsics):
        raise ValueError(
            f"Mismatch between number of source images ({len(src_imgs)}) "
            f"and extrinsics ({len(extrinsics)})."
        )
    # 設定された全てのパラメータを保存
    params = {
        "reid_eps": reid_eps,
        "reid_min_sample": reid_min_sample,
        "k": k,
        "fov_w": fov_w,
        "fov_h": fov_h,
        "scale_dist": scale_dist,
        "crop_records_path": str(crop_records_path),
    }
    save_json(str(output_path / "02_data" / "parameters.json"), params)

    all_crop_records = load_croprecords_json(str(crop_records_path))

    return create_bullettime_from_croprecords(
        src_imgs=src_imgs,
        output_path=output_path,
        extrinsics=extrinsics,
        all_crop_records=all_crop_records,
        reid_eps=reid_eps,
        reid_min_sample=reid_min_sample,
        k=k,
        fov_w=fov_w,
        fov_h=fov_h,
        scale_dist=scale_dist,
        length_idx=length_idx
    )

def generate_bullettime_by_geometric_person_reid(
        src_imgs: list[np.ndarray], output_path: Path, extrinsics_path: Path, crop_records_path: Path,
        min_rays: int=3, length_idx: list[int] = [9, 7, 5, 6, 8, 10] , sim_th: float=0.75, sim_th_factor: float=0.5):
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
        # 3次元復元
        reconstruct_record = create_reconstruction_record(most_similar_pair_path, all_crop_records, extrinsics_data, src_w, src_h, length_idx) 
        ref_point = reconstruct_record.gaze_point_3d 
        ref_point_camera_ids = reconstruct_record.cameras
        dist_th = reconstruct_record.total_length * sim_th_factor
        gaze_rays = calculate_gaze_rays(crop_records, extrinsics_data)
        # 復元した3次元注視点との最小距離が閾値以下となる直線を探索
        close_rays = find_rays_within_distance(ref_point, ref_point_camera_ids, dist_th, extrinsics_data, gaze_rays)
        # person_raysのパスをリスト化して結合
        close_rays_path = [ray.path for ray in close_rays]
        person_paths = most_similar_pair_path + close_rays_path
        # 結合したパスで3次元復元を再度適用
        reconstruct_record = create_reconstruction_record(person_paths, all_crop_records, extrinsics_data, src_w, src_h, length_idx)
        gaze_points_3d.append(reconstruct_record.gaze_point_3d)
        
        # 可視化
        most_similar_records = [r for r in all_crop_records if r.crop_img_path in most_similar_pair_path]
        most_similar_rays = calculate_gaze_rays(most_similar_records, extrinsics_data)
        visualize_geometric_reid(
            most_similar_rays=most_similar_rays,
            close_rays=close_rays,
            ref_point=ref_point,
            final_gaze_point=reconstruct_record.gaze_point_3d
        )
        
        # ログ保存 
        iter_data = {
            "most_similar_pair": most_similar_pair,
            "ref_point": ref_point,
            "dist_th": dist_th,
            "close_ray_path": close_rays_path,
            "gaze_3d_point": reconstruct_record.gaze_point_3d
        }
        geometric_reid_process.append(iter_data)
        save_geometric_reid_process_json(f"{output_path}/02_data/geometric_reid_process.json", geometric_reid_process)

        # 使用済みレコード削除
        crop_records = remove_records_by_paths(crop_records, person_paths)
    
    return gaze_points_3d

def main():
    parser = argparse.ArgumentParser(description="全方位画像からバレットタイム映像を生成する")
    parser.add_argument("-i", "--input", required=True, help="入力画像のパス")
    parser.add_argument("-o", "--output", required=True, help="出力先ディレクトリ")
    parser.add_argument("-p", "--pose", required=False, help="カメラ姿勢のJSONファイルのパス (images_aggregated_pose.json)")
    parser.add_argument("-c", "--crop_records", required=False, help="CropレコードのJSONファイルのパス")
    parser.add_argument(
        "-m", "--mode", 
        choices=[
            "generate_bullettime", 
            "generate_bullettime_from_croprecords", 
            "generate_bullettime_by_geometric_person_reid",
            "create_all_crop_records"
        ], 
        default="generate_bullettime",
        help="実行モードを選択 (generate_bullettime: 通常生成, generate_bullettime_from_croprecords: 既存のCropレコードを使用, generate_bullettime_by_geometric_person_reid: 幾何学的ReID手法, create_all_crop_records: クロップ画像生成のみ)"
    )
    args = parser.parse_args()

    src_imgs = iu.load_imgs(Path(args.input)) 
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extrinsics_path = Path(args.pose) if args.pose else None
    crop_records_path = Path(args.crop_records) if args.crop_records else None

    if args.mode in ["generate_bullettime", "generate_bullettime_from_croprecords", "generate_bullettime_by_geometric_person_reid"]:
        if extrinsics_path is None:
            parser.error(f"-p/--pose is required for '{args.mode}' mode.")

    if args.mode == "generate_bullettime":
        extrinsics = load_extrinsics(extrinsics_path)
        all_crop_records = generate_bullettime(src_imgs, output_path, extrinsics)
    elif args.mode == "generate_bullettime_from_croprecords":
        if crop_records_path is None:
            parser.error("--crop_records (-c) is required for 'generate_bullettime_from_croprecords' mode.")
        extrinsics = load_extrinsics(extrinsics_path)
        all_crop_records = generate_bullettime_from_croprecords(src_imgs, output_path, extrinsics, crop_records_path)
    elif args.mode == "generate_bullettime_by_geometric_person_reid":
        if crop_records_path is None:
            parser.error("--crop_records (-c) is required for 'generate_bullettime_by_geometric_person_reid' mode.")
        all_crop_records = generate_bullettime_by_geometric_person_reid(src_imgs, output_path, extrinsics_path, crop_records_path)
    elif args.mode == "create_all_crop_records":
        all_crop_records = create_all_crop_records(src_imgs, output_path)
    
if __name__ == "__main__":
    main()