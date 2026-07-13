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
    save_person_cluster_json,
    save_reconstruction_results_json,
    load_extrinsics,
    validate_and_filter_clusters,
    reconstruct_cluster_3d,
    cluster_gaze_points,
)

def create_crop_records(camera_id: int, src_img, output_path,
                        *, fov_w=60, fov_h=60, overlap=0.25, range_w=90 , range_h=60,
                        input_size=1504, conf=0.25, gaze_idx=0, length_idx=[2, 6, 12, 14, 16],
                        dist_th=5.0) -> list[CropRecord]:
    file_name_pattern = f"camera_{camera_id:02d}"                   
    # 正面方向の透視投影画像を生成
    ppis = generate_front_ppis(src_img, fov_w, fov_h, overlap, range_w, range_h)
    ppis_raw = [ppi.get_ppi() for ppi in ppis] 
    iu.save_imgs(ppis_raw, f"{output_path}/00_ppi", f"{file_name_pattern}_{{}}")
    # 骨格検出
    pose_detector = PD_YOLO(model_path = "C:\\Users\\naoki\\Prog\\bullettime\\yolo26x-pose.pt") 
    detection_results = pose_detector.detect_pose(ppis_raw, input_size=input_size, conf=conf)
    plotted_ppis = pose_detector.plot_detected_poses()
    iu.save_imgs(plotted_ppis, f"{output_path}/01_plotted_ppi", f"{file_name_pattern}_{{}}")
    # 注視点の抽出（クロップ・全方位角度変換まで実行）
    ppi_records = create_ppirecords(ppis, detection_results, gaze_idx=gaze_idx, length_idx=length_idx)
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

# 入力パラメータの設定，Osnet，スケーリング等
def generate_bullttime(src_imgs: list[np.ndarray], output_path: Path, extrinsics: np.ndarray,
                        eps=0.4, min_sample=2, k=5, fov_w=60.0, fov_h=60.0, scale_distance=3.0) -> list[CropRecord]:
    # バリデーション
    if len(src_imgs) != len(extrinsics):
        raise ValueError(
            f"Mismatch between number of source images ({len(src_imgs)}) "
            f"and extrinsics ({len(extrinsics)})."
        )
    # 局所画像群生成
    all_crop_records: list[CropRecord] = []
    for camera_id, src_img in enumerate(src_imgs):
        crop_records = create_crop_records(
            camera_id=camera_id,
            src_img=src_img,
            output_path=output_path
        )
        all_crop_records.extend(crop_records)
    # すべての CropRecord を保存
    save_croprecords_json(
        f"{output_path}/02_data/all_crop_records.json",
        all_crop_records
    )
    # Person ReId
    cropped_img_paths = [record.crop_img_path for record in all_crop_records]
    if len(cropped_img_paths) < 4:
        return
    person_cluster = OSNet.dbscan_by_k_reciprocal_jaccard(cropped_img_paths, eps=eps, min_samples=min_sample, k=k)
    save_person_cluster_json(f"{output_path}/02_data/person_cluster.json", person_cluster) # Person ReId生結果保存
    # クラスタの確認と不整合データの除外（ノイズクラスタも除外）
    person_cluster = validate_and_filter_clusters(person_cluster, all_crop_records, extrinsics)
    save_person_cluster_json(f"{output_path}/02_data/person_cluster_filtered.json", person_cluster) # フィルタ適用後のクラスタ結果保存
    # 3D Reconstruction
    reconstruction_results = {}
    src_h, src_w = src_imgs[0].shape[:2]
        
    for label, paths in person_cluster.items():
        reconstruction_results[str(label)] = reconstruct_cluster_3d(
            paths=paths,
            all_crop_records=all_crop_records,
            extrinsics=extrinsics,
            src_w=src_w,
            src_h=src_h
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
                scale_distance=scale_distance
            )
            bullettime_imgs.append(ppi)
            
        if len(bullettime_imgs) > 0:
            iu.save_imgs(bullettime_imgs, f"{output_path}/05_bullettime", f"person_{person_idx:02d}_{{}}")
    
    return all_crop_records
    
def main():
    parser = argparse.ArgumentParser(description="全方位画像からバレットタイム映像を生成する")
    parser.add_argument("-i", "--input", required=True, help="入力画像のパス")
    parser.add_argument("-o", "--output", required=True, help="出力先ディレクトリ")
    parser.add_argument("-p", "--pose", required=True, help="カメラ姿勢のJSONファイルのパス (images_aggregated_pose.json)")
    args = parser.parse_args()

    src_imgs = iu.load_imgs(Path(args.input)) 
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    extrinsics = load_extrinsics(Path(args.pose))

    all_crop_records = generate_bullttime(src_imgs, output_path, extrinsics)
    

if __name__ == "__main__":
    main()