# 外部ライブラリ
import argparse
import numpy as np
from pathlib import Path

# 自作ライブラリ
import img_utils as iu
from person_re_identification.osnet import OSNet

# bullettimeパッケージからのインポート
# pyrefly: ignore [missing-import]
from bullettime import (
    CropRecord,
    generate_bullettime_images,
    save_json,
    load_croprecords_json,
    save_person_cluster_json,
    save_reconstruction_results_json,
    load_extrinsics,
    validate_and_filter_clusters,
    create_reconstruction_record,
    cluster_gaze_points,
)

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
        bullettime_imgs = generate_bullettime_images(
            world_point=pt_3d,
            extrinsics=extrinsics,
            src_imgs=src_imgs,
            fov_w=fov_w,
            fov_h=fov_h,
            scale_dist=scale_dist
        )
        if len(bullettime_imgs) > 0:
            iu.save_imgs(bullettime_imgs, f"{output_path}/05_bullettime", f"person_{person_idx:02d}_{{}}")
    
    return all_crop_records

def main():
    parser = argparse.ArgumentParser(description="全方位画像からPerson ReID手法によりバレットタイム映像を生成する")
    parser.add_argument("-i", "--input", required=True, help="入力画像のパス")
    parser.add_argument("-o", "--output", required=True, help="出力先ディレクトリ")
    parser.add_argument("-p", "--pose", required=True, help="カメラ姿勢のJSONファイルのパス (images_aggregated_pose.json)")
    parser.add_argument("-c", "--crop_records", required=True, help="CropレコードのJSONファイルのパス")
    parser.add_argument("--reid_eps", type=float, default=0.4, help="DBSCANのeps閾値 (デフォルト: 0.4)")
    parser.add_argument("--reid_min_sample", type=int, default=3, help="DBSCANのmin_samples (デフォルト: 3)")
    parser.add_argument("-k", "--k", type=int, default=5, help="k-reciprocalのk (デフォルト: 5)")
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
    extrinsics = load_extrinsics(extrinsics_path)

    # バリデーション
    if len(src_imgs) != len(extrinsics):
        raise ValueError(
            f"Mismatch between number of source images ({len(src_imgs)}) "
            f"and extrinsics ({len(extrinsics)})."
        )

    # 設定された全てのパラメータを保存
    params = {
        "reid_eps": args.reid_eps,
        "reid_min_sample": args.reid_min_sample,
        "k": args.k,
        "fov_w": args.fov_w,
        "fov_h": args.fov_h,
        "scale_dist": args.scale_dist,
        "crop_records_path": str(crop_records_path),
        "length_idx": args.length_idx,
    }
    save_json(str(output_path / "02_data" / "parameters.json"), params)

    all_crop_records = load_croprecords_json(str(crop_records_path))

    create_bullettime_from_croprecords(
        src_imgs=src_imgs,
        output_path=output_path,
        extrinsics=extrinsics,
        all_crop_records=all_crop_records,
        reid_eps=args.reid_eps,
        reid_min_sample=args.reid_min_sample,
        k=args.k,
        fov_w=args.fov_w,
        fov_h=args.fov_h,
        scale_dist=args.scale_dist,
        length_idx=args.length_idx,
    )
    
if __name__ == "__main__":
    main()