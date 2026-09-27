# 外部ライブラリ
import argparse
import copy
from pathlib import Path
import numpy as np

# 自作ライブラリ
import img_utils as iu

# bullettimeパッケージからのインポート
from bullettime import (
    PD_YOLO,
    CropRecord,
    generate_front_ppis,
    create_ppirecords,
    distance_based_nms,
    convert_ppi_record_to_crop,
    save_json,
    save_ppirecords_json,
    save_croprecords_json,
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
    # 局所画像出力
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
    # 設定された全てのパラメータを保存
    params = {
        "fov_w": fov_w,
        "fov_h": fov_h,
        "ppi_overlap": ppi_overlap,
        "ppi_range_w": ppi_range_w,
        "ppi_range_h": ppi_range_h,
        "pd_input_size": pd_input_size,
        "pd_conf": pd_conf,
        "ppi_conf": ppi_conf,
        "gaze_idx": gaze_idx,
        "nms_dist_th": nms_dist_th,
    }
    save_json(
        f"{output_path}/02_data/crop_parameters.json",
        params
    )

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

def main():
    parser = argparse.ArgumentParser(description="全方位画像からCropRecordおよび局所画像を生成する")
    parser.add_argument("-i", "--input", required=True, help="入力画像のパス")
    parser.add_argument("-o", "--output", required=True, help="出力先ディレクトリ")
    parser.add_argument("--fov_w", type=float, default=60.0, help="PPIの画角(横)")
    parser.add_argument("--fov_h", type=float, default=60.0, help="PPIの画角(縦)")
    parser.add_argument("--overlap", type=float, default=0.25, help="PPI生成時のオーバーラップ率")
    parser.add_argument("--range_w", type=float, default=90.0, help="PPI生成範囲(横)")
    parser.add_argument("--range_h", type=float, default=60.0, help="PPI生成範囲(縦)")
    parser.add_argument("--pd_input_size", type=int, default=1504, help="姿勢推定の入力サイズ")
    parser.add_argument("--pd_conf", type=float, default=0.5, help="姿勢推定の信頼度閾値")
    parser.add_argument("--ppi_conf", type=float, default=0.25, help="PPI注視点抽出の信頼度閾値")
    parser.add_argument("--gaze_idx", type=int, default=0, help="注視点インデックス")
    parser.add_argument("--nms_dist_th", type=float, default=5.0, help="NMSの距離閾値")
    args = parser.parse_args()

    src_imgs = iu.load_imgs(Path(args.input))
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    create_all_crop_records(
        src_imgs=src_imgs,
        output_path=output_path,
        fov_w=args.fov_w,
        fov_h=args.fov_h,
        ppi_overlap=args.overlap,
        ppi_range_w=args.range_w,
        ppi_range_h=args.range_h,
        pd_input_size=args.pd_input_size,
        pd_conf=args.pd_conf,
        ppi_conf=args.ppi_conf,
        gaze_idx=args.gaze_idx,
        nms_dist_th=args.nms_dist_th,
    )

if __name__ == "__main__":
    main()
