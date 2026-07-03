# 外部ライブラリ
from PIL import GimpPaletteFile
import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import os
import json
# 自作ライブラリ
import img_utils as iu
from omni_directional_img_utils.e2p import E2P
from omni_directional_img_utils.ppi import PPI  
from person_re_identification.osnet import OSNet
from three_d_reconstruction import reconstruct_3d_points_from_omni_directional_img, extrinsic_to_R_t, xw_to_xc

from pd_yolo_pose import PD_YOLO

@dataclass
class PPIRecord:
    ppi_id: int
    detection_point_ppi: np.ndarray | None = None
    detection_point_omni_deg: np.ndarray | None = None
    detection_conf: float | None = None
    bbox_conf: float | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    bbox_xyxy_omni_deg: tuple[float, float, float, float] | None = None
@dataclass
class CropRecord:
    camera_id: int
    crop_img_path: str | None = None
    detection_point_omni_uv: np.ndarray | None = None 
# グローバル変数
_map_cache = {}
# 透視投影画像のマップを作成．生成を効率化
def get_or_create_ppi_map(src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h, scale=1):
    cache_key = (src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h, scale)
    
    if cache_key not in _map_cache:
        map = E2P(src_img_w, src_img_h)
        map.generate_map(fov_w, fov_h, eye_w, eye_h, 0, scale)
        _map_cache[cache_key] = map
    
    return _map_cache[cache_key]

# 正面方向の透視投影画像群の生成
def generate_front_ppis(src_img, fov_w, fov_h=None, overlap=0.25, range_w=90, range_h=60):
    # 使用する全方位カメラの解像度
    src_img_w = src_img.shape[1]
    src_img_h = src_img.shape[0]
    if fov_h is None:
        fov_h = fov_w
    # 透視投影画像を生成する視線角度の設定
    step_w = fov_w * (1 - overlap)
    step_h = fov_h * (1 - overlap)
    first_step_w = fov_w / 2
    first_step_h = fov_h / 2
    eyes_w = np.arange(-range_w + first_step_w, range_w, step_w)
    eyes_h= np.arange(-range_h + first_step_h, range_h, step_h)

    ppis = []

    for eye_h in eyes_h:
        for eye_w in eyes_w:
            # 透視投影画像の生成
            map = get_or_create_ppi_map(src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h)
            ppi = map.generate_img(src_img)
            ppi = PPI(src_img, ppi, eye_w, eye_h)
            ppis.append(ppi)
    return ppis

def generate_ppi(src_img, fov_w, fov_h, eye_w, eye_h, scale=1):
    src_img_w = src_img.shape[1]
    src_img_h = src_img.shape[0]
    map = get_or_create_ppi_map(src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h, scale)
    ppi = map.generate_img(src_img)
    return PPI(src_img, ppi, eye_w, eye_h)

def create_ppirecords(detection_results, keypoint_idx: int = 0) -> list[PPIRecord]:
    ppi_records: list[PPIRecord] = []

    for ppi_id, result in enumerate(detection_results):
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()

        for bbox_xyxy, detection_kps, box_conf in zip(boxes, keypoints, confs):
            detection_point = detection_kps[keypoint_idx, :]

            ppi_records.append(
                PPIRecord(
                    ppi_id=ppi_id,
                    detection_point_ppi=np.array(detection_point[:2], dtype=float),
                    detection_conf=float(detection_point[2]),
                    bbox_conf=float(box_conf),
                    bbox_xyxy=tuple(float(value) for value in bbox_xyxy),
                )
            )

    return ppi_records

def add_omni_point(ppis: list[PPI], ppi_records: list[PPIRecord]) -> list[PPIRecord]:
    for ppi_record in ppi_records:
        if ppi_record.ppi_id >= len(ppis):
            continue

        ppi = ppis[ppi_record.ppi_id]

        if ppi_record.detection_point_ppi is not None:
            ppi_record.detection_point_omni_deg = np.array(
                ppi.convert_ppi_point_to_src_angle_coor(
                    ppi_record.detection_point_ppi[0],
                    ppi_record.detection_point_ppi[1],
                ),
                dtype=float,
            )
        if ppi_record.bbox_xyxy is not None:
            x1, y1, x2, y2 = ppi_record.bbox_xyxy
            x1_deg, y1_deg = ppi.convert_ppi_point_to_src_angle_coor(x1, y1)
            x2_deg, y2_deg = ppi.convert_ppi_point_to_src_angle_coor(x2, y2)
            ppi_record.bbox_xyxy_omni_deg = (
                float(x1_deg),
                float(y1_deg),
                float(x2_deg),
                float(y2_deg),
            )

    return ppi_records

def distance_based_nms(ppi_records: list[PPIRecord], dist_th: float = 5.0) -> list[PPIRecord]:
    if ppi_records is None or len(ppi_records) == 0:
        return []
    if dist_th < 0:
        raise ValueError(f"距離の閾値 (dist_th) には 0 以上の数値を指定してください。指定値: {dist_th}")
    # 全方位画像上の注視点とConfのNoneチェック
    valid_records = [
        ppi_record
        for ppi_record in ppi_records
        if ppi_record.detection_point_omni_deg is not None and ppi_record.detection_conf is not None
    ]
    if len(valid_records) == 0:
        return []

    sorted_indices = np.array([ppi_record.detection_conf for ppi_record in valid_records]).argsort()[::-1]
    candidates_records = [valid_records[idx] for idx in sorted_indices]
    selected_records: list[PPIRecord] = []

    while len(candidates_records) > 0:
        p_max = candidates_records[0]
        if len(candidates_records) == 1:
            selected_records.append(p_max)
            break
        remaining_records = candidates_records[1:]
        p_max_point = p_max.detection_point_omni_deg[:2]
        remaining_points = np.array([record.detection_point_omni_deg[:2] for record in remaining_records], dtype=float)
        diff = remaining_points - p_max_point
        distances = np.linalg.norm(diff, axis=1)
        keep_mask = distances >= dist_th
        # keep_maskでFalseに該当するレコードを集約
        suppressed_records = [record for record, keep in zip(remaining_records, keep_mask) if not keep]
        candidates_records = [record for record, keep in zip(remaining_records, keep_mask) if keep]
        # p_maxと集約したレコードのbbox_confを比較して，bbox_confが最大のレコードのbbox_xyxy (およびbbox_conf) をp_maxに置き換え
        group = [p_max] + suppressed_records
        best_bbox_record = max(
            group,
            key=lambda r: r.bbox_conf if r.bbox_conf is not None else -1.0
        )

        p_max.bbox_conf = best_bbox_record.bbox_conf
        p_max.bbox_xyxy = best_bbox_record.bbox_xyxy
        p_max.bbox_xyxy_omni_deg = best_bbox_record.bbox_xyxy_omni_deg
        
        selected_records.append(p_max)

    return selected_records


def convert_ppi_record_to_crop(camera_id: int, ppis: list[PPI], ppi_record: PPIRecord, save_path: str) -> CropRecord:
    if ppi_record.ppi_id >= len(ppis) or ppi_record.bbox_xyxy_omni_deg is None:
        print(f"[Debug] Skipping: ppi_id={ppi_record.ppi_id}, bbox_xyxy_omni_deg={getattr(ppi_record, 'bbox_xyxy_omni_deg', None)}")
        return None
        
    ppi_obj = ppis[ppi_record.ppi_id]
    ppi_img = ppi_obj.get_ppi()
    u, v = ppi_obj.convert_ppi_point_to_src_img_coor(ppi_record.detection_point_ppi[0], ppi_record.detection_point_ppi[1])
    detection_point_omni_uv = np.array([u, v])
    
    # bbox_xyxy_omni_deg から xyxy を取得して ppi 座標に変換
    x1_deg, y1_deg, x2_deg, y2_deg = ppi_record.bbox_xyxy_omni_deg
    x1_ppi, y1_ppi = ppi_obj.convert_src_angle_coor_to_ppi_point(x1_deg, y1_deg)
    x2_ppi, y2_ppi = ppi_obj.convert_src_angle_coor_to_ppi_point(x2_deg, y2_deg)
    h, w = ppi_img.shape[:2]
    x1, y1 = max(0, x1_ppi), max(0, y1_ppi)
    x2, y2 = min(w, x2_ppi), min(h, y2_ppi)
    # 画像の切り抜き
    crop_img = ppi_img[y1:y2, x1:x2]
    
    print(f"[Debug] camera_id: {camera_id}, ppi_id: {ppi_record.ppi_id}")
    print(f"        omni_deg: ({x1_deg:.2f}, {y1_deg:.2f}, {x2_deg:.2f}, {y2_deg:.2f})")
    print(f"        ppi_point (raw float): ({x1_ppi:.2f}, {y1_ppi:.2f}, {x2_ppi:.2f}, {y2_ppi:.2f})")
    print(f"        ppi_point (int cropped): ({x1}, {y1}, {x2}, {y2}), ppi_img_shape: (w={w}, h={h})")
    print(f"        crop_img size: {crop_img.size if crop_img is not None else 0}, save_path: {save_path}")

    # 画像の保存
    if crop_img.size > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, crop_img)
        print(f"[Debug] Saved crop image to {save_path}")
    else:
        print(f"[Debug] Crop image size is 0, not saving.")
    
    return CropRecord(
        camera_id=camera_id,
        crop_img_path=save_path,
        detection_point_omni_uv=detection_point_omni_uv
    )

def print_ppirecords(ppi_records: list[PPIRecord]) -> None:
    print(f"PPIRecord count: {len(ppi_records)}")
    for ppi_record in ppi_records:
        print(
            "PPIRecord(" 
            f"ppi_id={ppi_record.ppi_id}, "
            f"detection_point_ppi={ppi_record.detection_point_ppi}, "
            f"detection_point_omni_deg={ppi_record.detection_point_omni_deg}, "
            f"detection_conf={ppi_record.detection_conf}, "
            f"bbox_conf={ppi_record.bbox_conf}, "
            f"bbox_xyxy={ppi_record.bbox_xyxy}, "
            f"bbox_xyxy_omni_deg={ppi_record.bbox_xyxy_omni_deg})"
        )

def ppi_record_to_dict(ppi_record: PPIRecord) -> dict:
    return {
        "ppi_id": ppi_record.ppi_id,
        "detection_point_ppi": None if ppi_record.detection_point_ppi is None else ppi_record.detection_point_ppi.tolist(),
        "detection_point_omni_deg": None if ppi_record.detection_point_omni_deg is None else ppi_record.detection_point_omni_deg.tolist(),
        "detection_conf": ppi_record.detection_conf,
        "bbox_conf": ppi_record.bbox_conf,
        "bbox_xyxy": None if ppi_record.bbox_xyxy is None else list(ppi_record.bbox_xyxy),
        "bbox_xyxy_omni_deg": None if ppi_record.bbox_xyxy_omni_deg is None else list(ppi_record.bbox_xyxy_omni_deg),
    }

def save_ppirecords_json(
    output_path: str,
    file_name_pattern: str,
    *,
    added_omni_records: list[PPIRecord],
    filtered_records: list[PPIRecord],
) -> str:
    os.makedirs(f"{output_path}/02_data", exist_ok=True)
    save_file_path = f"{output_path}/02_data/{file_name_pattern}_ppi_records.json"
    save_data = {
        "added_omni": [ppi_record_to_dict(ppi_record) for ppi_record in added_omni_records],
        "filtered_by_nms": [ppi_record_to_dict(ppi_record) for ppi_record in filtered_records],
    }

    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"PPIRecord JSON saved to: {save_file_path}")
    return save_file_path

def save_person_cluster_json(output_path: str, person_cluster: dict) -> str:
    os.makedirs(f"{output_path}/02_data", exist_ok=True)
    save_file_path = f"{output_path}/02_data/person_cluster.json"
    
    # JSONキーは文字列である必要があるため、明示的に文字列に変換
    save_data = {str(k): v for k, v in person_cluster.items()}
    
    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
        
    print(f"Person cluster JSON saved to: {save_file_path}")
    return save_file_path


def save_reconstruction_results_json(output_path: str, reconstruction_results: dict) -> str:
    os.makedirs(f"{output_path}/02_data", exist_ok=True)
    save_file_path = f"{output_path}/02_data/reconstruction_3d.json"
    
    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(reconstruction_results, f, indent=4, ensure_ascii=False)
        
    print(f"3D Reconstruction results saved to: {save_file_path}")
    return save_file_path


def generate_ppi_from_world_point(world_point: np.ndarray, extrinsic: np.ndarray, src_img: np.ndarray, fov_w: float, fov_h: float, scale_distance: float = 3.0) -> np.ndarray:
    R, t = extrinsic_to_R_t(extrinsic)
    gaze_vec = xw_to_xc(world_point, R, t)
    eye_w, eye_h = E2P.gaze_vec_to_angle(gaze_vec)
    cam_distance = np.linalg.norm(gaze_vec)
    scale = cam_distance / scale_distance
    generator = E2P(src_img.shape[1], src_img.shape[0])
    generator.generate_map(fov_w, fov_h, eye_w, eye_h, 0, scale)
    ppi = generator.generate_img(src_img)
    return ppi

def find_people_in_omni(camera_id: int, src_img, output_path,
                        *, fov_w=60, fov_h=60, overlap=0.25, range_w=90 , range_h=60,
                        input_size=1504, keypoint_idx=0, dist_th=5.0) -> list[CropRecord]:
    file_name_pattern = f"camera_{camera_id:02d}"                   
    # 正面方向の透視投影画像を生成
    ppis = generate_front_ppis(src_img, fov_w, fov_h, overlap, range_w, range_h)
    ppis_raw = [ppi.get_ppi() for ppi in ppis] 
    iu.save_imgs(ppis_raw, f"{output_path}/00_ppi", f"{file_name_pattern}_{{}}")
    # 骨格検出
    pose_detector = PD_YOLO(input_size=input_size) 
    detection_results = pose_detector.detect_pose(ppis_raw)
    plotted_ppis = pose_detector.plot_detected_poses()
    iu.save_imgs(plotted_ppis, f"{output_path}/01_plotted_ppi", f"{file_name_pattern}_{{}}")
    # 注視点の抽出
    ppi_records = create_ppirecords(detection_results, keypoint_idx=keypoint_idx)
    ppi_records_added_omni = add_omni_point(ppis, ppi_records)
    ppi_records_before_nms = copy.deepcopy(ppi_records_added_omni) # コピーが残らないから必要？
    ppi_records_filterd_by_nms = distance_based_nms(ppi_records_added_omni, dist_th=dist_th)
    save_ppirecords_json(
        output_path,
        file_name_pattern,
        added_omni_records=ppi_records_before_nms,
        filtered_records=ppi_records_filterd_by_nms,
    )
    #局所画像生成
    crop_records: list[CropRecord] = []
    for i, record in enumerate(ppi_records_filterd_by_nms):
        save_path = f"{output_path}/03_crop/{file_name_pattern}_{i:03d}.jpg"
        crop_record = convert_ppi_record_to_crop(camera_id, ppis, record, save_path)
        if crop_record is not None: crop_records.append(crop_record)
    return crop_records

# 入力パラメータの設定，Osnet，スケーリング等
def generate_bullttime(src_imgs: list[np.ndarray], output_path: str, extrinsics: np.ndarray) -> list[CropRecord]:
    # 局所画像生成
    all_crop_records: list[CropRecord] = []
    for camera_id, src_img in enumerate(src_imgs):
        
        crop_records = find_people_in_omni(
            camera_id=camera_id,
            src_img=src_img,
            output_path=output_path
        )
        all_crop_records.extend(crop_records)

    # Person ReId
    cropped_img_paths = [record.crop_img_path for record in all_crop_records if record.crop_img_path is not None]
    if len(cropped_img_paths) < 4:
        return
    person_cluster, _ = OSNet.cluster_imgs_with_auto_eps(cropped_img_paths, min_samples=2)
    save_person_cluster_json(output_path, person_cluster) # Person ReId結果保存

    # 3D Reconstruction
    reconstruction_results = {}
    src_h, src_w = src_imgs[0].shape[:2]
    crop_record_map = {record.crop_img_path: record for record in all_crop_records if record.crop_img_path is not None}
        
    for label, paths in person_cluster.items():
        try:
            int_label = int(label)
        except ValueError:
            int_label = -1
        if int_label < 0 or len(paths) < 2: # クラスタに属さない or クラスタが２以下はバレットタイム生成しない
            continue
                
        extrinsics_list = []
        corr_points = []
        valid_paths = []
        camera_ids = []
            
        for path in paths:
            record = crop_record_map.get(path)
            if record is None or record.detection_point_omni_uv is None:
                continue
            camera_id = record.camera_id
                
            if camera_id < len(extrinsics):
                extrinsics_list.append(extrinsics[camera_id])
                corr_points.append([record.detection_point_omni_uv])
                valid_paths.append(path)
                camera_ids.append(camera_id)
            
        if len(extrinsics_list) >= 2:
            three_d_point = reconstruct_3d_points_from_omni_directional_img(
                np.array(extrinsics_list),
                np.array(corr_points),
                src_w,
                src_h
            )
            point_3d_arr = three_d_point.flatten()
            point_3d = point_3d_arr.tolist()
                
            reconstruction_results[str(label)] = {
                "point_3d": point_3d,
                "cameras": camera_ids,
                "paths": valid_paths
            }
            
            # バレットタイム映像生成
            bullettime_imgs = []
            for camera_id, (src_img, extrinsic) in enumerate(zip(src_imgs, extrinsics)):
                ppi = generate_ppi_from_world_point(point_3d_arr, extrinsic, src_img, fov_w=60, fov_h=60, scale_distance=3.0)
                bullettime_imgs.append(ppi)
            
            if len(bullettime_imgs) > 0:
                iu.save_imgs(bullettime_imgs, f"{output_path}/05_bullettime", f"person_{label}_{{}}")
                
    if len(reconstruction_results) > 0:
        save_reconstruction_results_json(output_path, reconstruction_results)
    
    return all_crop_records
    

def load_extrinsics(pose_path: Path, num_cameras: int) -> np.ndarray | None:
    if not pose_path.exists():
        print(f"Warning: Pose file not found at {pose_path}")
        return None
        
    with open(pose_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)
        
    extrinsics_list = []
    for i in range(num_cameras):
        cam_key = f"camera_{i:02d}"
        if cam_key in pose_data:
            R = np.array(pose_data[cam_key]["R"], dtype=float)
            t = np.array(pose_data[cam_key]["t"], dtype=float)
            extrinsic = np.vstack([R, t])  # shape (4, 3)
            extrinsics_list.append(extrinsic)
        else:
            print(f"Warning: {cam_key} not found in pose data. Skipping.")
            
    if len(extrinsics_list) <= 0:
        return None
    
    extrinsics = np.array(extrinsics_list)
    print(f"Loaded extrinsics for {len(extrinsics)} cameras. Shape: {extrinsics.shape}")
    return extrinsics
        

def main():
    parser = argparse.ArgumentParser(description="単一画像から透視投影画像群を生成する")
    parser.add_argument("-i", "--input", required=True, help="入力画像のパス")
    parser.add_argument("-o", "--output", required=True, help="出力先ディレクトリ")
    parser.add_argument("-p", "--pose", required=True, help="カメラ姿勢のJSONファイルのパス (images_aggregated_pose.json)")
    args = parser.parse_args()

    input_path = Path(args.input)
    src_imgs = iu.load_imgs(input_path) 
    extrinsics = load_extrinsics(Path(args.pose), len(src_imgs))

    all_crop_records = generate_bullttime(src_imgs, args.output, extrinsics)
    

if __name__ == "__main__":
    main()