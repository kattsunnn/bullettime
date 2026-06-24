# 外部ライブラリ
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

from pd_yolo_pose import PD_YOLO

@dataclass
class PPIRecord:
    ppi_id: int
    detection_point_ppi: np.ndarray | None = None
    detection_point_omni: np.ndarray | None = None
    detection_conf: float | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
@dataclass
class CropRecord:
    camera_id: int
    crop_img: np.ndarray | None = None
    detection_point_omni: np.ndarray | None = None 
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




def create_ppirecords(detection_results, keypoint_idx: int = 0) -> list[PPIRecord]:
    ppi_records: list[PPIRecord] = []

    for ppi_id, result in enumerate(detection_results):
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()

        for bbox_xyxy, detection_kps in zip(boxes, keypoints):
            detection_point = detection_kps[keypoint_idx, :]

            ppi_records.append(
                PPIRecord(
                    ppi_id=ppi_id,
                    detection_point_ppi=np.array(detection_point[:2], dtype=float),
                    detection_conf=float(detection_point[2]),
                    bbox_xyxy=tuple(float(value) for value in bbox_xyxy),
                )
            )

    return ppi_records

def add_omni_point(ppis: list[PPI], ppi_records: list[PPIRecord]) -> list[PPIRecord]:
    for ppi_record in ppi_records:
        if ppi_record.ppi_id >= len(ppis):
            continue

        ppi = ppis[ppi_record.ppi_id]
        ppi_record.detection_point_omni = np.array(
            ppi.convert_ppi_point_to_angle_coor(
                ppi_record.detection_point_ppi[0],
                ppi_record.detection_point_ppi[1],
            ),
            dtype=float,
        )

    return ppi_records

def distance_based_nms(ppi_records: list[PPIRecord], dist_th: float = 5.0) -> list[PPIRecord]:
    if ppi_records is None or len(ppi_records) == 0:
        return []
    if dist_th < 0:
        raise ValueError(f"距離の閾値 (dist_th) には 0 以上の数値を指定してください。指定値: {dist_th}")

    valid_records = [
        ppi_record
        for ppi_record in ppi_records
        if ppi_record.detection_point_omni is not None and ppi_record.detection_conf is not None
    ]
    if len(valid_records) == 0:
        return []

    sorted_indices = np.array([ppi_record.detection_conf for ppi_record in valid_records]).argsort()[::-1]
    candidates_records = [valid_records[idx] for idx in sorted_indices]
    selected_records: list[PPIRecord] = []

    while len(candidates_records) > 0:
        p_max = candidates_records[0]
        selected_records.append(p_max)
        if len(candidates_records) == 1:
            break

        remaining_records = candidates_records[1:]
        p_max_point = p_max.detection_point_omni[:2]
        remaining_points = np.array([record.detection_point_omni[:2] for record in remaining_records], dtype=float)
        diff = remaining_points - p_max_point
        distances = np.linalg.norm(diff, axis=1)
        keep_mask = distances >= dist_th
        candidates_records = [record for record, keep in zip(remaining_records, keep_mask) if keep]

    return selected_records

def print_ppirecords(ppi_records: list[PPIRecord]) -> None:
    print(f"PPIRecord count: {len(ppi_records)}")
    for ppi_record in ppi_records:
        print(
            "PPIRecord(" 
            f"ppi_id={ppi_record.ppi_id}, "
            f"detection_point_ppi={ppi_record.detection_point_ppi}, "
            f"detection_point_omni={ppi_record.detection_point_omni}, "
            f"detection_conf={ppi_record.detection_conf}, "
            f"bbox_xyxy={ppi_record.bbox_xyxy})"
        )


def ppi_record_to_dict(ppi_record: PPIRecord) -> dict:
    return {
        "ppi_id": ppi_record.ppi_id,
        "detection_point_ppi": None if ppi_record.detection_point_ppi is None else ppi_record.detection_point_ppi.tolist(),
        "detection_point_omni": None if ppi_record.detection_point_omni is None else ppi_record.detection_point_omni.tolist(),
        "detection_conf": ppi_record.detection_conf,
        "bbox_xyxy": None if ppi_record.bbox_xyxy is None else list(ppi_record.bbox_xyxy),
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

def generate_ppi(src_img, fov_w, fov_h, eye_w, eye_h, scale=1):
    src_img_w = src_img.shape[1]
    src_img_h = src_img.shape[0]
    map = get_or_create_ppi_map(src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h, scale)
    ppi = map.generate_img(src_img)
    return PPI(src_img, ppi, eye_w, eye_h)

def convert_ppi_record_to_crop(camera_id: int, ppi_img: np.ndarray, ppi_record: PPIRecord) -> CropRecord:
    if ppi_record.bbox_xyxy is None:
        return CropRecord(camera_id=camera_id, detection_point_omni=ppi_record.detection_point_omni)
    # 座標を整数に変換
    x1, y1, x2, y2 = map(int, ppi_record.bbox_xyxy)
    # 画像の範囲外アクセスを防ぐ安全処理
    h, w = ppi_img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    # 画像の切り抜き
    crop_img = ppi_img[y1:y2, x1:x2]
    
    return CropRecord(
        camera_id=camera_id,
        crop_img=crop_img,
        detection_point_omni=ppi_record.detection_point_omni
    )

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
    for record in ppi_records_filterd_by_nms:
        ppi_img = ppis_raw[record.ppi_id]
        crop_record = convert_ppi_record_to_crop(camera_id, ppi_img, record)
        crop_records.append(crop_record)
    # 局所画像デバック
    crop_imgs_to_save = [
        record.crop_img for record in crop_records 
        if record.crop_img is not None and record.crop_img.size > 0
    ]
    if len(crop_imgs_to_save) > 0:
        iu.save_imgs(crop_imgs_to_save, f"{output_path}/03_crop", f"{file_name_pattern}_{{}}")

    return crop_records

def generate_bullttime(src_imgs: list[np.ndarray], output_path: str) -> list[CropRecord]:
    all_crop_records: list[CropRecord] = []
    for camera_id, src_img in enumerate(src_imgs):
        
        crop_records = find_people_in_omni(
            camera_id=camera_id,
            src_img=src_img,
            output_path=output_path
        )
        all_crop_records.extend(crop_records)

    return None
    

def main():
    parser = argparse.ArgumentParser(description="単一画像から透視投影画像群を生成する")
    parser.add_argument("-i", "--input", required=True, help="入力画像のパス")
    parser.add_argument("-o", "--output", required=True, help="出力先ディレクトリ")
    args = parser.parse_args()

    input_path = Path(args.input)
    src_imgs = iu.load_imgs(input_path) 

    all_crop_records = generate_bullttime(src_imgs, args.output)

if __name__ == "__main__":
    main()