import cv2
import os
import numpy as np

from .types import PPIRecord, CropRecord
from .ppi import PPI

def create_ppirecords(
    ppis: list[PPI],
    detection_results, 
    gaze_idx: int = 0, 
    length_idx: list[int] = [2, 6, 12, 14, 16],
    ppi_conf: float = 0.25
) -> list[PPIRecord]:
    if not (0 <= gaze_idx <= 16):
        raise ValueError(f"gaze_idx must be between 0 and 16. Got: {gaze_idx}")
    for idx in length_idx:
        if not (0 <= idx <= 16):
            raise ValueError(f"length_idx items must be between 0 and 16. Got invalid index: {idx}")

    ppi_records: list[PPIRecord] = []

    for ppi_id, result in enumerate(detection_results):
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()

        # ppi_idのバリデーション
        if ppi_id not in range(len(ppis)):
            continue
        ppi = ppis[ppi_id]
        ppi_img = ppi.get_ppi()
        h, w = ppi_img.shape[:2]

        for bbox_xyxy, detection_kps, box_conf in zip(boxes, keypoints, confs):
            # bboxの信頼度チェック
            if box_conf < ppi_conf:
                continue

            # すべてのキーポイント（インデックス 0〜16 のすべて）が検出されているか確認
            if not np.all(detection_kps[:, 2] > 0):
                continue

            # gaze_idx と length_idx のキーポイントの信頼度チェック
            target_indices = [gaze_idx] + list(length_idx)
            target_kps_conf = detection_kps[target_indices, 2]
            if not np.all(target_kps_conf >= ppi_conf):
                continue

            # クロップ画像 (bbox_img) の生成と検証
            x1, y1, x2, y2 = map(int, bbox_xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            bbox_img = ppi_img[y1:y2, x1:x2].copy()

            gaze_point = detection_kps[gaze_idx]
            gaze_point_ppi = np.array(gaze_point[:2], dtype=float)
            length_points_ppi = np.array([detection_kps[idx][:2] for idx in length_idx], dtype=float)

            # gaze_point の全方位角度変換
            gaze_point_omni_deg = np.array(
                ppi.convert_ppi_point_to_src_angle_coor(
                    gaze_point_ppi[0],
                    gaze_point_ppi[1],
                ),
                dtype=float,
            )

            ppi_records.append(
                PPIRecord(
                    ppi_id=ppi_id,
                    gaze_point_ppi=gaze_point_ppi,
                    gaze_conf=float(gaze_point[2]),
                    length_points_ppi=length_points_ppi,
                    bbox_conf=float(box_conf),
                    bbox_img=bbox_img,
                    gaze_point_omni_deg=gaze_point_omni_deg,
                )
            )

    return ppi_records

def distance_based_nms(ppi_records: list[PPIRecord], dist_th: float = 5.0) -> list[PPIRecord]:
    if dist_th < 0:
        raise ValueError(f"距離の閾値 (dist_th) には 0 以上の数値を指定してください。指定値: {dist_th}")

    sorted_indices = np.array([ppi_record.gaze_conf for ppi_record in ppi_records]).argsort()[::-1]
    candidates_records = [ppi_records[idx] for idx in sorted_indices]
    selected_records: list[PPIRecord] = []

    while len(candidates_records) > 0:
        p_max = candidates_records[0]
        if len(candidates_records) == 1:
            selected_records.append(p_max)
            break
        remaining_records = candidates_records[1:]
        p_max_point = p_max.gaze_point_omni_deg[:2]
        remaining_points = np.array([record.gaze_point_omni_deg[:2] for record in remaining_records], dtype=float)
        diff = remaining_points - p_max_point
        distances = np.linalg.norm(diff, axis=1)
        keep_mask = distances >= dist_th
        # keep_maskでFalseに該当するレコードを集約
        suppressed_records = [record for record, keep in zip(remaining_records, keep_mask) if not keep]
        candidates_records = [record for record, keep in zip(remaining_records, keep_mask) if keep]
        # p_maxと集約したレコードのbbox_confを比較して，bbox_confが最大のレコードのbbox_confとbbox_imgをp_maxに置き換え
        group = [p_max] + suppressed_records
        best_bbox_record = max(
            group,
            key=lambda r: r.bbox_conf if r.bbox_conf is not None else -1.0
        )

        p_max.bbox_conf = best_bbox_record.bbox_conf
        p_max.bbox_img = best_bbox_record.bbox_img
        
        selected_records.append(p_max)

    return selected_records


def convert_ppi_record_to_crop(camera_id: int, ppi: PPI, ppi_record: PPIRecord, save_path: str) -> CropRecord:
    # 注視点のUV座標への変換
    u, v = ppi.convert_ppi_point_to_src_img_coor(ppi_record.gaze_point_ppi[0], ppi_record.gaze_point_ppi[1])
    gaze_point_omni_uv = np.array([u, v])
    # 身長（長さ）計測用のUV座標への変換
    uv_list = []
    for pt in ppi_record.length_points_ppi:
        u_pt, v_pt = ppi.convert_ppi_point_to_src_img_coor(pt[0], pt[1])
        uv_list.append([u_pt, v_pt])
    length_points_omni_uv = np.array(uv_list, dtype=float)

    # 画像の保存
    crop_img = ppi_record.bbox_img
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, crop_img)
    return CropRecord(
        camera_id=camera_id,
        crop_img_path=save_path,
        gaze_point_omni_uv=gaze_point_omni_uv,
        length_points_omni_uv=length_points_omni_uv
    )