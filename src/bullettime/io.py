import json
import os
from dataclasses import asdict
import numpy as np
from pathlib import Path

from .types import PPIRecord, CropRecord, ReconstructionRecord

def save_json(file_path: str, data: any) -> str:
    """データをJSON形式でファイルに保存する共通関数"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"JSON saved to: {file_path}")
    return file_path


def save_ppirecords_json(
    save_file_path: str,
    ppi_records: list[PPIRecord],
) -> str:
    def ppi_record_to_dict(ppi_record: PPIRecord) -> dict:
        return {
            "ppi_id": ppi_record.ppi_id,
            "gaze_point_ppi": ppi_record.gaze_point_ppi.tolist(),
            "gaze_point_omni_deg": ppi_record.gaze_point_omni_deg.tolist(),
            "gaze_conf": ppi_record.gaze_conf,
            "length_points_ppi": ppi_record.length_points_ppi.tolist(),
            "bbox_conf": ppi_record.bbox_conf,
            "has_bbox_img": True,
        }

    save_data = [ppi_record_to_dict(rec) for rec in ppi_records]
    return save_json(save_file_path, save_data)


def save_croprecords_json(
    save_file_path: str,
    crop_records: list[CropRecord],
) -> str:
    def crop_record_to_dict(crop_record: CropRecord) -> dict:
        return {
            "camera_id": crop_record.camera_id,
            "crop_img_path": crop_record.crop_img_path,
            "gaze_point_omni_uv": crop_record.gaze_point_omni_uv.tolist(),
            "length_points_omni_uv": crop_record.length_points_omni_uv.tolist(),
            "gaze_point_omni_deg": crop_record.gaze_point_omni_deg.tolist(),
        }

    save_data = [crop_record_to_dict(rec) for rec in crop_records]
    return save_json(save_file_path, save_data)


def save_person_cluster_json(save_file_path: str, person_cluster: dict) -> str:
    save_data = {str(k): v for k, v in person_cluster.items()}
    return save_json(save_file_path, save_data)


def save_reconstruction_results_json(save_file_path: str, reconstruction_results: dict[str, ReconstructionRecord]) -> str:
    save_data = {label: asdict(record) for label, record in reconstruction_results.items()}
    return save_json(save_file_path, save_data)
    
# 全方位カメラの外部パラメータを読み込む
def load_extrinsics(pose_path: Path) -> np.ndarray:
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found at {pose_path}")
    with open(pose_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    extrinsics_list = []
    for cam_key, cam_val in pose_data.items():
        if isinstance(cam_val, dict) and "R" in cam_val and "t" in cam_val:
            R = np.array(cam_val["R"], dtype=float)
            t = np.array(cam_val["t"], dtype=float)
            extrinsic = np.vstack([R, t])  # shape (3, 3)
            extrinsics_list.append(extrinsic)
        else:
            print(f"Warning: 'R' or 't' not found in '{cam_key}'. Skipping.")

    if len(extrinsics_list) <= 0:
        raise ValueError(f"No valid camera extrinsics found in {pose_path}")
    extrinsics_array = np.array(extrinsics_list)
    print(f"Loaded extrinsics for {len(extrinsics_list)} camera")
    return extrinsics_array


def load_croprecords_json(file_path: str) -> list[CropRecord]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    crop_records = []
    for item in data:
        rec = CropRecord(
            camera_id=item["camera_id"],
            crop_img_path=item["crop_img_path"],
            gaze_point_omni_uv=np.array(item["gaze_point_omni_uv"], dtype=float),
            length_points_omni_uv=np.array(item["length_points_omni_uv"], dtype=float),
            gaze_point_omni_deg=np.array(item["gaze_point_omni_deg"], dtype=float)
        )
        crop_records.append(rec)
    return crop_records


def save_gaze_rays_json(save_file_path: str, gaze_rays: list[dict]) -> str:
    """gaze_raysをJSON形式で保存する"""
    def ray_to_dict(ray: dict) -> dict:
        return {
            "camera_id": ray["camera_id"],
            "origin": ray["origin"].tolist() if isinstance(ray["origin"], np.ndarray) else ray["origin"],
            "direction": ray["direction"].tolist() if isinstance(ray["direction"], np.ndarray) else ray["direction"]
        }
    
    save_data = [ray_to_dict(ray) for ray in gaze_rays]
    return save_json(save_file_path, save_data)

        