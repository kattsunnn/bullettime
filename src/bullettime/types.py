from dataclasses import dataclass
import numpy as np

@dataclass
class PPIRecord:
    ppi_id: int
    gaze_point_ppi: np.ndarray 
    gaze_conf: float
    length_points_ppi: np.ndarray 
    bbox_conf: float
    gaze_point_omni_deg: np.ndarray
    bbox_img: np.ndarray

@dataclass
class CropRecord:
    camera_id: int
    crop_img_path: str
    gaze_point_omni_uv: np.ndarray 
    length_points_omni_uv: np.ndarray
    gaze_point_omni_deg: np.ndarray


@dataclass
class ReconstructionRecord:
    gaze_point_3d: list[float]
    length_points_3d: list[list[float]]
    total_length: float
    cameras: list[int]
    paths: list[str]